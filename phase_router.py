"""
PhaseRouter — litellm pre-call hook that auto-switches models by task phase
AND difficulty.

Loaded via litellm_settings.callbacks in <CONFIG_YAML>.
Classifies each request (keyword + structural, ~0ms, no LLM call) and rewrites
data["model"] before the router selects a deployment. Existing fallback ladders
per-model are unaffected.

Two-layer classification:
  1. Phase: what kind of task? (thinking, planning, orchestration, coding, lookup)
  2. Difficulty: how hard? (easy, hard) — only for thinking/planning/coding

Easy tasks → <EASY_MODEL> (free tier)
Hard tasks → <HARD_CODING_MODEL> / <HARD_PLANNING_MODEL> (paid, but only when genuinely needed)

Logs every routing decision to <LOG_PATH>.
"""

import re
import os
import time
import logging
from typing import Any, Dict, List, Optional, Union

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.caching.caching import DualCache

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "<LOG_PATH>")
_logger = logging.getLogger("phase_router")
_logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(_LOG_PATH)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
_logger.addHandler(_fh)
_logger.propagate = False

# ---------------------------------------------------------------------------
# Phase + Difficulty → model mapping
# ---------------------------------------------------------------------------
# Cost-aware routing (2026-08-09):
#   - <EASY_MODEL> is FREE (unlimited) — use for all easy tasks + lookup
#   - <HARD_CODING_MODEL> is PAID — use only for hard thinking/coding
#   - <HARD_PLANNING_MODEL> is PAID — use only for hard planning
#   - <ORCHESTRATION_MODEL> is FREE — always for orchestration (agent deployment)
#   - <LONG_CONTEXT_MODEL> — long context fallback
#
# Difficulty escalation is conservative: need ≥1 hard signal AND user intent
# (hard keyword, quality-qualifier, or terse directive) AND more hard than easy
# signals to escalate. A lone context signal only escalates with user intent and
# not an easy reply. False positives waste money; false negatives just mean
# <EASY_MODEL> handles it (which is fine — it is strong).

PHASE_MAP: Dict[str, str] = {
    # Easy tasks → <EASY_MODEL> (free, unlimited)
    "thinking":         "<EASY_MODEL>",            # easy thinking → <EASY_MODEL>
    "planning":         "<EASY_MODEL>",            # easy planning → <EASY_MODEL>
    "coding":           "<EASY_MODEL>",            # easy coding → <EASY_MODEL>
    "lookup":           "<EASY_MODEL>",            # always <EASY_MODEL> (fast, free)
    "default":          "<EASY_MODEL>",            # fallback → <EASY_MODEL>

    # Hard tasks → paid models (only when difficulty escalates)
    "thinking_hard":    "<HARD_CODING_MODEL>",  # hard thinking → <HARD_CODING_MODEL>
    "planning_hard":    "<HARD_PLANNING_MODEL>",   # hard planning → <HARD_PLANNING_MODEL>
    "coding_hard":      "<HARD_CODING_MODEL>",  # hard coding → <HARD_CODING_MODEL>

    # Orchestration → always <ORCHESTRATION_MODEL> (free, agent deployment design center)
    "orchestration":    "<ORCHESTRATION_MODEL>",

    # Long context → <LONG_CONTEXT_MODEL> (1M ctx)
    "long_context":     "<LONG_CONTEXT_MODEL>",
}

# Models temporarily skipped because they're emitting malformed output.
_SKIP_MODELS = {
    "<SKIPPED_MODEL>",  # emits tool-call syntax as raw content, not JSON tool_calls
}

# Fallback chain — when primary fails (timeout, malformed tool call, or
# downstream HTTP 5xx), litellm's cooldown layer consults this.
# Format: primary_model -> [fallback1, fallback2, fallback3]
FALLBACKS: Dict[str, List[str]] = {
    "<HARD_CODING_MODEL>":    ["<EASY_MODEL>",          "<FALLBACK_MODEL_D>",     "<FALLBACK_MODEL_B>"],
    "<ORCHESTRATION_MODEL>":         ["<EASY_MODEL>",          "<FALLBACK_MODEL_D>",     "<FALLBACK_MODEL_B>"],
    "<EASY_MODEL>":              ["<EASY_MODEL_FALLBACK>",              "<FALLBACK_MODEL_D>",     "<HARD_CODING_MODEL>"],
    "<EASY_MODEL_FALLBACK>":                  ["<FALLBACK_MODEL_D>",      "<ORCHESTRATION_MODEL>",   "<HARD_CODING_MODEL>"],
    "<HARD_PLANNING_MODEL>":     ["<HARD_PLANNING_FALLBACK>",       "<EASY_MODEL>",        "<HARD_CODING_MODEL>"],
    "<HARD_PLANNING_FALLBACK>":           ["<HARD_PLANNING_MODEL>", "<EASY_MODEL>",        "<HARD_CODING_MODEL>"],
}

# Models that are "default" from Claude Code — safe to rewrite.
_REWRITABLE_MODELS = {
    "<EASY_MODEL>",
    "<EASY_MODEL_FALLBACK>",
    "<OPUS_ALIAS>",
    "<OPUS_MODEL>",
}

# Models known to support Anthropic-format tool calling.
_TOOL_CAPABLE_MODELS = {
    "<EASY_MODEL>",
    "<EASY_MODEL_FALLBACK>",
    "<ORCHESTRATION_MODEL>",
    "<FAST_MODEL_A>",
    "<FAST_MODEL_B>",
    "<FALLBACK_MODEL_B>",
    "<FALLBACK_MODEL_C>",
    "<FALLBACK_MODEL_A>",
    "<FALLBACK_MODEL_E>",
    "<FALLBACK_MODEL_D>",
    "<LONG_CONTEXT_MODEL>",
    "<HARD_PLANNING_MODEL>",
    "<HARD_PLANNING_FALLBACK>",
    "<HARD_CODING_MODEL>",
}

# ---------------------------------------------------------------------------
# Keyword sets for phase classification
# ---------------------------------------------------------------------------

_THINKING_KW = re.compile(
    r"\b(why|explain|explanation|analy[sz]e|reason|design|architect|investigat|understand|"
    r"deep\s*think|consider|evaluat|assess|implications?|trade-?offs?|"
    r"consequences?|theoretic|prov|deduc|infer|conclud|"
    r"compar|contrast|critiqu|review|exam|explor|concept|"
    r"philosoph|abstract|fundamental|principle|mechanism|"
    r"caus|correlat|depend|interact|dynamics?|emergenc|"
    r"trace|diagnos|root\s*cause|walk\s+through|break\s+down)\b",
    re.IGNORECASE,
)

_PLANNING_KW = re.compile(
    r"\b(plan|planning|approach|strategy|steps?|roadmap|outline|blueprint|"
    r"scaffold|structure|decompos|break\s*down|milestones?|phases?|"
    r"sequenc|prioriti[sz]|order\s*of|workflow|pipeline|staged|"
    r"architect|design\s+pattern|system\s+design|high\s*level|"
    r"technical\s+design|specific|comprehensive\s+plan|"
    r"migration|rollout|deployment\s+plan|release\s+plan)\b",
    re.IGNORECASE,
)

_ORCHESTRATION_KW = re.compile(
    r"\b(spawn|delegate|parallel|fan\s*out|orchestrat|coordinat|dispatch|"
    r"subagent|sub-?agent|workflow|multi-?agent|distribut|scatter|"
    r"batch|fleet|swarm|pipeline\s*of|"
    r"background|async|concurrent|workers?|threads?|"
    r"task\s+queue|job\s+queue|work\s+queue|"
    r"parallel\s+agents?|multiple\s+agents?|agent\s+team)\b",
    re.IGNORECASE,
)

_CODING_KW = re.compile(
    r"\b(implement|fix|refactor|write\s+code|code\s+review|debug|patch|"
    r"function|method|class|variable|import|export|compile|lint|"
    r"test|unit\s*test|integration\s*test|coverage|type\s*hint|"
    r"regex|algorithm|optimi[sz]|perf|benchmark|profile|"
    r"api|endpoint|route|handler|middleware|serializer|"
    r"database|query|sql|migration|schema|model|"
    r"frontend|backend|fullstack|deploy|docker|kubernetes|"
    r"git|commit|branch|merge|pull\s+request|pr\s+review|"
    r"script|bash|shell|python|javascript|typescript|java|"
    r"engineer|reverse\s+engineer|decompil|disassembl|binary|"
    r"production\s+ready|error\s+handling|failure\s+mode|"
    r"build|make|create|generate|scaffold)\b",
    re.IGNORECASE,
)

_LOOKUP_KW = re.compile(
    r"\b(what\s+is|what\s+are|find|search|list|grep|where|which|"
    r"show\s+me|get\s+the|fetch|lookup|check\s+if|does\s+exist|"
    r"how\s+many|count|status\s+of|list\s+all|enumerate|"
    r"who\s+is|who\s+(made|created|built|wrote|developed)|when\s+did|where\s+is|tell\s+me|"
    r"what\s+does|what\s+kind|what\s+type|define)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Difficulty keyword sets
# ---------------------------------------------------------------------------
# HARD keywords signal the user wants deep, thorough, or complex work.
# These trigger escalation from <EASY_MODEL> to the paid <HARD_*> models.
# Conservative: need ≥1 hard signal AND user intent AND more hard than easy.
# A lone context signal (50k+ tokens) only escalates with a directive/hard
# keyword — never on "ok"/"yes"/"what is" easy replies.

_HARD_KW = re.compile(
    r"\b(prove|deduc|infer|implications?|trade-?offs?|consequences?|"
    r"edge\s*cases?|comprehensive|thorough|exhaustive|rigorous|in\s+depth|"
    r"deep\s*dive|root\s*cause|investigat|trace|diagnos|"
    r"architect|design\s+pattern|system\s+design|"
    r"optimi[sz]|refactor|restructur|rearchitect|"
    r"security\s+review|threat\s+model|risk\s+assess|"
    r"perform|scalab|concurr|race\s+condition|deadlock|"
    r"memory\s+leak|profil|bottleneck|latency|throughput|"
    r"multi-?step|multi-?stage|complex|intricate|nuanc|"
    r"subtle|advanced|expert|senior|staff\s+level|"
    r"production\s+ready|enterprise|mission\s+critical|critical|"
    r"no\s+gaps?|no\s+shortcuts?|production-?grade|battle-?tested|"
    r"regression|compatibility|breaking\s+change|"
    r"analy[sz]e|evaluat|assess|critiqu|"
    r"compar|contrast|weigh|reason\s+about|"
    r"first\s+principles?|from\s+scratch|ground\s+up|"
    r"end\s+to\s+end|full\s+stack|entire\s+system|"
    r"document|specific|detailed|elaborat|"
    r"why\s+does|how\s+does\s+this\s+work|understand\s+the\s+internals?|"
    r"walk\s+me\s+through|break\s+this\s+down|step\s+by\s+step|"
    r"corner\s+case|failure\s+mode|error\s+handling\s+strategy|"
    r"best\s+practices?|anti-?pattern|code\s+smell|"
    r"review\s+for|audit|pentest|vulnerability|"
    r"reverse\s+engineer|decompil|disassembl|"
    r"cryptograph|protocol|state\s+machine|"
    r"distributed|consensus|replication|sharding|"
    r"compiler|interpreter|runtime|virtual\s+machine|"
    r"formal|mathematical|proof|lemma|theorem|"
    r"migration\s+strategy|rollout|deployment\s+plan|"
    r"decompos|monolith|microservice|"
    r"scalability|high\s+availability|disaster\s+recovery|"
    r"engineer|reverse\s+engineer|production|"
    # Quality-qualifier hard signal — the user's terse style appends these
    # ("do it properly", "make sure it's right", "no half-assed").
    r"properly|correctly|done\s+right|the\s+right\s+way|as\s+it\s+should|"
    r"actually\s+work(?:s|ing)?|carefully|rigorously|no\s+half-?assed|"
    r"for\s+real(?:,\s*this\s+time)?|this\s+time\s+for\s+real|"
    r"do\s+it\s+right|make\s+it\s+work|make\s+it\s+right|get\s+it\s+right|"
    # Correction/friction hard signal — user is dissatisfied, wants it done
    # precisely this time ("still not", "not yet", "exactly").
    r"still\s+not|not\s+yet|precisely|exactly|"
    r"not\s+(?:triggering|working|getting|right|good|done|enough)|"
    r"doesn['’]?t\s+work|didn['’]?t\s+work|broke|broken|wrong|"
    r"incomplete|missing|overlooked|you\s+(?:missed|forgot)|"
    r"try\s+again|redo|more\s+broad|broader|more\s+precise)\b",
    re.IGNORECASE,
)

# DIRECTIVE keywords — the user's terse imperative style ("do it", "fix this",
# "get it working"). A directive on top of a large context is a strong hard
# signal even when the literal text has zero analytical vocabulary.
_DIRECTIVE_KW = re.compile(
    r"\b(do\s+it|do\s+this|do\s+that|do\s+the\s+thing|"
    r"make\s+it|make\s+this|make\s+it\s+work|make\s+it\s+right|"
    r"make\s+sure|ensure\s+(?:it|this|that)|"
    r"fix\s+it|fix\s+this|get\s+it\s+working|get\s+this\s+working|"
    r"work\s+it\s+out|sort\s+it|sort\s+this|handle\s+it|handle\s+this|"
    r"deal\s+with\s+it|take\s+care\s+of\s+it|"
    r"build\s+it|write\s+it|implement\s+it|finish\s+it|finish\s+this|"
    r"ship\s+it|push\s+this|wrap\s+this\s+up|"
    r"need\s+it\s+done|want\s+it\s+done|get\s+this\s+done|"
    r"make\s+it\s+happen|now\s+do|go\s+do|"
    r"we\s+need\s+to|we\s+should|let['’]s\s+get|time\s+to\s+get|"
    # Correction/friction phrasing from the user's actual transcripts:
    # "still not", "not good yet", "try again", "resume".
    r"still\s+(?:not|doesn['’]?t|isn['’]?t|won['’]?t)|"
    r"not\s+(?:working|triggering|getting|good|right|done|enough)|"
    r"doesn['’]?t\s+work|didn['’]?t\s+work|broke|broken|wrong|"
    r"incomplete|missing|overlooked|you\s+(?:missed|forgot)|"
    r"try\s+again|redo|resume|continue\s+(?:this|where|from)|"
    r"more\s+broad|broader|more\s+precise|precisely)\b",
    re.IGNORECASE,
)

# EASY keywords signal the user wants a quick, simple answer.
# These keep the task on <EASY_MODEL> even if some hard keywords are present.
_EASY_KW = re.compile(
    r"\b(briefly|quickly|simple|one\s+line|short|concise|tl;?dr|"
    r"just\s+the|only\s+the|skip\s+the|don't\s+explain|"
    r"yes\s+or\s+no|what\s+is|who\s+is|where\s+is|when\s+is|"
    r"list\s+the|name\s+the|give\s+me\s+the|"
    r"copy|paste|snippet|example|sample|template|"
    r"basic|intro|beginner|starter|boilerplate|"
    r"lookup|check|verify|confirm|"
    r"how\s+to|how\s+do\s+i|how\s+many|"
    r"what\s+does\s+this\s+mean|what\s+does\s+this\s+do|"
    r"remind\s+me|refresh\s+my\s+memory|"
    r"quick\s+question|simple\s+question|"
    r"just\s+wondering|curious|"
    r"summarize|tldr|bottom\s+line|"
    r"headline|bullet\s+point|"
    r"fast|asap|urgent\s+but\s+simple)\b",
    re.IGNORECASE,
)

# Tool name patterns → phase hints
_TOOL_PHASE = {
    "Agent":        "orchestration",
    "Workflow":     "orchestration",
    "TaskCreate":   "orchestration",
    "TaskUpdate":   "orchestration",
    "Edit":         "coding",
    "Write":        "coding",
    "NotebookEdit": "coding",
    "Read":         "lookup",
    "Glob":         "lookup",
    "Grep":         "lookup",
    "Bash":         "coding",
    "WebSearch":    "lookup",
    "WebFetch":     "lookup",
    "LSP":          "lookup",
}

# Estimated token threshold for long-context phase
_LONG_CTX_THRESHOLD = 180_000

# Difficulty thresholds
_HARD_KW_MIN = 1          # need at least this many hard keyword hits (amplified: was 2)
_HARD_PROMPT_LEN = 1000   # prompts longer than this (chars) add a hard signal
_HARD_CTX_TOKENS = 50_000 # context > this many tokens adds a hard signal
_HARD_SIGNALS_MIN = 1     # ANY single hard signal can escalate (was 2 — the
                          # 50k-ctx signal alone was the dominant blocker: 236/300
                          # classifications sat at exactly 1 signal and stayed on <EASY_MODEL>).
                          # False-positive guard moved into the classifier: a lone
                          # context signal only escalates when the user's own
                          # message shows intent (directive/hard keyword) and is
                          # NOT an easy "ok/yes/what is" reply.


class PhaseRouter(CustomLogger):
    """litellm pre-call hook that routes by task phase + difficulty."""

    PHASE_MAP = PHASE_MAP

    # --- No-op overrides for hooks we don't use ---

    async def async_post_call_success_hook(self, *args: Any, **kwargs: Any) -> Any:
        return kwargs.get("response")

    async def async_post_call_failure_hook(self, *args: Any, **kwargs: Any) -> Any:
        return None

    async def async_pre_call_check(self, *args: Any, **kwargs: Any) -> Any:
        return None

    async def async_success_hook(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def async_failure_hook(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """Classify the request phase+difficulty and rewrite data['model']."""
        try:
            original_model = data.get("model", "")
            messages = data.get("messages", [])
            tools = data.get("tools", [])
            metadata = data.get("metadata", {}) or {}

            # --- Safety rail: respect explicit non-default model overrides ---
            force = metadata.get("FORCE_PHASE_ROUTING", False)
            if original_model and original_model not in _REWRITABLE_MODELS and not force:
                _logger.debug(
                    "skip: model=%s is explicit (not default), not rewriting",
                    original_model,
                )
                return data

            # --- Safety rail: skip non-completion calls ---
            if call_type not in ("completion", "acompletion", "chat_completion", "achatcompletion", "anthropic_messages"):
                return data

            # --- Classify ---
            phase = self._classify(messages, tools, metadata)
            target_model = self.PHASE_MAP.get(phase)

            if not target_model or target_model == original_model:
                _logger.debug(
                    "noop: phase=%s model=%s (no rewrite needed)",
                    phase, original_model,
                )
                return data

            # --- Safety rail: skip models with known-bad upstream behavior ---
            if target_model in _SKIP_MODELS:
                fallback = FALLBACKS.get(target_model, [original_model])[0]
                _logger.info(
                    "skip: phase=%s target=%s in SKIP list, using fallback %s",
                    phase, target_model, fallback,
                )
                target_model = fallback

            # --- Safety rail: when tools are present, only reroute to tool-capable models ---
            if tools and target_model not in _TOOL_CAPABLE_MODELS:
                fallback = FALLBACKS.get(target_model, [original_model])[0]
                _logger.info(
                    "skip: phase=%s target=%s not tool-capable, using fallback %s (tools=%d)",
                    phase, target_model, fallback, len(tools),
                )
                target_model = fallback
                if tools and target_model not in _TOOL_CAPABLE_MODELS:
                    _logger.warning(
                        "fallback %s also not tool-capable, keeping %s",
                        target_model, original_model,
                    )
                    return data

            # --- Rewrite ---
            data["model"] = target_model

            if "litellm_params" not in data or not isinstance(data["litellm_params"], dict):
                data["litellm_params"] = {}
            data["litellm_params"]["phase_tag"] = phase
            data["litellm_params"]["phase_original_model"] = original_model
            data["litellm_params"]["phase_routed_model"] = target_model
            data["litellm_params"]["phase_routed_ts"] = time.time()

            _logger.info(
                "ROUTE phase=%s | %s → %s | tools=%d msgs=%d",
                phase, original_model, target_model,
                len(tools), len(messages),
            )
            return data

        except Exception as exc:
            _logger.error("PhaseRouter error (passing through): %s", exc, exc_info=True)
            return data

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------

    def _classify(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> str:
        """Return phase string (includes difficulty suffix for escalatable phases).

        Returns one of:
          thinking, thinking_hard, planning, planning_hard,
          coding, coding_hard, orchestration, lookup, long_context, default
        """

        text = self._recent_text(messages, last_n=2)
        est_tokens = self._estimate_tokens(messages)

        # --- Phase classification (keyword-based, user intent first) ---

        kw_scores = {
            "thinking":      len(_THINKING_KW.findall(text)) if text else 0,
            "planning":      len(_PLANNING_KW.findall(text)) if text else 0,
            "orchestration": len(_ORCHESTRATION_KW.findall(text)) if text else 0,
            "coding":        len(_CODING_KW.findall(text)) if text else 0,
            "lookup":        len(_LOOKUP_KW.findall(text)) if text else 0,
        }

        kw_winner = max(kw_scores, key=kw_scores.get)
        kw_max = kw_scores[kw_winner]

        # Determine the base phase
        phase = "default"

        # 1. If user prompt has a clear keyword winner (≥2 hits), trust it.
        #    Orchestration must WIN outright (score > other phases) to be chosen —
        #    a single weak orchestration hit is never enough to beat planning/coding.
        if kw_max >= 2:
            if kw_winner == "planning" and kw_max < 3:
                pass  # planning needs stronger signal
            elif kw_winner == "orchestration":
                runner_up = sorted(
                    kw_scores.values(), reverse=True,
                )[1] if len(kw_scores) > 1 else 0
                # Require orchestration to be the strict winner, and the margin
                # to be meaningful (>=2 clear lead). Otherwise defer to tools
                # / the next-best phase instead of defaulting to <ORCHESTRATION_MODEL>.
                if kw_max > runner_up and (kw_max - runner_up) >= 2:
                    phase = "orchestration"
            else:
                phase = kw_winner

        # 2. If no clear keyword winner, try tool-based detection
        if phase == "default":
            tool_phases = self._phases_from_tools(tools)
            if tool_phases:
                if tool_phases == {"coding"}:
                    phase = "coding"
                elif tool_phases == {"lookup"}:
                    phase = "lookup"
                elif "orchestration" in tool_phases:
                    # Only orchestration if the USER text (last message) shows
                    # orchestration intent AND no other phase out-scores it.
                    user_txt = self._recent_text(messages[-1:], last_n=1)
                    if user_txt and _ORCHESTRATION_KW.search(user_txt):
                        if kw_scores["orchestration"] >= kw_scores["coding"] and \
                           kw_scores["orchestration"] >= kw_scores["planning"]:
                            phase = "orchestration"

        # 3. If still default, use keyword winner even if weak (≥1 hit) —
        #    but never let a weak orchestration steal from coding/planning/thinking.
        if phase == "default" and kw_max >= 1:
            if kw_winner == "orchestration" and kw_max < 2:
                # Weak orchestration: fall through to the next-best phase.
                other = {
                    k: v for k, v in kw_scores.items() if k != "orchestration"
                }
                if other:
                    next_winner = max(other, key=other.get)
                    if other[next_winner] >= 1:
                        phase = next_winner
            else:
                phase = kw_winner

        # 4. Long context overrides everything
        if est_tokens > _LONG_CTX_THRESHOLD:
            phase = "long_context"

        # --- Difficulty assessment (only for escalatable phases) ---

        # Difficulty runs on escalatable phases AND on "default". A terse
        # directive ("still not getting triggered precisely", "do it properly")
        # has zero phase keywords → lands on "default" — and a default with a
        # directive/hard intent must still be able to escalate. So normalize
        # default-with-intent to "coding" (directive work is coding work).
        hard_probe = len(_HARD_KW.findall(text)) if text else 0
        directive_probe = len(_DIRECTIVE_KW.findall(text)) if text else 0
        if phase == "default" and (hard_probe >= 1 or directive_probe >= 1):
            phase = "coding"

        # Directive overrides thinking: a terse imperative ("we need to be more
        # broad", "fix it") is coding intent even if it happens to contain a
        # thinking word ("analyze the way i type"). Escalation is what matters;
        # the label should read as the work being done.
        if phase == "thinking" and directive_probe >= 1:
            phase = "coding"

        if phase in ("thinking", "planning", "coding"):
            hard_score = len(_HARD_KW.findall(text)) if text else 0
            easy_score = len(_EASY_KW.findall(text)) if text else 0
            directive_score = len(_DIRECTIVE_KW.findall(text)) if text else 0

            # Structural hard signals — a directive counts as a hard signal so a
            # terse imperative ("fix this", "not good yet") escalates even on a
            # small context with zero analytical vocabulary.
            hard_signals = 0
            if hard_score >= _HARD_KW_MIN:
                hard_signals += 1
            if directive_score >= 1:
                hard_signals += 1
            if len(text) > _HARD_PROMPT_LEN:
                hard_signals += 1
            if est_tokens > _HARD_CTX_TOKENS:
                hard_signals += 1

            # User-message intent (the part that's under the operator's control):
            # a hard keyword, a quality-qualifier, or a terse directive.
            user_intent = (hard_score >= 1) or (directive_score >= 1)

            # Easy-message guard — a lone context signal must NOT escalate when
            # the user's own message is an easy reply ("ok", "yes", "what is X").
            ctx_only = (hard_signals == 1 and est_tokens > _HARD_CTX_TOKENS
                        and hard_score == 0 and directive_score == 0
                        and len(text) <= _HARD_PROMPT_LEN)
            easy_reply = (easy_score >= 1 and not user_intent)

            # Escalate if: enough hard keyword hits AND more hard than easy
            # AND enough structural signals. Amplified (2026-08-10):
            #   - hard keywords lowered 2→1 (more triggers)
            #   - 50k-ctx signal now counts toward hard_signals
            #   - need >=2 signals so 50k-ctx alone can't false-positive into paid
            # Broadened (2026-08-10, this edit):
            #   - guard lowered 2→1: ANY single signal can escalate
            #   - NEW directive signal: terse imperatives ("do it", "fix this",
            #     "get it working") now count as user intent
            #   - context-only escalation needs user intent and NOT an easy reply
            #     ("ok"/"yes"/"what is") — big context alone no longer escalates
            is_hard = (
                hard_signals >= _HARD_SIGNALS_MIN
                and user_intent
                and (hard_score > easy_score or directive_score >= 1)
            )

            # Lone-context special case: big session + a directive/terse task but
            # zero analytical keywords ("we need to be more broad" on 100k tokens).
            if not is_hard and ctx_only and user_intent and not easy_reply:
                is_hard = True

            if is_hard:
                phase = phase + "_hard"

            _logger.debug(
                "classify phase=%s hard=%d easy=%d directive=%d hard_signals=%d → %s "
                "(tokens~%d, tools=%d, kw=%s)",
                phase.replace("_hard", ""), hard_score, easy_score,
                directive_score, hard_signals, phase, est_tokens, len(tools),
                kw_scores,
            )
        else:
            _logger.debug(
                "classify → %s (tokens~%d, tools=%d, kw=%s)",
                phase, est_tokens, len(tools), kw_scores,
            )

        return phase

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _phases_from_tools(tools: List[Dict[str, Any]]) -> set:
        """Extract phase hints from tool schemas."""
        phases = set()
        for t in tools:
            func = t.get("function", t)
            name = func.get("name", "")
            if name in _TOOL_PHASE:
                phases.add(_TOOL_PHASE[name])
                continue
            for pattern, phase in _TOOL_PHASE.items():
                if pattern in name:
                    phases.add(phase)
                    break
        return phases

    @staticmethod
    def _recent_text(messages: List[Dict[str, Any]], last_n: int = 4) -> str:
        """Extract text content from the last N messages."""
        texts = []
        for msg in messages[-last_n:]:
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            texts.append(part.get("text", ""))
                        elif part.get("type") == "tool_use":
                            texts.append(part.get("name", ""))
        return " ".join(texts)

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total_chars += len(str(part.get("text", "")))
        return total_chars // 4


# Module-level instance so the config can reference the instance, not the class.
phase_router = PhaseRouter()
