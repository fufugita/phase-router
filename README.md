# Phase Router

A litellm pre-call hook that auto-switches models by task phase and difficulty.

Loaded via `litellm_settings.callbacks` in the litellm proxy config. Classifies
each request (keyword + structural, ~0ms, no LLM call) and rewrites the model
selection before the router picks a deployment. Existing per-model fallback
ladders are unaffected.

## How it works

Two-layer classification:

1. **Phase** — what kind of task? (thinking, planning, orchestration, coding, lookup)
2. **Difficulty** — how hard? (easy, hard) — only for thinking/planning/coding

| Phase | Difficulty | Model |
|---|---|---|
| thinking | easy | `<EASY_MODEL>` (free tier) |
| thinking | hard | `<HARD_CODING_MODEL>` (paid tier) |
| planning | easy | `<EASY_MODEL>` |
| planning | hard | `<HARD_PLANNING_MODEL>` |
| coding | easy | `<EASY_MODEL>` |
| coding | hard | `<HARD_CODING_MODEL>` |
| orchestration | — | `<ORCHESTRATION_MODEL>` |
| lookup | — | `<EASY_MODEL>` |

## Setup

1. Place `phase_router.py` in your litellm config directory.
2. Add to your `litellm_settings.callbacks` in the proxy YAML:

```yaml
litellm_settings:
  callbacks: custom_callbacks.phase_router
```

3. Restart the litellm proxy.

Routing decisions are logged to `phase_router.log` alongside the script.

## Configuration

All model assignments live in the `PHASE_MAP` dict at the top of the file.
Replace the placeholder values with your actual litellm model names. The
`FALLBACKS` dict defines per-model fallback chains when a primary fails.

Easy-task ladders fall back to the next-best free model only. Hard-task
primaries degrade to another paid model first (paid→paid), never silently to
the free tier.

## Tooling note

Phase detection is driven by the user's prompt, not the tool inventory.
Orchestration tools (Agent, Workflow, TaskCreate, TaskUpdate) are always
loaded in agent contexts, so they are deliberately excluded from tool-based
phase votes — otherwise every request with tools would classify as
orchestration. A working session (tool inventory present + large context)
counts as a structural hard signal so substantive continuations escalate
even when the trailing message is a terse directive or a bare tool result.
