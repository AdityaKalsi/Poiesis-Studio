# Film Agents — Multi-Agent Filmmaking Pipeline (v1 Prototype)

A LangGraph-based prototype that proves the *planning/reasoning* layer of
a multi-agent filmmaking system: script → breakdown → scene intent →
shot list → edited sequence, with a Critic agent gating quality at each
stage. **No video/image generation yet, by design** — this proves the
reasoning loop works before spending compute on rendering.

## Pipeline

```
script_analyst -> scene_design -> cinematography -> critic_cinematography
    --(scene fails, retries left)--> cinematography   [revision loop]
    --(all pass, or retries exhausted)--> editing -> final_critic -> END
```

Every agent reads/writes typed Pydantic objects (`film_agents/schemas.py`)
instead of free text, so outputs are inspectable, diffable, and scorable.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Run

```bash
python main.py                      # uses sample_script.txt
python main.py path/to/your_script.txt
```

Output prints to stdout at every stage and is also written to
`output.json` as the full structured `ProjectState`.

## What to look at first

1. **`sample_script.txt`** — a 2-scene test script with a clear tension
   beat (Sarah hiding a letter/gun before Marcus arrives). Good for
   sanity-checking whether the agents pick up on subtext.
2. **`film_agents/prompts.py`** — this is where the actual film-craft
   knowledge lives. If output quality is off, tune these first before
   touching the graph.
3. **`film_agents/graph.py`** — the orchestration. The revision loop
   (`route_after_cinematography_critic`) is the smallest possible version
   of the "Critic gates every stage" principle from the architecture doc.

## Extending this prototype (in priority order)

1. **Add a Scheduler/Producer stage** that just estimates cost/time per
   scene — cheap to add, proves the Orchestrator concept without new
   generation capability.
2. **Add a Continuity Agent** that maintains a running graph of
   characters/props/facts across scenes and flags contradictions — this
   is the next-highest-value addition before you touch rendering at all.
3. **Add a single rendering agent for ONE shot type** (e.g. call an
   image-generation model for a single establishing frame per scene) —
   resist doing this for every shot until the planning layer is solid.
4. **Add human-in-the-loop checkpoints**: pause after `scene_design` and
   `critic_cinematography` for manual approval before continuing — swap
   the relevant `add_edge` for a LangGraph `interrupt`.
5. **Parallelize scene processing**: right now scenes within a stage are
   processed in a Python loop; LangGraph supports fan-out/fan-in
   (`Send` API) to genuinely parallelize per-scene work once you're
   ready to scale beyond a handful of scenes.

## Notes

- Default model is `claude-sonnet-5`; override with `FILM_AGENTS_MODEL`.
- Structured output uses `.with_structured_output(PydanticModel)` —
  if you swap providers, make sure the model supports tool-calling based
  structured output.
- `max_revisions` (in `ProjectState`) caps the cinematography retry loop
  at 2 by default — raise it if the Critic Agent keeps failing sound
  shot lists, but investigate the critic prompt first, since infinite
  retries usually mean the rubric or the brief is ambiguous, not that
  the cinematographer needs more attempts.
