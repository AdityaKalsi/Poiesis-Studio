"""
run_movie.py -- runs the ENTIRE pipeline in one command: script -> reasoning
pipeline -> generation pipeline -> final .mp4.

This sits alongside main.py (which only runs the reasoning stage) rather
than replacing it -- main.py stays useful on its own for iterating on
the reasoning pipeline without triggering paid generation API calls.

Usage:
    python run_movie.py path/to/script.txt
    python run_movie.py                      # uses sample_script.txt

Requires, in the environment:
    GEMINI_API_KEY (or GOOGLE_API_KEY)   -- image generation + visual critic
    FAL_KEY                              -- Wan 2.5 video generation via fal.ai
    HF_TOKEN                             -- only if config.models.text2image_backend
                                             is switched to "huggingface"
Also requires ffmpeg on PATH -- generation/video_editor.py shells out to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from film_agents.schemas import ProjectState
from film_agents.graph import build_reasoning_graph

from film_generation.adapter import build_generation_state
from film_generation.config import load_config
from film_generation.pipeline import run_generation


# Must match the thread_id your real run_movie.py used to populate reasoning.db.
REASONING_THREAD_ID = "REASONING_THREAD_ID"  # change this if you used a different thread_id in run_movie.py

TEST_GENERATION_DB = "test_generation.db"   # isolated -- never touches generation.db
TEST_THREAD_ID = "checkpoint_flow_test"


def load_completed_reasoning_state() -> ProjectState:
    """Pull the finished ProjectState straight out of reasoning.db.
    Does NOT invoke the graph -- zero LLM calls happen here."""
    graph = build_reasoning_graph()
    thread_config = {"configurable": {"thread_id": REASONING_THREAD_ID}}

    existing = graph.get_state(thread_config)
    if existing is None or not existing.values:
        sys.exit(
            f"No checkpoint found for thread_id={REASONING_THREAD_ID!r} in "
            f"reasoning.db -- run the reasoning pipeline first, or fix "
            f"REASONING_THREAD_ID above to match what run_movie.py used."
        )
    if existing.next:
        sys.exit(
            f"Checkpoint for {REASONING_THREAD_ID!r} is incomplete "
            f"(pending nodes: {existing.next}) -- this test expects a "
            f"fully finished reasoning run."
        )

    print(f"[Reasoning] loaded completed checkpoint for thread_id={REASONING_THREAD_ID!r} "
          f"(no LLM calls made)")
    return ProjectState(**existing.values)

def run_reasoning_stage(script_path: str) -> ProjectState:

    script_text = Path(script_path).read_text()

    reasoning_graph = build_reasoning_graph()

    print(f"[1/2] Reasoning pipeline running on: {script_path}\n{'=' * 60}")

    final_reasoning_state = reasoning_graph.invoke(ProjectState(script_text=script_text), config={
        "configurable": {
            "thread_id": "REASONING_THREAD_ID"
        }
    })

    # Save the final state to a JSON file
    final_state = ProjectState(**final_reasoning_state)
    out_path = Path("output.json")
    out_path.write_text(final_state.model_dump_json(indent=2))
    print(f"Reasoning output written to {out_path.resolve()}\n")

    # Visualize LangGraph
    png = reasoning_graph.get_graph().draw_mermaid_png()
    with open("reasoning_graph.png", "wb") as f:
        f.write(png)
    print("Graph saved as reasoning_graph.png")

    return final_state


def run_generation_stage(project_state: ProjectState, resume: bool = False,
                          thread_id: str = "percy_jackson"):
    print(f"[2/2] Generation pipeline running \n{'=' * 60}")
    config = load_config()

    return run_generation(project_state, config, resume=resume, thread_id=thread_id)


def main() -> None:
    script_path = sys.argv[1] if len(sys.argv) > 1 else "sample_script.txt"
    resume = "--resume" in sys.argv

    #final_resoning_state = run_reasoning_stage(script_path)
    final_resoning_state = load_completed_reasoning_state()

    if not final_resoning_state.breakdown or not final_resoning_state.shot_lists:
        print("Reasoning pipeline produced no usable breakdown/shot lists -- "
              "stopping before generation (nothing to generate from).")
        sys.exit(1)

    final_gen_state = run_generation_stage(final_resoning_state, resume=resume)
    #final_gen_state = run_generation_stage(final_resoning_state)

    # Write the complete generation stage state to a JSON file
    gen_out_path = Path("generation_output.json")
    gen_out_path.write_text(final_gen_state.model_dump_json(indent=2))
    print(f"Generation output written to {gen_out_path.resolve()}\n")
    print("\n" + "=" * 60)
    for line in final_gen_state.log:
        print(f"  {line}")

    # Visualize LangGraph
    

    if final_gen_state.final_movie_path:
        print(f"\nDone. Final movie: {final_gen_state.final_movie_path}")
    else:
        print("\nGeneration pipeline finished but final_movie_path was never "
              "set -- check the log above for where it stopped.")


if __name__ == "__main__":
    main()