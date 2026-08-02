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

from film_generation.config import load_config
from film_generation.pipeline import run_generation


def run_reasoning_stage(script_path: str) -> ProjectState:
    script_text = Path(script_path).read_text()
    graph = build_reasoning_graph()

    print(f"[1/2] Reasoning pipeline running on: {script_path}\n{'=' * 60}")
    final_state_dict = graph.invoke(ProjectState(script_text=script_text))
    final_state = ProjectState(**final_state_dict)

    out_path = Path("output.json")
    out_path.write_text(final_state.model_dump_json(indent=2))
    print(f"      Reasoning output written to {out_path.resolve()}\n")
    # Visualize LangGraph
    png = graph.get_graph().draw_mermaid_png()
    
    with open("reasoning_graph.png", "wb") as f:
        f.write(png)
    
    print("Graph saved as reasoning_graph.png")
    return final_state


def run_generation_stage(project_state: ProjectState):
    print(f"[2/2] Generation pipeline running (calls paid image/video APIs)\n{'=' * 60}")
    config = load_config()
    return run_generation(project_state, config)


def main() -> None:
    script_path = sys.argv[1] if len(sys.argv) > 1 else "sample_script.txt"

    project_state = run_reasoning_stage(script_path)

    if not project_state.breakdown or not project_state.shot_lists:
        print("Reasoning pipeline produced no usable breakdown/shot lists -- "
              "stopping before generation (nothing to generate from).")
        sys.exit(1)

    final_gen_state = run_generation_stage(project_state)
    # Write the complete generation stage state to a JSON file
    gen_out_path = Path("generation_output.json")
    gen_out_path.write_text(final_gen_state.model_dump_json(indent=2))
    print(f"      Generation output written to {gen_out_path.resolve()}\n")
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
