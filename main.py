"""
Run the prototype multi-agent film pipeline end-to-end on a sample script.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py [path/to/script.txt]

This produces a structured shooting plan (breakdown -> intent briefs ->
shot lists -> edited sequence, each with critic scores) -- no rendering,
no video. That's deliberate: prove the reasoning pipeline first.
"""

import sys
import json
from pathlib import Path

from dotenv import load_dotenv

from film_agents.schemas import ProjectState
from film_agents.graph import build_graph

load_dotenv()


def main():
    script_path = sys.argv[1] if len(sys.argv) > 1 else "sample_script.txt"
    script_text = Path(script_path).read_text()

    graph = build_graph()
    initial_state = ProjectState(script_text=script_text)

    print(f"Running pipeline on: {script_path}\n{'=' * 60}")
    final_state_dict = graph.invoke(initial_state)
    final_state = ProjectState(**final_state_dict)

    print("\n--- PIPELINE LOG ---")
    for line in final_state.log:
        print(line)

    print("\n--- SCRIPT BREAKDOWN ---")
    print(final_state.breakdown.model_dump_json(indent=2))

    print("\n--- SCENE INTENT BRIEFS ---")
    for b in final_state.intent_briefs:
        print(b.model_dump_json(indent=2))

    print("\n--- SHOT LISTS ---")
    for sl in final_state.shot_lists:
        print(sl.model_dump_json(indent=2))

    print("\n--- CINEMATOGRAPHY CRITIQUES (final pass) ---")
    for c in final_state.cinematography_critiques:
        print(c.model_dump_json(indent=2))

    print("\n--- EDITED SEQUENCES ---")
    for seq in final_state.edited_sequences:
        print(seq.model_dump_json(indent=2))

    print("\n--- FINAL CRITIQUES ---")
    for c in final_state.final_critiques:
        print(c.model_dump_json(indent=2))

    out_path = Path("output.json")
    out_path.write_text(final_state.model_dump_json(indent=2))
    print(f"\nFull structured output written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
