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
import time

from film_agents.schemas import ProjectState
from film_agents.graph import build_graph

load_dotenv()


def main():
    script_path = sys.argv[1] if len(sys.argv) > 1 else "sample_script.txt"
    script_text = Path(script_path).read_text()

    graph = build_graph()
    print(graph)
    initial_state = ProjectState(script_text=script_text)

    print(f"Running pipeline on: {script_path}\n{'=' * 60}")

    final_state_dict = None
    for step in graph.stream(initial_state):
        node_name = list(step.keys())[0]
        print(f"[{time.strftime('%H:%M:%S')}] ✓ completed node: {node_name}")
        final_state_dict = step[node_name]

    final_state = ProjectState(**final_state_dict)


if __name__ == "__main__":
    main()
