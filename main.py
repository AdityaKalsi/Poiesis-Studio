import sys
import json
from pathlib import Path

from film_agents.schemas import ProjectState
from film_agents.graph import build_graph



def main():
    script_path = sys.argv[1] if len(sys.argv) > 1 else "sample_script.txt"
    script_text = Path(script_path).read_text()

    graph = build_graph()
    initial_state = ProjectState(script_text=script_text)

    print(f"Running pipeline on: {script_path}\n{'=' * 60}")
    final_state_dict = graph.invoke(initial_state)
    final_state = ProjectState(**final_state_dict)

    # Visualize LangGraph
    png = graph.get_graph().draw_mermaid_png()

    with open("pipeline_graph.png", "wb") as f:
        f.write(png)

    print("Graph saved as pipeline_graph.png")

    out_path = Path("output.json")
    out_path.write_text(final_state.model_dump_json(indent=2))
    print(f"\nFull structured output written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
