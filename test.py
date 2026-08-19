# check_generation_state.py
from film_generation.pipeline import build_generation_graph
from film_generation.config import load_config
from film_generation.schemas import GenerationState
from run_movie import load_completed_reasoning_state

reasoning_state = load_completed_reasoning_state()   # zero LLM calls, reads reasoning.db
config = load_config()
graph = build_generation_graph(reasoning_state, config)  # just wires the graph, no execution

thread_config = {"configurable": {"thread_id": "percy_jackson"}}  # match your real thread_id
snapshot = graph.get_state(thread_config)

if snapshot is None or not snapshot.values:
    print("No checkpoint yet for this thread_id.")
else:
    print("Pending node(s) (what runs next on resume):", snapshot.next)
    print()

    gen = GenerationState(**snapshot.values)  # typed access instead of raw dict
    print("Characters generated: ", list(gen.character_refs.keys()))
    print("Scene anchors:        ", list(gen.scene_anchors.keys()))
    print("Shot prompts built:   ", len(gen.shot_prompts))
    print("Images generated:     ", [f"{i.scene_number}-{i.shot_number}" for i in gen.generated_images])
    print("Critiques so far:     ", len(gen.visual_critiques))
    print("Clips generated:      ", [f"{c.scene_number}-{c.shot_number}" for c in gen.generated_clips])
    print("EDLs planned:         ", [e.scene_number for e in gen.edit_decision_lists])
    print("Final movie path:     ", gen.final_movie_path)