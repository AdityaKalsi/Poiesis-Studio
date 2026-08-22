import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


THREAD_ID = "percy_jackson"

serializer = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("film_generation.schemas", "GenerationState"),
        ("film_generation.schemas", "EditDecisionList"),
        ("film_generation.schemas", "GeneratedClip"),
    ]
)

conn = sqlite3.connect("generation.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn, serde=serializer)

checkpoint = checkpointer.get_tuple(
    {"configurable": {"thread_id": THREAD_ID}}
)

if checkpoint is None:
    print("No checkpoint found.")
    exit()

values = checkpoint.checkpoint["channel_values"]

edls = values.get("edit_decision_lists", [])

print(f"Number of EDLs: {len(edls)}")

for edl in edls:
    print(f"\nScene: {edl.scene_number}")
    print(f"FPS: {edl.fps}")
    print(f"Number of decisions: {len(edl.decisions)}")

    for decision in edl.decisions:
        print(
            f"  Shot: {decision.shot_number} | "
            f"Frames: {decision.start_frame}-{decision.end_frame} | "
            f"Transition: {decision.transition_in}"
        )