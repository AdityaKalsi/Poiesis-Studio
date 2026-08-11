"""
test_checkpoint_flow.py

Isolated test to answer one question: when a node downstream of
character_reference blows up, does character_reference's own checkpoint
actually survive in the generation checkpointer?

What this does:
    1. Loads the ALREADY-COMPLETED reasoning checkpoint from reasoning.db
       via graph.get_state() -- does NOT call graph.invoke(), so no
       script_analyst/scene_design/cinematography/etc. LLM calls happen.
    2. Runs adapter.build_generation_state() on that ProjectState, same
       as pipeline.run_generation() does.
    3. Runs a TRUNCATED generation graph containing only two nodes:
       character_reference -> scene_anchor, where scene_anchor is
       replaced with a function that raises on purpose.
    4. After the (expected) crash, re-opens the checkpoint via
       graph.get_state() and also queries the raw sqlite table directly,
       to confirm character_refs was persisted even though scene_anchor
       never returned.

Uses a SEPARATE db file (test_generation.db) so this never touches your
real generation.db. Safe to run and delete afterward.

Run from the project root (same directory you run run_movie.py from):
    python test_checkpoint_flow.py
"""

from __future__ import annotations

import sqlite3
import sys

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from film_agents.schemas import ProjectState
from film_agents.graph import build_reasoning_graph

from film_generation.adapter import build_generation_state
from film_generation.config import load_config
from film_generation.schemas import GenerationState
from film_generation.generation.asset_manager import AssetManager
from film_generation.generation import character_reference as char_ref_module


# Must match the thread_id your real run_movie.py used to populate reasoning.db.
REASONING_THREAD_ID = "movie_001"

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


def build_truncated_generation_graph(project_state: ProjectState, config):
    """Only character_reference -> scene_anchor. scene_anchor is swapped
    for a function that always raises, so we can inspect what survived."""
    assets = AssetManager(config.output_dir)

    def node_character_reference(state: GenerationState) -> dict:
        print(f"[TEST] character_reference: resolving refs for "
              f"{len(project_state.breakdown.scenes)} scenes")
        refs = dict(state.character_refs)
        for scene in project_state.breakdown.scenes:
            for character in scene.cast:
                if character in refs:
                    continue
                description = ", ".join(scene.wardrobe_notes) or "no description available"
                ref = char_ref_module.generate_character_reference(
                    character_name=character,
                    description=description,
                    config=config,
                    assets=assets,
                )
                refs[character] = ref
        print(f"[TEST] character_reference: resolved {len(refs)} refs -- "
              f"returning now (this is the point LangGraph checkpoints)")
        return {
            "character_refs": refs,
            "log": state.log + [f"[TEST] character_reference resolved {len(refs)} characters"],
        }

    def node_scene_anchor_deliberate_failure(state: GenerationState) -> dict:
        print("[TEST] scene_anchor: raising on purpose now")
        raise RuntimeError(
            "Deliberate test failure in scene_anchor -- checking whether "
            "character_reference's checkpoint survives this."
        )

    graph = StateGraph(GenerationState)
    graph.add_node("character_reference", node_character_reference)
    graph.add_node("scene_anchor", node_scene_anchor_deliberate_failure)
    graph.set_entry_point("character_reference")
    graph.add_edge("character_reference", "scene_anchor")
    graph.add_edge("scene_anchor", END)

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("film_agents.schemas", "GenerationState"),
            ("film_agents.schemas", "CharacterReference"),
            ("film_agents.schemas", "SceneAnchor"),
            ("film_agents.schemas", "ShotPrompt"),
            ("film_agents.schemas", "GeneratedImage"),
            ("film_agents.schemas", "VisualCritique"),
            ("film_agents.schemas", "GeneratedClip"),
            ("film_agents.schemas", "EditDecisionList"),
        ]
    )
    conn = sqlite3.connect(database=TEST_GENERATION_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn, serde=serializer)
    return graph.compile(checkpointer=checkpointer), conn


def inspect_checkpoint_db(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 60)
    print(f"Raw inspection of {TEST_GENERATION_DB}")
    print("=" * 60)

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    print(f"Tables present: {tables}")

    if "checkpoints" not in tables:
        print("!! No 'checkpoints' table -- SqliteSaver schema was never created.")
        return

    cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
    print(f"thread_ids with data: {[row[0] for row in cur.fetchall()]}")

    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (TEST_THREAD_ID,))
    print(f"Checkpoint rows for {TEST_THREAD_ID!r}: {cur.fetchone()[0]}")


def run_interrupt_control_test(project_state: ProjectState, config, initial_state: GenerationState) -> None:
    """Control test: stop cleanly AFTER character_reference using LangGraph's
    own interrupt_after, with NO exception anywhere. If this also shows 0
    rows, the problem is unrelated to the exception -- it's that commits
    aren't happening at all with this connection setup. If THIS test shows
    a row but the exception-based one doesn't, the exception is causing a
    rollback."""
    print("\n" + "#" * 60)
    print("CONTROL TEST: interrupt_after=['character_reference'], no exception")
    print("#" * 60)

    assets = AssetManager(config.output_dir)

    def node_character_reference(state: GenerationState) -> dict:
        refs = dict(state.character_refs)
        for scene in project_state.breakdown.scenes:
            for character in scene.cast:
                if character in refs:
                    continue
                description = ", ".join(scene.wardrobe_notes) or "no description available"
                ref = char_ref_module.generate_character_reference(
                    character_name=character, description=description,
                    config=config, assets=assets,
                )
                refs[character] = ref
        return {"character_refs": refs, "log": state.log + ["[CONTROL] character_reference resolved"]}

    def node_scene_anchor_never_reached(state: GenerationState) -> dict:
        raise AssertionError("This should never run -- interrupt_after should stop before this.")

    graph = StateGraph(GenerationState)
    graph.add_node("character_reference", node_character_reference)
    graph.add_node("scene_anchor", node_scene_anchor_never_reached)
    graph.set_entry_point("character_reference")
    graph.add_edge("character_reference", "scene_anchor")
    graph.add_edge("scene_anchor", END)

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("film_agents.schemas", "GenerationState"),
            ("film_agents.schemas", "CharacterReference"),
            ("film_agents.schemas", "SceneAnchor"),
        ]
    )
    conn = sqlite3.connect(database=TEST_GENERATION_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn, serde=serializer)
    if hasattr(checkpointer, "setup"):
        checkpointer.setup()  # explicit, in case tables need re-creating -- harmless if already present

    compiled = graph.compile(checkpointer=checkpointer, interrupt_after=["character_reference"])
    thread_config = {"configurable": {"thread_id": "checkpoint_flow_control"}}

    compiled.invoke(initial_state, config=thread_config)  # should NOT raise at all

    state_after = compiled.get_state(thread_config)
    print(f"[CONTROL] state.next after clean interrupt: {state_after.next if state_after else None}")
    print(f"[CONTROL] character_refs present in returned state: "
          f"{bool(state_after and state_after.values.get('character_refs'))}")

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("checkpoint_flow_control",))
    row_count_before_commit = cur.fetchone()[0]
    print(f"[CONTROL] checkpoint rows visible BEFORE explicit commit: {row_count_before_commit}")

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("checkpoint_flow_control",))
    row_count_after_commit = cur.fetchone()[0]
    print(f"[CONTROL] checkpoint rows visible AFTER explicit commit: {row_count_after_commit}")

    if row_count_before_commit == 0 and row_count_after_commit > 0:
        print("[CONTROL DIAGNOSIS] Writes are happening but never auto-committed "
              "by the checkpointer/connection -- this is a missing-commit issue, "
              "not something caused by the exception test.")
    elif row_count_after_commit == 0:
        print("[CONTROL DIAGNOSIS] Still 0 even after explicit commit and even "
              "with NO exception involved -- the write itself isn't happening "
              "(check langgraph-checkpoint-sqlite version / whether .setup() "
              "was required first / whether this conn object is really the "
              "one being used internally).")
    else:
        print("[CONTROL DIAGNOSIS] Commits work fine in the clean-stop case. "
              "The exception path is the differentiator -- worth checking if "
              "your langgraph version rolls back the whole tick on an "
              "unhandled node exception.")

    conn.close()


def main() -> None:
    project_state = load_completed_reasoning_state()

    print("[Adapter] building GenerationState from the loaded ProjectState...")
    initial_state = build_generation_state(project_state)

    config = load_config()

    # --- exception-based test (original) ---
    graph, conn = build_truncated_generation_graph(project_state, config)
    thread_config = {"configurable": {"thread_id": TEST_THREAD_ID}}

    try:
        graph.invoke(initial_state, config=thread_config)
    except Exception as exc:  # noqa: BLE001 -- intentionally broad, this is a test harness
        print(f"\n[TEST] graph raised as expected: {exc!r}")
    else:
        print("\n[TEST] graph did NOT raise -- scene_anchor's deliberate "
              "failure didn't trigger. Check the patch above.")

    state_after_crash = graph.get_state(thread_config)
    if state_after_crash is not None and state_after_crash.values.get("character_refs"):
        print(f"\n[RESULT] character_refs WAS checkpointed: "
              f"{list(state_after_crash.values['character_refs'].keys())}")
        print(f"[RESULT] Pending node(s) on resume would be: {state_after_crash.next}")
    else:
        print("\n[RESULT] character_refs was NOT found in the checkpoint "
              "after the crash.")

    inspect_checkpoint_db(conn)

    # Explicit commit, then re-check the SAME connection -- proves whether
    # this was an uncommitted-transaction issue or a genuinely missing write.
    conn.commit()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (TEST_THREAD_ID,))
    print(f"[POST-COMMIT CHECK] rows for {TEST_THREAD_ID!r} after explicit conn.commit(): "
          f"{cur.fetchone()[0]}")

    conn.close()

    # --- control test: same node, but clean interrupt, no exception ---
    run_interrupt_control_test(project_state, config, initial_state)

    print(f"\nDone. Delete {TEST_GENERATION_DB} (and its -wal/-shm siblings, "
          f"if present) when you're done inspecting.")


if __name__ == "__main__":
    main()