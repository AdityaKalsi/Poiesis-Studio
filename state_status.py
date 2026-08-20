from film_generation.pipeline import build_generation_graph
from film_generation.config import load_config
from run_movie import load_completed_reasoning_state

# Change this whenever needed
GENERATION_THREAD_ID = "percy_jackson"


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_key(obj):
    """Readable scene-shot key for generated items."""
    if hasattr(obj, "scene_number") and hasattr(obj, "shot_number"):
        return f"{obj.scene_number}-{obj.shot_number}"

    if hasattr(obj, "scene_number"):
        return f"Scene {obj.scene_number}"

    return str(obj)


# ---------------------------------------------------------
# Load graph
# ---------------------------------------------------------

reasoning_state = load_completed_reasoning_state()
config = load_config()

graph = build_generation_graph(reasoning_state, config)

thread_config = {
    "configurable": {
        "thread_id": GENERATION_THREAD_ID
    }
}

state = graph.get_state(thread_config)

if state is None or not state.values:
    print(f"No checkpoint found for thread: {GENERATION_THREAD_ID}")
    exit()


values = state.values


# ---------------------------------------------------------
# Basic checkpoint info
# ---------------------------------------------------------

print_section("GENERATION PIPELINE STATUS")

print(f"Thread ID     : {GENERATION_THREAD_ID}")
print(f"Pending nodes : {state.next or 'None - pipeline completed'}")


# ---------------------------------------------------------
# Character References
# ---------------------------------------------------------

print_section("CHARACTER REFERENCES")

character_refs = values.get("character_refs", {})

print(f"Count: {len(character_refs)}")

if character_refs:
    for name, ref in character_refs.items():
        print(f"  ✓ {name}")
else:
    print("  None generated/checkpointed")


# ---------------------------------------------------------
# Scene Anchors
# ---------------------------------------------------------

print_section("SCENE ANCHORS")

scene_anchors = values.get("scene_anchors", {})

print(f"Count: {len(scene_anchors)}")

if scene_anchors:
    for scene_number in scene_anchors:
        print(f"  ✓ Scene {scene_number}")
else:
    print("  None generated")


# ---------------------------------------------------------
# Shot Prompts
# ---------------------------------------------------------

print_section("SHOT PROMPTS")

shot_prompts = values.get("shot_prompts", [])

print(f"Count: {len(shot_prompts)}")

for prompt in shot_prompts:
    print(f"  ✓ {get_key(prompt)}")


# ---------------------------------------------------------
# Generated Images
# ---------------------------------------------------------

print_section("GENERATED IMAGES")

generated_images = values.get("generated_images", {})

print(f"Count: {len(generated_images)}")

if isinstance(generated_images, dict):
    for key in generated_images:
        print(f"  ✓ {key}")
else:
    for image in generated_images:
        print(f"  ✓ {get_key(image)}")


# ---------------------------------------------------------
# Image Generation Failures
# ---------------------------------------------------------

print_section("SHOT GENERATION FAILURES")

failures = values.get("shot_generation_failures", {})

active_failures = {
    key: failure
    for key, failure in failures.items()
    if getattr(failure, "reason_code", None)
}

print(f"Active failures: {len(active_failures)}")

if active_failures:
    for key, failure in active_failures.items():
        print(
            f"  ✗ {key} | "
            f"reason={failure.reason_code} | "
            f"attempts={failure.attempt_count}"
        )
else:
    print("  No active failures")


# ---------------------------------------------------------
# Visual Critiques
# ---------------------------------------------------------

print_section("VISUAL CRITIQUES")

critiques = values.get("visual_critiques", [])

passed = [c for c in critiques if c.passes]
failed = [c for c in critiques if not c.passes]

print(f"Total  : {len(critiques)}")
print(f"Passed : {len(passed)}")
print(f"Failed : {len(failed)}")

for critique in critiques:
    status = "✓ PASS" if critique.passes else "✗ FAIL"

    print(
        f"  {status} | "
        f"{critique.scene_number}-{critique.shot_number}"
    )


# ---------------------------------------------------------
# Revision Counts
# ---------------------------------------------------------

print_section("SHOT REVISION COUNTS")

revision_counts = values.get("shot_revision_counts", {})

if revision_counts:
    for key, count in revision_counts.items():
        print(f"  {key}: {count}")
else:
    print("  No revisions yet")


# ---------------------------------------------------------
# Generated Video Clips
# ---------------------------------------------------------

print_section("GENERATED VIDEO CLIPS")

generated_clips = values.get("generated_clips", {})

print(f"Count: {len(generated_clips)}")

if isinstance(generated_clips, dict):
    for key in generated_clips:
        print(f"  ✓ {key}")
else:
    for clip in generated_clips:
        print(f"  ✓ {get_key(clip)}")


# ---------------------------------------------------------
# Edit Decision Lists
# ---------------------------------------------------------

print_section("EDIT DECISION LISTS")

edls = values.get("edit_decision_lists", [])

print(f"Count: {len(edls)}")

for edl in edls:
    print(f"  ✓ Scene {edl.scene_number}")


# ---------------------------------------------------------
# Final Movie
# ---------------------------------------------------------

print_section("FINAL OUTPUT")

final_movie = values.get("final_movie_path")

if final_movie:
    print(f"✓ Final movie: {final_movie}")
else:
    print("Final movie not generated yet")


# ---------------------------------------------------------
# Logs
# ---------------------------------------------------------

print_section("PIPELINE LOG")

logs = values.get("log", [])

if logs:
    for entry in logs[-10:]:
        print(f"  {entry}")
else:
    print("  No logs")


print("\n" + "=" * 70)
print("END OF STATE")
print("=" * 70)