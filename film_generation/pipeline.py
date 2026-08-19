"""
LangGraph wiring for the generation pipeline, mirroring the structure of
film_agents/graph.py:

    character_reference + scene_anchor -> prompt_generation
        -> shot_generation -> visual_critic
            --(any shot fails, retries left)--> prompt_generation  [loop]
            --(all pass, or out of retries)--> image2video
        -> cut_planning -> video_editing -> END

FAN-OUT / JOIN PATTERN
-----------------------
character_reference, scene_anchor, shot_generation, visual_critic, and
image2video each process one item per Send task (one character, one
scene, one shot) so the paid model calls for a batch run in parallel
instead of serially.

Every Send-fanned node is followed by a plain "join" node before the next
conditional-edge router. This matters: a conditional edge attached
directly to a Send-fanned node gets evaluated once PER PARALLEL TASK
INSTANCE, not once total -- so N parallel character-reference tasks would
each independently re-run the next router and re-issue Sends for the
following stage, multiplying the work N times over. A plain node with a
plain edge into it collapses that back down to one evaluation per
superstep, because LangGraph only schedules a plain node once per step no
matter how many upstream tasks triggered it.

RESUME / IDEMPOTENCY
---------------------
LangGraph's checkpointer commits state at superstep granularity, not per
Send task. If shot 4 of 7 raises inside the shot_generation superstep,
the WHOLE superstep is discarded -- shots 1-3's results never reach the
checkpoint, even though they succeeded. Resuming re-enters at the same
pending node and the router re-issues Send for all 7 shots again.

To make that cheap instead of wasteful, every generate_* node below first
calls _cache_lookup(), which is meant to short-circuit the paid API call
if the artifact already exists on disk (keyed by the same cache_key/seed
your GeneratedImage schema already carries). It currently degrades to
"always regenerate" via hasattr guards -- wire has_cached()/load_cached()
(or whatever your AssetManager actually exposes) to get real skip-on-
resume behavior. Once that's wired, `resume=True` gives you the practical
equivalent of "continue from shot 5": shots 1-3 return from cache almost
instantly, shot 4 (and 5-7) actually call the model.

Each generate_* node also wraps its model call in try/except and re-raises
with the specific scene/shot/character in the message, so a failed run
tells you exactly what to fix before you resume -- rather than a bare
stack trace from three layers down in the model backend.
"""


from __future__ import annotations

from pathlib import Path

from click import prompt
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from film_generation.models.text2image import GenerationBlockedError
from film_generation.utils.prompt_utils import remediation_feedback_for

from film_agents.schemas import ProjectState
from film_generation.adapter import build_generation_state, characters_in_scene, props_in_scene
from film_generation.config import GenerationConfig, load_config
from film_generation.generation import (
    character_reference as char_ref_module,
    cut_planner,
    scene_anchor as anchor_module,
    shot_generator as shot_gen_module,
    video_editor,
    visual_critic as critic_module,
)
from film_generation.models import image2video as i2v_module
from film_generation.generation.asset_manager import AssetManager
from film_generation.generation.prompt_generator import generate_shot_prompt
from film_generation.schemas import GenerationState,ShotGenerationFailure
from film_generation.generation.continuity_manager import ContinuityManager
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from film_generation.schemas import GenerationState, ShotGenerationFailure, CharacterReference, SceneAnchor, GeneratedImage, GeneratedClip


def _cache_lookup(assets: AssetManager, cache_key: str):
    """Best-effort skip of a paid generation call when this exact artifact
    was already produced (e.g. on a previous attempt at the same superstep
    that later failed on a different item). Requires AssetManager to expose
    has_cached(key) / load_cached(key) -- wire these to whatever your asset
    manager already tracks on disk. Returns None (== "generate it") until
    then, so this is safe to leave in place before that's implemented.
    """
    if hasattr(assets, "has_cached") and hasattr(assets, "load_cached"):
        if assets.has_cached(cache_key):
            return assets.load_cached(cache_key)
    return None


def build_generation_graph(reasoning_state: ProjectState, config: GenerationConfig):

    brief_by_scene = {b.scene_number: b for b in reasoning_state.intent_briefs}
    scene_by_num = {s.scene_number: s for s in reasoning_state.breakdown.scenes}
    shot_list_by_scene = {sl.scene_number: sl for sl in reasoning_state.shot_lists}
    edited_seq_by_scene = {seq.scene_number: seq for seq in reasoning_state.edited_sequences}

    assets = AssetManager(config.output_dir)

    # -- character reference ---------------------------------------------

    def route_to_character_refs(state: GenerationState):
        print(f"[Character Reference] Checking for {len(reasoning_state.breakdown.scenes)} scenes")
        characters_to_generate: dict[str, str] = {}
        for scene in reasoning_state.breakdown.scenes:
            description = ", ".join(scene.wardrobe_notes) or "no description available"
            for character in scene.cast:
                if character in state.character_refs:
                    continue
                if character not in characters_to_generate:
                    characters_to_generate[character] = description
        if not characters_to_generate:
            return route_to_scene_anchors(state)  # nothing to do, delegate straight through

        return [
            Send("generate_character_reference", {"character_name": c, "description": d})
            for c, d in characters_to_generate.items()
        ]

    def node_generate_character_reference(payload: dict) -> dict:
        name = payload["character_name"]
        cache_key = f"charref-{name}"
        cached = _cache_lookup(assets, cache_key)
        if cached is not None:
            print(f"[Character Reference] Cache hit for {name}, skipping regeneration")
            return {"character_refs": {name: CharacterReference(**cached)}}

        print("\n" + "=" * 80 + "\n")
        print(f"[Character Reference] Generating reference for {name}")
        try:
            ref = char_ref_module.generate_character_reference(
                character_name=name, description=payload["description"], config=config, assets=assets,
            )
        except Exception as exc:
            raise RuntimeError(f"[Character Reference] failed for {name!r}: {exc}") from exc
        assets.save_cache(cache_key, ref.model_dump())
        print("\n" + "=" * 80 + "\n")
        return {"character_refs": {name: ref}}

    def node_join_character_refs(state: GenerationState) -> dict:
        return {}

    # -- scene anchor -------------------------------------------------------

    def route_to_scene_anchors(state: GenerationState):
        print(f"[Scene Anchor] Checking for {len(reasoning_state.breakdown.scenes)} scenes")
        scenes_to_generate: dict[int, tuple] = {}
        for scene in reasoning_state.breakdown.scenes:
            if scene.scene_number in state.scene_anchors:
                continue
            if scene.scene_number not in scenes_to_generate:
                brief = brief_by_scene[scene.scene_number]
                scenes_to_generate[scene.scene_number] = (scene, brief.visual_objective)
        if not scenes_to_generate:
            return "prompt_generation"  # nothing to do, skip straight through

        return [
            Send("generate_scene_anchor", {"scene_number": sn, "scene": scene, "visual_objective": vo})
            for sn, (scene, vo) in scenes_to_generate.items()
        ]

    def node_generate_scene_anchor(payload: dict) -> dict:
        scene_number = payload["scene_number"]
        cache_key = f"anchor-{scene_number}"
        cached = _cache_lookup(assets, cache_key)
        if cached is not None:
            print(f"[Scene Anchor] Cache hit for scene {scene_number}, skipping regeneration")
            return {"scene_anchors": {scene_number: SceneAnchor(**cached)}} 

        print("\n" + "=" * 80 + "\n")
        print(f"[Scene Anchor] Generating anchor for scene {scene_number}")
        try:
            anchor = anchor_module.generate_scene_anchor(
                scene=payload["scene"],
                visual_objective=payload["visual_objective"],
                config=config,
                assets=assets,
            )
        except Exception as exc:
            raise RuntimeError(f"[Scene Anchor] failed for scene {scene_number}: {exc}") from exc
        print("\n" + "=" * 80 + "\n")
        assets.save_cache(cache_key, anchor.model_dump())
        return {"scene_anchors": {scene_number: anchor}}

    def node_join_scene_anchors(state: GenerationState) -> dict:
        return {}

    # -- prompt generation (batch -- runs once, needs the full picture) -----

    def node_prompt_generation(state: GenerationState) -> dict:
        print("\n" + "=" * 80 + "\n")
        print(f"[Prompt Generation] Generating prompts for scenes")

        critic_feedback = {
            f"{c.scene_number}-{c.shot_number}": c.feedback
            for c in state.visual_critiques
            if not c.passes
        }
        generation_failure_feedback = {
            key: remediation_feedback_for(failure.reason_code)
            for key, failure in state.shot_generation_failures.items()
            if failure.reason_code is not None
        }
        failing_feedback = dict(critic_feedback)
        for key, remediation in generation_failure_feedback.items():

            failing_feedback[key] = (
                f"{failing_feedback[key]} | {remediation}" if key in failing_feedback else remediation
            )
        if failing_feedback:
            print(f"[Prompt Generation] Regenerating {len(failing_feedback)} shot(s) due to prior failure:")
            for key, feedback in failing_feedback.items():
                print(f"    {key}: {feedback}")
        # NEW: de-duplicated set of shots that already have a generated image
        # on disk/in-state, regardless of whether the critic has looked at it
        # yet. This is what "already handled, don't resend" should actually
        # mean at the generation-retry stage -- critique-passing is a later,
        # separate gate.
        generated_keys = {
            f"{img.scene_number}-{img.shot_number}" for img in state.generated_images
        }

        prompts = []
        revision_counts = dict(state.shot_revision_counts)
        continuity = ContinuityManager(state.continuity)
        for scene_number, shot_list in shot_list_by_scene.items():
            brief = brief_by_scene[scene_number]
            cast = characters_in_scene(reasoning_state, scene_number)
            props = props_in_scene(reasoning_state, scene_number)
            anchor = state.scene_anchors.get(scene_number)
            for shot in shot_list.shots:
                key = f"{scene_number}-{shot.shot_number}"
                prior_feedback = failing_feedback.get(key)

                # CHANGED: skip shots that already have a generated image and
                # aren't currently flagged as failing (critic OR generation).
                # Previously this checked visual_critiques.passes, which stays
                # empty until the critic stage runs -- so every generation-retry
                # loop was rebuilding prompts for shots that had already
                # succeeded, just because they hadn't been critiqued yet.
                already_handled = key in generated_keys
                if already_handled and key not in failing_feedback:
                    continue

                if key in critic_feedback:
                    revision_counts[key] = revision_counts.get(key, 0) + 1
                prompt = generate_shot_prompt(
                    scene_number=scene_number,
                    shot=shot,
                    brief=brief,
                    character_refs=state.character_refs,
                    scene_anchor=anchor,
                    continuity=continuity,
                    cast=cast,
                    props=props,
                    prior_feedback=prior_feedback,
                    assets=assets,
                )
                prompts.append(prompt)
        print(f"[Prompt Generation] Completed")
        print("\n" + "=" * 80 + "\n")
        return {
            "shot_prompts": prompts,
            "shot_revision_counts": revision_counts,
            "log": state.log + [f"[Prompt Generation] {len(prompts)} prompts (re)built"],
        }

    # -- shot generation ----------------------------------------------------

    def route_to_shot_generation(state: GenerationState):
        print(f"[Shot Generation] Checking for {len(state.shot_prompts)} shot prompts")
        if not state.shot_prompts:
            return route_to_visual_critic(state)

        return [
            Send("generate_shot_image", {
                "prompt": prompt,
                "character_refs": state.character_refs,
                "scene_anchor": state.scene_anchors.get(prompt.scene_number),
                "prior_attempt_count": state.shot_generation_failures.get(
                    f"{prompt.scene_number}-{prompt.shot_number}",
                    ShotGenerationFailure(scene_number=prompt.scene_number, shot_number=prompt.shot_number),
                ).attempt_count,
            })
            for prompt in state.shot_prompts
        ]

    def node_generate_shot_image(payload: dict) -> dict:
        prompt = payload["prompt"]
        key = f"{prompt.scene_number}-{prompt.shot_number}"
        seed = config.defaults.seed or hash(key) % (2**31)
        cache_key = f"shot-{key}-{seed}"

        cached = _cache_lookup(assets, cache_key)
        if cached is not None:
            print(f"[Shot Generation] Cache hit for shot {key}, skipping regeneration")
            return {"generated_images": {key: GeneratedImage(**cached)}, "generation_call_count": 0}

        prior_attempts = payload.get("prior_attempt_count", 0)

        print("\n" + "=" * 80 + "\n")
        print(f"[Shot Generation] Generating image for shot {key}")
        try:
            image = shot_gen_module.generate_shot_image(
                shot_prompt=prompt,
                character_refs=payload["character_refs"],
                scene_anchor=payload["scene_anchor"],
                config=config,
                assets=assets,
                seed=seed,
            )
        except GenerationBlockedError as exc:
            print(f"[Shot Generation] blocked on shot {key}: reason={exc.reason_code}")
            print(f"    prompt text: {prompt.positive_prompt[:300]}")
            return {
                "shot_generation_failures": {
                    key: {
                        "scene_number": prompt.scene_number,
                        "shot_number": prompt.shot_number,
                        "reason_code": exc.reason_code,
                        "raw_message": str(exc),
                        "attempt_count": prior_attempts + 1,
                    }
                }
            }
        except Exception as exc:
            raise RuntimeError(f"[Shot Generation] failed on shot {key}: {exc}") from exc

        assets.save_cache(cache_key, image.model_dump())
        print("\n" + "=" * 80 + "\n")
        result = {"generated_images": {key: image}, "generation_call_count": 1}
        if prior_attempts:
            result["shot_generation_failures"] = {
                key: {"scene_number": prompt.scene_number, "shot_number": prompt.shot_number,
                    "reason_code": None, "raw_message": "", "attempt_count": prior_attempts}
            }
        return result

    def route_after_shot_generation(state: GenerationState):
        still_failing = {
            k: f for k, f in state.shot_generation_failures.items() if f.reason_code is not None
        }
        if not still_failing:
            return route_to_visual_critic(state)

        max_gen_retries = config.retries.max_shot_generation_retries  # add to config
        exhausted = [k for k, f in still_failing.items() if f.attempt_count > max_gen_retries]
        if exhausted:
            print(f"[Shot Generation] giving up on {exhausted} after {max_gen_retries} retries")
            # these shots proceed with no image; route_to_visual_critic already
            # only critiques state.generated_images, so they're naturally skipped

        retryable = {k: f for k, f in still_failing.items() if f.attempt_count <= max_gen_retries}
        if not retryable:
            return route_to_visual_critic(state)

        return "prompt_generation"

    def node_join_shot_images(state: GenerationState) -> dict:
        return {}

    # -- visual critic --------------------------------------------------------

    def route_to_visual_critic(state: GenerationState):
        print(f"[Visual Critic] Checking for {len(state.generated_images)} shots")
        if not state.generated_images:
            return route_to_image2video(state)  # nothing to do, delegate straight through

        return [
            Send("critique_shot_image", {
                "image": image,
                "brief": brief_by_scene[image.scene_number],
                "character_refs": state.character_refs,
            })
            for image in state.generated_images
        ]

    def node_critique_shot_image(payload: dict) -> dict:
        image = payload["image"]
        key = f"{image.scene_number}-{image.shot_number}"
        print(f"[Visual Critic] Critiquing shot {key}")
        try:
            critique = critic_module.critique_shot_image(
                image=image,
                brief=payload["brief"],
                character_refs=payload["character_refs"],
                config=config,
            )
        except Exception as exc:
            raise RuntimeError(f"[Visual Critic] failed on shot {key}: {exc}") from exc
        print(f"[Visual Critic] {critique}")
        print("\n" + "=" * 80 + "\n")
        return {"visual_critiques": [critique]}

    def node_join_critiques(state: GenerationState) -> dict:
        return {}

    def route_after_visual_critic(state: GenerationState):
        any_failing = any(not c.passes for c in state.visual_critiques)
        max_revision = max(state.shot_revision_counts.values(), default=0)
        if any_failing and max_revision < config.retries.max_shot_revisions:
            return "prompt_generation"
        return route_to_image2video(state)

    # -- image2video ------------------------------------------------------------

    def route_to_image2video(state: GenerationState):
        passing_keys = {f"{c.scene_number}-{c.shot_number}" for c in state.visual_critiques if c.passes}
        clips_by_key = {f"{c.scene_number}-{c.shot_number}": c for c in state.generated_clips}
        to_animate = []
        for image in state.generated_images:
            key = f"{image.scene_number}-{image.shot_number}"
            if key not in passing_keys or key in clips_by_key:
                continue
            shot = next(
                s for s in shot_list_by_scene[image.scene_number].shots
                if s.shot_number == image.shot_number
            )
            to_animate.append((image, shot))

        print(f"[Image2Video] Checking for {len(to_animate)} shots to animate")
        if not to_animate:
            return "cut_planning"  # nothing to do, skip straight through

        return [
            Send("animate_shot", {"image": image, "movement": shot.movement})
            for image, shot in to_animate
        ]

    def node_animate_shot(payload: dict) -> dict:
        image = payload["image"]
        key = f"{image.scene_number}-{image.shot_number}"
        cache_key = f"clip-{key}"

        cached = _cache_lookup(assets, cache_key)
        if cached is not None:
            print(f"[Image2Video] Cache hit for shot {key}, skipping regeneration")
            return {"generated_clips": {key: GeneratedClip(**cached)}}

        print(f"[Image2Video] Generating clip for shot {key}")
        try:
            clip = i2v_module.animate(image=image, shot=payload["movement"], config=config, assets=assets)
        except Exception as exc:
            raise RuntimeError(f"[Image2Video] failed on shot {key}: {exc}") from exc
        assets.save_cache(cache_key, clip.model_dump())
        print("\n" + "=" * 80 + "\n")
        return {"generated_clips": {key: clip}}

    # -- cut planning / video editing (batch -- run once each) --------------

    def node_cut_planning(state: GenerationState) -> dict:
        print(f"[Cut Planning] Planning cuts for {len(edited_seq_by_scene)} scenes")
        edls = list(state.edit_decision_lists)
        edls_by_scene = {e.scene_number: e for e in edls}
        for scene_number, edited_sequence in edited_seq_by_scene.items():
            if scene_number in edls_by_scene:
                continue
            clips_by_shot = {
                c.shot_number: c for c in state.generated_clips if c.scene_number == scene_number
            }
            edl = cut_planner.plan_cuts(edited_sequence, clips_by_shot, fps=config.defaults.fps)
            edls_by_scene[scene_number] = edl
        print("\n" + "=" * 80 + "\n")
        return {
            "edit_decision_lists": list(edls_by_scene.values()),
            "log": state.log + [f"[Cut Planning] {len(edls_by_scene)} scene EDLs built"],
        }

    def node_video_editing(state: GenerationState) -> dict:
        scene_paths = []
        print(f"[Video Editing] Assembling {len(state.edit_decision_lists)} scenes into final movie")
        for edl in state.edit_decision_lists:
            clips_by_shot = {
                c.shot_number: c for c in state.generated_clips if c.scene_number == edl.scene_number
            }
            out_path = Path(config.output_dir) / f"scene_{edl.scene_number:03d}" / "assembled.mp4"
            video_editor.assemble_scene(edl, clips_by_shot, out_path)
            scene_paths.append(out_path)

        final_path = assets.final_movie_path(state.project_title)
        video_editor.concatenate_final_movie(scene_paths, final_path)
        return {
            "final_movie_path": str(final_path),
            "log": state.log + [f"[Video Editing] final movie written to {final_path}"],
        }

    # -- graph ------------------------------------------------------------

    graph = StateGraph(GenerationState)
    graph.add_node("generate_character_reference", node_generate_character_reference)
    graph.add_node("join_character_refs", node_join_character_refs)
    graph.add_node("generate_scene_anchor", node_generate_scene_anchor)
    graph.add_node("join_scene_anchors", node_join_scene_anchors)
    graph.add_node("prompt_generation", node_prompt_generation)
    graph.add_node("generate_shot_image", node_generate_shot_image)
    graph.add_node("join_shot_images", node_join_shot_images)
    graph.add_node("critique_shot_image", node_critique_shot_image)
    graph.add_node("join_critiques", node_join_critiques)
    graph.add_node("animate_shot", node_animate_shot)
    graph.add_node("cut_planning", node_cut_planning)
    graph.add_node("video_editing", node_video_editing)

    graph.add_conditional_edges(
        START, route_to_character_refs,
        ["generate_character_reference", "generate_scene_anchor", "prompt_generation"],
    )

    graph.add_edge("generate_character_reference", "join_character_refs")
    graph.add_conditional_edges(
        "join_character_refs", route_to_scene_anchors,
        ["generate_scene_anchor", "prompt_generation"],
    )

    graph.add_edge("generate_scene_anchor", "join_scene_anchors")
    graph.add_edge("join_scene_anchors", "prompt_generation")

    graph.add_conditional_edges(
        "prompt_generation", route_to_shot_generation,
        ["generate_shot_image", "critique_shot_image", "animate_shot", "cut_planning"],
    )

    graph.add_edge("generate_shot_image", "join_shot_images")
    graph.add_conditional_edges(
        "join_shot_images", route_after_shot_generation,
        ["critique_shot_image", "animate_shot", "cut_planning", "prompt_generation"],
    )

    graph.add_edge("critique_shot_image", "join_critiques")
    graph.add_conditional_edges(
        "join_critiques", route_after_visual_critic,
        ["prompt_generation", "animate_shot", "cut_planning"],
    )

    graph.add_edge("animate_shot", "cut_planning")
    graph.add_edge("cut_planning", "video_editing")
    graph.add_edge("video_editing", END)

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("film_generation.schemas", "GenerationState"),
            ("film_generation.schemas", "CharacterReference"),
            ("film_generation.schemas", "SceneAnchor"),
            ("film_generation.schemas", "ShotPrompt"),
            ("film_generation.schemas", "GeneratedImage"),
            ("film_generation.schemas", "VisualCritique"),
            ("film_generation.schemas", "GeneratedClip"),
            ("film_generation.schemas", "EditDecisionList"),
            ("film_generation.schemas", "ShotGenerationFailure"),
        ]
    )
    conn = sqlite3.connect(database="generation.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn, serde=serializer)
    return graph.compile(checkpointer=checkpointer)


def run_generation(
    reasoning_state: ProjectState,
    config: GenerationConfig | None = None,
    resume: bool = False,
    thread_id: str = "percy_jackson",
):
    if config is None:
        config = load_config()

    graph = build_generation_graph(reasoning_state, config)
    thread_config = {"configurable": {"thread_id": thread_id}}

    existing = graph.get_state(thread_config)
    has_checkpoint = existing is not None and bool(existing.values)

    print(f"[Generation] Resume requested: {resume} | checkpoint found: {has_checkpoint} "
          f"| thread_id: {thread_id!r}")
    if has_checkpoint:
        print(f"[Generation] checkpoint pending nodes: {existing.next}")

    try:
        if resume and has_checkpoint:
            # None input = "don't overwrite state, continue from last checkpoint".
            # LangGraph re-enters at the superstep in `existing.next`. If that
            # superstep is a Send fan-out, ALL of its Send tasks re-run (not
            # just the ones that hadn't finished before the crash) -- see the
            # module docstring above. Items whose artifacts already exist on
            # disk return quickly via _cache_lookup(); only genuinely missing
            # items pay for another model call.
            final_gen_state = graph.invoke(None, config=thread_config)
        else:
            if resume and not has_checkpoint:
                print("[Generation] resume=True but no checkpoint exists for this "
                      "thread_id; starting fresh.")
            initial_gen_state = build_generation_state(reasoning_state)
            final_gen_state = graph.invoke(initial_gen_state, config=thread_config)
    except Exception:
        stopped_at = graph.get_state(thread_config)
        print(f"\n[Generation] FAILED. Pending node(s) when it stopped: {stopped_at.next}")
        print(f"[Generation] Fix the underlying issue, then re-run with "
              f"resume=True, thread_id={thread_id!r} to continue from here.")
        raise

    png = graph.get_graph().draw_mermaid_png()
    with open("generation_graph.png", "wb") as f:
        f.write(png)
    print("Graph saved as generation_graph.png")
    return GenerationState(**final_gen_state)
