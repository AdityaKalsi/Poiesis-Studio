"""
LangGraph wiring for the generation pipeline, mirroring the structure of
film_agents/graph.py:

    character_reference + scene_anchor -> prompt_generation
        -> shot_generation -> visual_critic
            --(any shot fails, retries left)--> shot_generation  [loop]
            --(all pass, or out of retries)--> image2video
        -> cut_planning -> video_editing -> END

Nodes process all outstanding items per call (same batch-per-node idiom
as film_agents/graph.py's node_cinematography), rather than one shot per
graph step -- keeps the graph shape identical to the reasoning pipeline's
and keeps per-node LLM/model call batching efficient.

project_state (reasoning pipeline output) and config are closed over via
build_generation_graph() rather than stored in GenerationState, since
they're read-only for the duration of a generation run -- only
GenerationState (schemas.py) is meant to be graph-mutable state.

NOTE: character_reference, scene_anchor, shot_generator, visual_critic,
and image2video currently raise NotImplementedError (see their modules)
pending a model-backend decision. This graph is structurally complete
and reviewable now; running it end-to-end requires those to be wired.
"""


from __future__ import annotations

from pathlib import Path

from langgraph.graph import StateGraph, END
from pydantic import config


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
from film_generation.schemas import GenerationState
from film_generation.generation.continuity_manager import ContinuityManager 
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer



def build_generation_graph(project_state: ProjectState, config: GenerationConfig):

    brief_by_scene = {b.scene_number: b for b in project_state.intent_briefs}
    scene_by_num = {s.scene_number: s for s in project_state.breakdown.scenes}
    shot_list_by_scene = {sl.scene_number: sl for sl in project_state.shot_lists}
    edited_seq_by_scene = {seq.scene_number: seq for seq in project_state.edited_sequences}

    assets = AssetManager(config.output_dir)

    # -- nodes ----------------------------------------------------------

    def node_character_reference(state: GenerationState) -> dict:
        print(f"[Character Reference] Generating references for {len(project_state.breakdown.scenes)} scenes")
        refs = dict(state.character_refs)
        
        for scene in project_state.breakdown.scenes:
            for character in scene.cast:
                if character in refs:
                    continue
                description=", ".join(scene.wardrobe_notes) or "no description available"
                print(description)
                ref = char_ref_module.generate_character_reference(
                    character_name=character,
                    description=description,
                    config=config,
                    assets=assets,
                )
                refs[character] = ref
        print("\n" + "="*80 + "\n")
        return {
            "character_refs": refs,
            "log": state.log + [f"[Character Reference] {len(refs)} characters resolved"],
        }

    def node_scene_anchor(state: GenerationState) -> dict:
        print(f"[Scene Anchor] Generating anchors for {len(project_state.breakdown.scenes)} scenes")
        anchors = dict(state.scene_anchors)
        for scene in project_state.breakdown.scenes:
            if scene.scene_number in anchors:
                continue
            brief = brief_by_scene[scene.scene_number]
            anchor = anchor_module.generate_scene_anchor(
                scene=scene, visual_objective=brief.visual_objective, config=config, assets=assets
            )
            anchors[scene.scene_number] = anchor
        print("\n" + "="*80 + "\n")
        return {
            "scene_anchors": anchors,
            "log": state.log + [f"[Scene Anchor] {len(anchors)} scenes resolved"],
        }

    def node_prompt_generation(state: GenerationState) -> dict:
        print(f"[Prompt Generation] Generating prompts for scenes")
        failing_feedback = {
            f"{c.scene_number}-{c.shot_number}": c.feedback
            for c in state.visual_critiques
            if not c.passes
        }
        prompts = []
        revision_counts = dict(state.shot_revision_counts)
        continuity = ContinuityManager(state.continuity)
        for scene_number, shot_list in shot_list_by_scene.items():
            brief = brief_by_scene[scene_number]
            cast = characters_in_scene(project_state, scene_number)
            props = props_in_scene(project_state, scene_number)
            anchor = state.scene_anchors.get(scene_number)
            for shot in shot_list.shots:
                key = f"{scene_number}-{shot.shot_number}"
                prior_feedback = failing_feedback.get(key)
                # Skip shots that already passed and aren't being revised
                already_passed = any(
                    c.scene_number == scene_number
                    and c.shot_number == shot.shot_number
                    and c.passes
                    for c in state.visual_critiques
                )
                if already_passed and key not in failing_feedback:
                    continue
                if prior_feedback is not None:
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
                    assets=assets
                )
                prompts.append(prompt)
        print(f"[Prompt Generation] {prompts} prompts generated for {len(prompts)} shots")
        print("\n" + "="*80 + "\n")
        return {
            "shot_prompts": prompts,
            "shot_revision_counts": revision_counts,
            "log": state.log + [f"[Prompt Generation] {len(prompts)} prompts (re)built"],
        }

    def node_shot_generation(state: GenerationState) -> dict:
        print(f"[Shot Generation] Generating images for {len(state.shot_prompts)} shots")
        images = list(state.generated_images)
        images_by_key = {f"{i.scene_number}-{i.shot_number}": i for i in images}
        for prompt in state.shot_prompts:
            key = f"{prompt.scene_number}-{prompt.shot_number}"
            seed = config.defaults.seed or hash(key) % (2**31)
            image = shot_gen_module.generate_shot_image(
                shot_prompt=prompt,
                character_refs=state.character_refs,
                scene_anchor=state.scene_anchors.get(prompt.scene_number),
                config=config,
                assets=assets,
                seed=seed,
            )
            images_by_key[key] = image
        
        print("\n" + "="*80 + "\n")
        return {
            "generated_images": list(images_by_key.values()),
            "generation_call_count": state.generation_call_count + len(state.shot_prompts),
            "log": state.log + [f"[Shot Generation] {len(state.shot_prompts)} images generated"],
        }


    def node_visual_critic(state: GenerationState) -> dict:
        print(f"[Visual Critic] Critiquing {len(state.generated_images)} shots")
        critiques = []
        for image in state.generated_images:
            brief = brief_by_scene[image.scene_number]
            critique = critic_module.critique_shot_image(
                image=image,
                brief=brief,
                character_refs=state.character_refs,
                config=config,
            )
            critiques.append(critique)
        print(f"[Visual Critic] {critiques} images critiqued")
        print("\n" + "="*80 + "\n")
        return {
            "visual_critiques": critiques,
            "log": state.log + [f"[Visual Critic] scored {len(critiques)} images"],
        }


    def route_after_visual_critic(state: GenerationState) -> str:
        any_failing = any(not c.passes for c in state.visual_critiques)
        max_revision = max(state.shot_revision_counts.values(), default=0)
        if any_failing and max_revision < config.retries.max_shot_revisions:
            return "revise"
        return "advance"


    def node_image2video(state: GenerationState) -> dict:
        print(f"[Image2Video] Generating clips for {len(state.generated_images)} shots")
        clips = list(state.generated_clips)
        clips_by_key = {f"{c.scene_number}-{c.shot_number}": c for c in clips} #checking if a clip already exists for a shot using dictionary and not lists because it is O(1) instead of O(n)
        passing_keys = {f"{c.scene_number}-{c.shot_number}" for c in state.visual_critiques if c.passes}
        for image in state.generated_images:
            key = f"{image.scene_number}-{image.shot_number}"
            if key not in passing_keys or key in clips_by_key:
                continue
            shot = next(
                s for s in shot_list_by_scene[image.scene_number].shots
                if s.shot_number == image.shot_number
            )
            clip = i2v_module.animate(image=image, shot=shot.movement, config=config, assets=assets)
            clips_by_key[key] = clip
        print("\n" + "="*80 + "\n")
        return {
            "generated_clips": list(clips_by_key.values()),
            "log": state.log + [f"[Image2Video] {len(clips_by_key)} clips available"],
        }


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
        print("\n" + "="*80 + "\n")
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

    graph.add_node("character_reference", node_character_reference)
    graph.add_node("scene_anchor", node_scene_anchor)
    graph.add_node("prompt_generation", node_prompt_generation)
    graph.add_node("shot_generation", node_shot_generation)
    graph.add_node("visual_critic", node_visual_critic)
    graph.add_node("image2video", node_image2video)
    graph.add_node("cut_planning", node_cut_planning)
    graph.add_node("video_editing", node_video_editing)

    graph.set_entry_point("character_reference")
    graph.add_edge("character_reference", "scene_anchor")
    graph.add_edge("scene_anchor", "prompt_generation")
    graph.add_edge("prompt_generation", "shot_generation")
    graph.add_edge("shot_generation", "visual_critic")

    graph.add_conditional_edges(
        "visual_critic",
        route_after_visual_critic,
        {"revise": "prompt_generation", "advance": "image2video"},
    )

    graph.add_edge("image2video", "cut_planning")
    graph.add_edge("cut_planning", "video_editing")
    graph.add_edge("video_editing", END)

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
    conn = sqlite3.connect(database='generation.db', check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn,serde=serializer)
    return graph.compile(checkpointer=checkpointer)


def run_generation(
    project_state: ProjectState,
    config: GenerationConfig | None = None,
    resume: bool = False,
    thread_id: str = "percy_jackson",
):
    if config is None:
        config = load_config()

    graph = build_generation_graph(project_state, config)
    thread_config = {"configurable": {"thread_id": thread_id}}

    existing = graph.get_state(thread_config)
    has_checkpoint = existing is not None and bool(existing.values)

    print(f"Resume requested: {resume} | checkpoint found: {has_checkpoint} | thread_id: {thread_id}")

    if resume and has_checkpoint:
        # None as input tells LangGraph "don't overwrite state, just continue
        # the existing thread from its last saved checkpoint."
        final_state_dict = graph.invoke(None, config=thread_config)
    else:
        if resume and not has_checkpoint:
            print("  -> resume=True but no checkpoint exists for this thread_id; starting fresh.")
        initial_state = build_generation_state(project_state)
        final_state_dict = graph.invoke(initial_state, config=thread_config)

    png = graph.get_graph().draw_mermaid_png()

    with open("generation_graph.png", "wb") as f:
        f.write(png)

    print("Graph saved as generation_graph.png")
    return GenerationState(**final_state_dict)