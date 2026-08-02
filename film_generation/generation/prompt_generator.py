"""
Converts a single Shot (from the reasoning pipeline's ShotList) into a
diffusion-ready ShotPrompt: enriched with camera/lighting/style vocabulary,
grounded with character/scene identity references, and carrying forward
whatever the ContinuityManager currently knows about wardrobe/condition
for the entities in frame.

This is plain composition logic, not an LLM call -- the reasoning
pipeline already decided *what* each shot should show (agents.py /
cinematography_agent). This file's job is turning that structured
decision into the prose a T2I model needs, deterministically and
cheaply, so re-running it during a shot retry doesn't cost an LLM call.
"""

from __future__ import annotations

from film_agents.schemas import Shot, SceneIntentBrief
from film_generation.generation.continuity_manager import ContinuityManager
from film_generation.schemas import CharacterReference, SceneAnchor, ShotPrompt
from film_generation.generation.asset_manager import AssetManager
from film_generation.utils.prompt_utils import (
    DEFAULT_NEGATIVE_PROMPT,
    SHOT_TYPE_LANGUAGE,
    lens_hint,
    lighting_hint_for_tone,
)


def generate_shot_prompt(
    scene_number: int,
    shot: Shot,
    brief: SceneIntentBrief,
    character_refs: dict[str, CharacterReference],
    scene_anchor: SceneAnchor | None,
    continuity: ContinuityManager,
    cast: list[str],
    props: list[str],
    prior_feedback: str | None = None,
    assets: AssetManager | None = None,
) -> ShotPrompt:
    print(f"Generating prompt for scene {scene_number}, shot {shot.shot_number}")
    shot_language = SHOT_TYPE_LANGUAGE.get(shot.shot_type, shot.shot_type)
    lighting = lighting_hint_for_tone(brief.tone)
    lens = lens_hint(shot.lens_mm)
    continuity_context = continuity.get_context_for_shot(cast, props)
    
    referenced_characters = [name for name in cast if name in character_refs]
    parts = [
        shot.description,
        shot_language,
        f"tone: {brief.tone}",
        lighting,
    ]
    
    if lens:
        parts.append(lens)
    if shot.movement:
        parts.append(f"camera movement: {shot.movement}")
    if scene_anchor is not None:
        parts.append(f"environment consistent with: {scene_anchor.layout_description}")
    if continuity_context:
        parts.append(f"continuity: {continuity_context}")
    if prior_feedback:
        parts.append(f"revision notes to address: {prior_feedback}")

    positive_prompt = ", ".join(p for p in parts if p)

    shotprompt = ShotPrompt(
        scene_number=scene_number,
        shot_number=shot.shot_number,
        positive_prompt=positive_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        character_ref_ids=referenced_characters,
        scene_anchor_id=scene_anchor.scene_number if scene_anchor else None,
    )

    
    json_path = (
        assets.output_dir
        / f"scene_{scene_number}"
        / f"shot_{shot.shot_number}_prompt.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(shotprompt.model_dump_json(indent=2))

    return shotprompt
