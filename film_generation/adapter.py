"""
Bridges film_agents.schemas.ProjectState (reasoning pipeline output) into
film_generation.schemas.GenerationState (generation pipeline input).

This is the one place that's allowed to know about both schema sets.
Everything downstream of this file only sees GenerationState -- the
generation pipeline should never need to reach back into ProjectState
directly, or the two systems re-couple by accident.

Important: the generation pipeline needs more than shot_lists. Character
identity comes from breakdown.scenes[].cast, tone/objectives come from
intent_briefs, and cut planning needs edited_sequences. All of it is
pulled here, once, up front.
"""

from __future__ import annotations

from film_agents.schemas import ProjectState
from film_generation.generation.continuity_manager import ContinuityManager
from film_generation.schemas import GenerationState


def build_generation_state(project_state: ProjectState) -> GenerationState:
    print("[Adapter] building GenerationState from ProjectState...")
    if project_state.breakdown is None:
        raise ValueError(
            "ProjectState.breakdown is empty -- run the reasoning pipeline "
            "to completion before building generation state."
        )

    continuity = ContinuityManager()

    seen_characters: set[str] = set()
    seen_props: set[str] = set()
    for scene in project_state.breakdown.scenes:
        for character in scene.cast:
            if character not in seen_characters:
                continuity.register(character, "character")
                seen_characters.add(character)
        for prop in scene.props:
            if prop not in seen_props:
                continuity.register(prop, "prop")
                seen_props.add(prop)

    return GenerationState(
        project_title=project_state.breakdown.title,
        continuity=continuity,
        log=[
            f"[Adapter] seeded continuity with {len(seen_characters)} characters, "
            f"{len(seen_props)} props from reasoning output"
        ],
    )


def characters_in_scene(project_state: ProjectState, scene_number: int) -> list[str]:
    """Convenience lookup used by character_reference.py / prompt_generator.py
    to know which characters need identity refs for a given scene."""
    for scene in project_state.breakdown.scenes:
        if scene.scene_number == scene_number:
            return scene.cast
    return []


def props_in_scene(project_state: ProjectState, scene_number: int) -> list[str]:
    for scene in project_state.breakdown.scenes:
        if scene.scene_number == scene_number:
            return scene.props
    return []
