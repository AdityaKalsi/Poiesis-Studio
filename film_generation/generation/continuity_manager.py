"""
ContinuityManager tracks the evolving state of every character and prop
across the film -- wardrobe, condition, last-seen location -- and
supplies that context to prompt_generator before each shot is generated.

Architectural note: this is deliberately a plain class instantiated once
per run and threaded through GenerationState (see schemas.py), NOT a
pipeline stage that runs once and hands off a static output like
character_reference.py or scene_anchor.py. Character/prop state changes
*during* generation (a character gets rained on in scene 4; that wetness
should still show in scene 5 until they change), so this has to be
queryable and mutable throughout the run, not produced once up front.
"""

from __future__ import annotations
from typing import Optional

from film_generation.schemas import EntityState


class ContinuityManager:
    def __init__(self) -> None:
        self._state: dict[str, EntityState] = {}

    # -- initialization -----------------------------------------------------

    def register(self, entity_name: str, entity_type: str) -> None:
        """Call once per character/prop when the run starts, seeded from
        the reasoning pipeline's ScriptBreakdown.cast / .props."""
        if entity_name not in self._state:
            self._state[entity_name] = EntityState(
                entity_name=entity_name, entity_type=entity_type
            )

    # -- read -----------------------------------------------------------

    def get_state(self, entity_name: str) -> Optional[EntityState]:
        return self._state.get(entity_name)

    def get_context_for_shot(self, cast: list[str], props: list[str]) -> str:
        """Renders current continuity state for the entities appearing in
        a given shot, as a prompt-ready text block. prompt_generator.py
        appends this to the enriched prompt so wardrobe/condition carries
        forward automatically instead of being re-specified per shot."""
        lines: list[str] = []
        for name in cast + props:
            state = self._state.get(name)
            if state is None:
                continue
            parts = []
            if state.wardrobe:
                parts.append(f"wearing {state.wardrobe}")
            if state.condition:
                parts.append(state.condition)
            if parts:
                lines.append(f"{name}: {', '.join(parts)}")
        return "\n".join(lines)

    # -- write ----------------------------------------------------------

    def update_state(
        self,
        entity_name: str,
        *,
        wardrobe: Optional[str] = None,
        condition: Optional[str] = None,
        scene_number: Optional[int] = None,
        location: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        state = self._state.get(entity_name)
        if state is None:
            # Prop/character wasn't pre-registered from the script
            # breakdown -- register on the fly rather than dropping the
            # update, but this is worth logging upstream as a data gap.
            state = EntityState(entity_name=entity_name, entity_type="prop")
            self._state[entity_name] = state

        if wardrobe is not None:
            state.wardrobe = wardrobe
        if condition is not None:
            state.condition = condition
        if scene_number is not None:
            state.last_seen_scene = scene_number
        if location is not None:
            state.last_seen_location = location
        if note is not None:
            state.notes.append(note)

    # -- consistency checking --------------------------------------------

    def check_consistency(self, entity_name: str, image_embedding) -> float:
        """Compares a generated shot's embedding against the entity's
        reference embedding. Returns a similarity score in [0, 1].

        NOT IMPLEMENTED YET: depends on which embedding model backs the
        chosen consistency strategy (config.consistency.strategy). Wire
        this once models/embeddings.py exists -- visual_critic.py calls
        this to catch identity drift, which is the generation-side
        equivalent of the reasoning pipeline's critic_agent check.
        """
        raise NotImplementedError(
            "check_consistency requires an embedding backend -- see "
            "models/embeddings.py (not yet implemented)."
        )

    def snapshot(self) -> dict[str, EntityState]:
        """Full state dump, useful for debugging/logging a run."""
        return dict(self._state)
