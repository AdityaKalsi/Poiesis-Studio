"""
Data contracts for the generation pipeline.

Kept separate from film_agents/schemas.py on purpose: the reasoning
pipeline and generation pipeline are two different systems with two
different failure modes (LLM reasoning errors vs. GPU/diffusion errors)
and two different cost profiles. They should only talk to each other
through the adapter (see adapter.py), never share state objects directly.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from film_generation.config import TransitionType


# ---------------------------------------------------------------------------
# Identity / environment references
# ---------------------------------------------------------------------------

class CharacterReference(BaseModel):
    character_name: str
    description: str = Field(..., description="Text description used to seed generation")
    reference_image_path: str
    # Payload shape depends on config.consistency.strategy:
    #   reference_image -> same as reference_image_path; the image itself
    #                       IS the payload, passed as multimodal input to
    #                       Gemini alongside the shot prompt (current default)
    #   ip_adapter / instantid -> path to a stored image embedding (.npy/.pt)
    #   lora                   -> path to a trained LoRA checkpoint
    #   text_only               -> None, description text is all that's reused
    identity_payload_path: Optional[str] = None
    identity_payload_kind: Literal["image", "embedding", "lora", "none"] = "none"
    seed: Optional[int] = None
    approved: bool = False  # set True once it passes the human/critic gate


class SceneAnchor(BaseModel):
    scene_number: int
    environment_image_path: str
    layout_description: str
    lighting_description: str
    seed: Optional[int] = None
    approved: bool = False


# ---------------------------------------------------------------------------
# Continuity state (mutated as generation proceeds, not produced once)
# ---------------------------------------------------------------------------

class EntityState(BaseModel):
    """Current known state of one character or prop at a point in the film.
    Continuity manager mutates this after every shot; prompt_generator
    reads it before generating the next one."""

    entity_name: str
    entity_type: Literal["character", "prop"]
    wardrobe: Optional[str] = None
    condition: Optional[str] = None  # e.g. "wet from rain", "bandaged"
    last_seen_scene: Optional[int] = None
    last_seen_location: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompting / generation artifacts
# ---------------------------------------------------------------------------

class ShotPrompt(BaseModel):
    scene_number: int
    shot_number: str
    positive_prompt: str
    negative_prompt: str = ""
    character_ref_ids: list[str] = Field(default_factory=list)
    scene_anchor_id: Optional[int] = None
    seed: Optional[int] = None


class GeneratedImage(BaseModel):
    scene_number: int
    shot_number: str
    image_path: str
    prompt_used: ShotPrompt
    model_backend: str
    seed: int
    cache_key: str


class VisualCritique(BaseModel):
    scene_number: int
    shot_number: str
    score: int = Field(..., ge=1, le=10)
    identity_similarity: Optional[float] = None
    passes: bool
    feedback: str


class GeneratedClip(BaseModel):
    scene_number: int
    shot_number: str
    video_path: str
    source_image_path: str
    duration_seconds: float
    model_backend: str


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

class EditDecision(BaseModel):
    shot_number: str
    start_frame: int
    end_frame: int
    transition_in: TransitionType
    transition_duration_frames: int = 0


class EditDecisionList(BaseModel):
    """The literal, executable form of an EditedSequence -- video_editor.py
    consumes this, never the LLM's prose pacing_notes directly."""

    scene_number: int
    fps: int
    decisions: list[EditDecision]


# ---------------------------------------------------------------------------
# Top-level generation state (the LangGraph state object)
# ---------------------------------------------------------------------------

class GenerationState(BaseModel):
    project_title: str

    character_refs: dict[str, CharacterReference] = Field(default_factory=dict)
    scene_anchors: dict[int, SceneAnchor] = Field(default_factory=dict)

    shot_prompts: list[ShotPrompt] = Field(default_factory=list)
    generated_images: list[GeneratedImage] = Field(default_factory=list)
    visual_critiques: list[VisualCritique] = Field(default_factory=list)
    generated_clips: list[GeneratedClip] = Field(default_factory=list)
    edit_decision_lists: list[EditDecisionList] = Field(default_factory=list)

    shot_revision_counts: dict[str, int] = Field(default_factory=dict)  # key: "scene-shot"
    generation_call_count: int = 0

    final_movie_path: Optional[str] = None
    log: list[str] = Field(default_factory=list)

    # Live ContinuityManager instance (see generation/continuity_manager.py).
    # Typed as Any to avoid a schemas.py -> generation/ circular import;
    # always a ContinuityManager at runtime.
    continuity: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True
