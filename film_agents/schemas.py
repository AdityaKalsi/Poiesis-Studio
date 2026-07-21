"""
Shared data contracts for the multi-agent film pipeline.

Every agent reads from and writes to these typed structures instead of
free-text. This is what lets agents hand off work reliably and lets the
Critic Agent score outputs programmatically.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Script Analyst Agent output
# ---------------------------------------------------------------------------

class SceneBreakdown(BaseModel):
    """One scene's extracted production elements (the automated
    equivalent of a professional script breakdown sheet)."""

    scene_number: int
    heading: str = Field(..., description="e.g. 'INT. KITCHEN - NIGHT'")
    synopsis: str = Field(..., description="1-2 sentence summary of what happens")
    cast: list[str] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    wardrobe_notes: list[str] = Field(default_factory=list)
    vfx_or_sfx: list[str] = Field(default_factory=list)
    stunts: list[str] = Field(default_factory=list)
    special_equipment: list[str] = Field(default_factory=list)
    sound_notes: list[str] = Field(default_factory=list)


class ScriptBreakdown(BaseModel):
    title: str
    scenes: list[SceneBreakdown]


# ---------------------------------------------------------------------------
# 2. Scene Design Agent output
# ---------------------------------------------------------------------------

class SceneIntentBrief(BaseModel):
    """The single most important artifact in the pipeline. Every
    downstream agent (Cinematography, Lighting, Editing, Critic) checks
    its work against this, not against generic 'quality'."""

    scene_number: int
    purpose: str = Field(..., description="Why this scene exists in the story")
    emotional_objective: str = Field(..., description="What the audience should feel by scene end")
    visual_objective: str = Field(..., description="The single image/idea that best communicates the scene's meaning")
    character_objectives: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of character name -> what they want in this scene",
    )
    tone: str = Field(..., description="e.g. 'tense', 'comedic', 'melancholy'")


# ---------------------------------------------------------------------------
# 3. Cinematography Agent output
# ---------------------------------------------------------------------------

ShotType = Literal[
    "wide", "medium", "close_up", "insert", "cutaway",
    "pov", "tracking", "crane", "dolly", "handheld", "steadicam",
]


class Shot(BaseModel):
    shot_number: str = Field(..., description="e.g. '4A'")
    shot_type: ShotType
    lens_mm: Optional[int] = Field(None, description="Approximate focal length")
    description: str = Field(..., description="What is framed and why")
    movement: Optional[str] = Field(None, description="Camera movement, if any")
    narrative_purpose: str = Field(..., description="Which beat/objective from the intent brief this shot serves")


class ShotList(BaseModel):
    scene_number: int
    shots: list[Shot]


# ---------------------------------------------------------------------------
# 4. Editing Agent output
# ---------------------------------------------------------------------------

class EditedSequence(BaseModel):
    scene_number: int
    shot_order: list[str] = Field(..., description="Ordered list of shot_numbers as cut together")
    pacing_notes: str = Field(..., description="How rhythm/duration should modulate through the scene")
    transitions_in_out: str = Field(..., description="How this scene should begin and end relative to neighbors")


# ---------------------------------------------------------------------------
# 5. Critic / QC Agent output
# ---------------------------------------------------------------------------

class CriticReport(BaseModel):
    scene_number: int
    stage: str = Field(..., description="Which pipeline stage this critique is for")
    score: int = Field(..., ge=1, le=10, description="1-10 alignment with the scene intent brief")
    passes: bool
    feedback: str = Field(..., description="Specific, actionable revision notes if it does not pass")


# ---------------------------------------------------------------------------
# 6. Top-level graph state
# ---------------------------------------------------------------------------

class ProjectState(BaseModel):
    """The single shared state object that flows through the LangGraph
    pipeline. Every node reads what it needs and writes its output back
    here -- this is the 'Project State Store' described in the
    architecture doc, simplified to fit in-memory for a prototype."""

    script_text: str

    breakdown: Optional[ScriptBreakdown] = None
    intent_briefs: list[SceneIntentBrief] = Field(default_factory=list)
    shot_lists: list[ShotList] = Field(default_factory=list)
    edited_sequences: list[EditedSequence] = Field(default_factory=list)

    cinematography_critiques: list[CriticReport] = Field(default_factory=list)
    final_critiques: list[CriticReport] = Field(default_factory=list)

    cinematography_revision_count: int = 0
    max_revisions: int = 2

    log: list[str] = Field(default_factory=list)
