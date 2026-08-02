"""
Converts EditedSequence (the reasoning pipeline's editing_agent output --
shot_order, pacing_notes, transitions_in_out, all in prose) into an
EditDecisionList: literal, frame-accurate start/end points and transition
types that video_editor.py can actually execute.

video_editor.py must never be handed pacing_notes directly -- "hold wide
shot to let dread build, then accelerate cutting" is not executable.
This file is where that prose gets turned into numbers.

Uses a constrained structured-output LLM call (same pattern as
film_agents/agents.py) rather than hand-written heuristics, because
interpreting "accelerate cutting as the door opens" into per-shot
duration trims is exactly the kind of judgment call an LLM handles
better than a rule table -- but the OUTPUT schema is strictly typed so
video_editor.py never has to parse prose either.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from film_agents.schemas import EditedSequence
from film_generation.config import TransitionType
from film_generation.schemas import EditDecision, EditDecisionList, GeneratedClip

CUT_PLANNER_SYSTEM = """You are an Editor's assistant. You are given a
scene's edit plan in prose (shot order, pacing notes, transition notes)
and the ACTUAL available duration of each shot's generated clip in
seconds. Decide, for each shot in order:

- trim_to_seconds: how much of the available clip duration to use
  (must be <= the clip's actual duration; shorten clips to accelerate
  pacing, use closer to full duration to let a shot breathe)
- transition_in: how this shot should begin relative to the previous one
- transition_duration_frames: 0 for a hard cut, >0 for crossfade/fade

Respect the pacing_notes and transitions_in_out description literally --
if it says pacing accelerates toward the end, later shots should have
smaller trim_to_seconds than earlier ones."""


class ShotCutDecision(BaseModel):
    shot_number: str
    trim_to_seconds: float = Field(..., gt=0)
    transition_in: TransitionType
    transition_duration_frames: int = Field(default=0, ge=0)


class CutPlanLLMOutput(BaseModel):
    decisions: list[ShotCutDecision]


def _llm() -> ChatOllama:
    return ChatOllama(model="qwen2.5:7b", temperature=0.2)


def plan_cuts(
    edited_sequence: EditedSequence,
    clips_by_shot: dict[str, GeneratedClip],
    fps: int = 24,
) -> EditDecisionList:
    print(f"Planning cuts for scene {edited_sequence.scene_number} at {fps} fps...")
    available_durations = "\n".join(
        f"- shot {shot_number}: {clips_by_shot[shot_number].duration_seconds:.1f}s available"
        for shot_number in edited_sequence.shot_order
        if shot_number in clips_by_shot
    )
    prompt = (
        f"Shot order: {edited_sequence.shot_order}\n"
        f"Pacing notes: {edited_sequence.pacing_notes}\n"
        f"Transitions: {edited_sequence.transitions_in_out}\n\n"
        f"Available clip durations:\n{available_durations}\n"
    )
    print(f"Prompt for cut planning:\n{prompt}")
    structured_llm = _llm().with_structured_output(CutPlanLLMOutput)
    result: CutPlanLLMOutput = structured_llm.invoke(
        [("system", CUT_PLANNER_SYSTEM), ("human", prompt)]
    )

    decisions: list[EditDecision] = []
    cursor_frame = 0
    decision_by_shot = {d.shot_number: d for d in result.decisions}
    print(f"Cut planning decisions: {decision_by_shot}")
    for shot_number in edited_sequence.shot_order:
        decision = decision_by_shot.get(shot_number)
        clip = clips_by_shot.get(shot_number)
        if decision is None or clip is None:
            continue  # missing clip/decision -- caller should treat as incomplete plan

        trim_seconds = min(decision.trim_to_seconds, clip.duration_seconds)
        frame_count = max(1, round(trim_seconds * fps))

        decisions.append(
            EditDecision(
                shot_number=shot_number,
                start_frame=cursor_frame,
                end_frame=cursor_frame + frame_count,
                transition_in=decision.transition_in,
                transition_duration_frames=decision.transition_duration_frames,
            )
        )
        cursor_frame += frame_count
    print(f"Final cut plan: {decisions}")
    return EditDecisionList(
        scene_number=edited_sequence.scene_number, fps=fps, decisions=decisions
    )
