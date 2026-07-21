"""
Agent implementations. Each agent is a thin wrapper around a
structured-output LLM call: system prompt + typed input -> typed Pydantic
output. Keeping agents this simple is deliberate for a v1 prototype --
the interesting engineering is in the graph wiring and the shared state,
not in exotic per-agent logic.
"""

import os
import time
from langchain_ollama import ChatOllama


from film_agents.schemas import (
    ScriptBreakdown,
    SceneBreakdown,
    SceneIntentBrief,
    ShotList,
    EditedSequence,
    CriticReport,
)
from film_agents.prompts import (
    SCRIPT_ANALYST_SYSTEM,
    SCENE_DESIGN_SYSTEM,
    CINEMATOGRAPHY_SYSTEM,
    EDITING_SYSTEM,
    CRITIC_SYSTEM,
)


def _llm(temperature: float = 0.4) -> ChatOllama:
    return ChatOllama(model="qwen2.5:7b", temperature=temperature)


# ---------------------------------------------------------------------------
# Script Analyst Agent
# ---------------------------------------------------------------------------

def script_analyst_agent(script_text: str) -> ScriptBreakdown:
    print("  → [Script Analyst] calling LLM...")
    t0 = time.time()
    structured_llm = _llm(temperature=0.1).with_structured_output(ScriptBreakdown)
    result =  structured_llm.invoke(
        [
            ("system", SCRIPT_ANALYST_SYSTEM),
            ("human", f"Break down this script:\n\n{script_text}"),
        ]
    )
    print(f"  ← [Script Analyst] done in {time.time() - t0:.1f}s")
    print(result.model_dump_json(indent=2))
    print("\n" + "="*80 + "\n")
    return result


# ---------------------------------------------------------------------------
# Scene Design Agent (acting as Director)
# ---------------------------------------------------------------------------

def scene_design_agent(scene: SceneBreakdown) -> SceneIntentBrief:
    print("  → [Scene Design] calling LLM...")
    structured_llm = _llm(temperature=0.5).with_structured_output(SceneIntentBrief)
    prompt = (
        f"Scene {scene.scene_number}: {scene.heading}\n"
        f"Synopsis: {scene.synopsis}\n"
        f"Cast: {', '.join(scene.cast) or 'none listed'}\n"
        f"Props: {', '.join(scene.props) or 'none listed'}\n"
    )
    result = structured_llm.invoke(
        [("system", SCENE_DESIGN_SYSTEM), ("human", prompt)]
    )
    result.scene_number = scene.scene_number
    print(result.model_dump_json(indent=2))
    print("\n" + "="*80 + "\n")
    return result

# ---------------------------------------------------------------------------
# Cinematography Agent
# ---------------------------------------------------------------------------

def cinematography_agent(
    scene: SceneBreakdown,
    brief: SceneIntentBrief,
    prior_feedback: str | None = None,
) -> ShotList:
    print("  → [Cinematography Agent] calling LLM...")
    structured_llm = _llm(temperature=0.5).with_structured_output(ShotList)
    prompt = (
        f"Scene {scene.scene_number}: {scene.heading}\n"
        f"Synopsis: {scene.synopsis}\n\n"
        f"Director's Intent Brief:\n"
        f"- Purpose: {brief.purpose}\n"
        f"- Emotional objective: {brief.emotional_objective}\n"
        f"- Visual objective: {brief.visual_objective}\n"
        f"- Character objectives: {brief.character_objectives}\n"
        f"- Tone: {brief.tone}\n"
    )
    if prior_feedback:
        prompt += f"\nPRIOR CRITIC FEEDBACK YOU MUST ADDRESS:\n{prior_feedback}\n"

    result = structured_llm.invoke(
        [("system", CINEMATOGRAPHY_SYSTEM), ("human", prompt)]
    )
    
    result.scene_number = scene.scene_number
    print(result.model_dump_json(indent=2))
    print("\n" + "="*80 + "\n")
    return result

# ---------------------------------------------------------------------------
# Editing Agent
# ---------------------------------------------------------------------------

def editing_agent(brief: SceneIntentBrief, shot_list: ShotList) -> EditedSequence:
    print("  → [Editing Agent] calling LLM...")
    structured_llm = _llm(temperature=0.4).with_structured_output(EditedSequence)
    shots_desc = "\n".join(
        f"- {s.shot_number} ({s.shot_type}): {s.description} "
        f"[serves: {s.narrative_purpose}]"
        for s in shot_list.shots
    )
    prompt = (
        f"Director's Intent Brief:\n"
        f"- Purpose: {brief.purpose}\n"
        f"- Emotional objective: {brief.emotional_objective}\n"
        f"- Tone: {brief.tone}\n\n"
        f"Shot list:\n{shots_desc}\n"
    )
    result = structured_llm.invoke([("system", EDITING_SYSTEM), ("human", prompt)])
    result.scene_number = shot_list.scene_number
    print(result.model_dump_json(indent=2))
    print("\n" + "="*80 + "\n")
    return result
# ---------------------------------------------------------------------------
# Critic / QC Agent
# ---------------------------------------------------------------------------

def critic_agent(brief: SceneIntentBrief, stage: str, artifact_description: str) -> CriticReport:
    print("  → [Critic Agent] calling LLM...")
    structured_llm = _llm(temperature=0.0).with_structured_output(CriticReport)
    prompt = (
        f"Stage being evaluated: {stage}\n\n"
        f"Director's Intent Brief:\n"
        f"- Purpose: {brief.purpose}\n"
        f"- Emotional objective: {brief.emotional_objective}\n"
        f"- Visual objective: {brief.visual_objective}\n"
        f"- Character objectives: {brief.character_objectives}\n"
        f"- Tone: {brief.tone}\n\n"
        f"Artifact to evaluate:\n{artifact_description}\n"
    )
    result = structured_llm.invoke([("system", CRITIC_SYSTEM), ("human", prompt)])
    result.scene_number = brief.scene_number
    result.stage = stage
    print(result.model_dump_json(indent=2))
    print("\n" + "="*80 + "\n")
    return result
    exit()


