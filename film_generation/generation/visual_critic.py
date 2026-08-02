"""
Scores a generated shot image against the scene's intent brief AND checks
identity drift against the character reference -- the generation-side
equivalent of film_agents.agents.critic_agent, and the node that makes
pipeline.py's retry loop meaningful.

Both checks happen in a single Gemini multimodal call: the generated
image, the character reference image(s) (if any appear in the shot), and
a structured-output request asking for a brief-alignment score AND an
identity-match confidence. This replaces the originally-planned separate
embeddings-model identity check (see config.py's ConsistencyConfig note)
-- with a hosted VLM already in the loop for the brief judgment, asking
it to also compare faces avoids standing up a second model just for
identity.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from film_agents.schemas import SceneIntentBrief
from film_generation.config import GenerationConfig
from film_generation.schemas import CharacterReference, GeneratedImage, VisualCritique

CRITIC_INSTRUCTIONS = """You are an exacting but fair visual QC critic for
a film production pipeline. You will see:
  1. A GENERATED shot image.
  2. Zero or more CHARACTER REFERENCE images (if provided) -- these show
     what each character is supposed to look like.

Evaluate the generated image against the director's intent brief given
below, and, if reference images were provided, whether the character(s)
in the generated shot match those references.

Score 1-10 on how well the image serves the brief's purpose, emotional
objective, and visual objective. passes_brief = true only if score >= 7.

identity_match_confidence: 0.0-1.0, how confident you are the character(s)
in the generated image are the same individual(s) shown in the reference
image(s). If no reference images were provided, return 1.0 (nothing to
check against).

feedback must be specific and actionable -- what to change and why."""


class _CritiqueOutput(BaseModel):
    score: int = Field(..., ge=1, le=10)
    passes_brief: bool
    identity_match_confidence: float = Field(..., ge=0.0, le=1.0)
    feedback: str


def critique_shot_image(
    image: GeneratedImage,
    brief: SceneIntentBrief,
    character_refs: dict[str, CharacterReference],
    config: GenerationConfig,
) -> VisualCritique:
    from google import genai
    from google.genai import types
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set -- required for the visual critic.")

    client = genai.Client(api_key=api_key)

    relevant_refs = [
        character_refs[name] for name in image.prompt_used.character_ref_ids if name in character_refs
    ]
    print(f"Critiquing shot image for scene {image.scene_number}, shot {image.shot_number} with {relevant_refs} character references...")

    brief_text = (
        f"Director's Intent Brief:\n"
        f"- Purpose: {brief.purpose}\n"
        f"- Emotional objective: {brief.emotional_objective}\n"
        f"- Visual objective: {brief.visual_objective}\n"
        f"- Tone: {brief.tone}\n\n"
        f"Reference images provided: {len(relevant_refs)}\n"
    )

    contents: list = [types.Part.from_bytes(data=Path(image.image_path).read_bytes(), mime_type="image/png")]
    for ref in relevant_refs:
        contents.append(types.Part.from_bytes(data=Path(ref.reference_image_path).read_bytes(), mime_type="image/png"))
    contents.append(brief_text)
    response = client.models.generate_content(
        model=config.models.vlm_backend,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=CRITIC_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=_CritiqueOutput,
        ),
    )

    result: _CritiqueOutput = response.parsed
    print(f"Critique result for scene {image.scene_number}, shot {image.shot_number}: {result}")
    passes = result.passes_brief and (
        not relevant_refs or result.identity_match_confidence >= config.consistency.identity_similarity_threshold
    )
    return VisualCritique(
        scene_number=image.scene_number,
        shot_number=image.shot_number,
        score=result.score,
        identity_similarity=result.identity_match_confidence if relevant_refs else None,
        passes=passes,
        feedback=result.feedback,
    )
