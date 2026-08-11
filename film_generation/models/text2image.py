"""
models/text2image.py -- the only file allowed to know which image-gen
SDK we're actually calling. generation/*.py files call generate_image()
below and never touch google.genai or huggingface_hub directly, so
swapping backends never means touching character_reference.py,
scene_anchor.py, or shot_generator.py.

Both backends are fully implemented. Gemini is active by default because
it's the only one of the two that accepts reference images alongside the
prompt (needed for character/scene consistency) -- HuggingFace's
InferenceClient.text_to_image is text-only. To swap: change
config.models.text2image_backend to "huggingface", or just flip which
branch is commented out in generate_image() below if you want it
hardcoded rather than config-driven.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import InferenceClient
import io
from film_generation.config import GenerationConfig
from google.genai import types
import time
from google import genai
from google.genai.errors import ServerError


@dataclass
class ImageResult:
    image_bytes: bytes
    mime_type: str = "image/png"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.image_bytes)
        return path


class GenerationBlockedError(RuntimeError):
    """Raised when the image generator returns a blocked or unusable result."""


# ---------------------------------------------------------------------------
# Gemini backend (active) -- supports text + reference image(s) -> image
# ---------------------------------------------------------------------------

def _generate_gemini(
    prompt: str,
    config: GenerationConfig,
    reference_image_paths: list[str] | None = None,
    max_retries: int = 3,
) -> ImageResult:
    print("\n" + "="*80 + "\n")
    print(f"[Gemini] Generating image")


    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set -- required for the Gemini image backend."
        )

    client = genai.Client(api_key=api_key)

    # Multimodal contents: reference images first (grounds identity/style),
    # then the text prompt describing what to generate.
    contents: list = []
    for ref_path in reference_image_paths or []:
        contents.append(types.Part.from_bytes(
            data=Path(ref_path).read_bytes(),
            mime_type="image/png",
        ))
    contents.append(prompt)

    delay = 2  # starting delay in seconds
    for attempt in range(max_retries):
        try:
               response = client.models.generate_content(
                       model=config.models.gemini_image_model,
                       contents=contents,
                       config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                   )
               return response
        except ServerError as e:
               print(f"Attempt {attempt + 1} hit Google 500 error. Retrying in {delay} seconds...")
               time.sleep(delay)
               delay *= 2  # Exponential backoff
    
    

    # Gemini returns candidates=[] (and .parts=None) on safety blocks or
    # empty generations -- never assume .parts is iterable.
    if not response.candidates:
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", "unknown")
        raise GenerationBlockedError(
            f"Gemini returned no candidates (block_reason={block_reason}) "
            f"for prompt: {prompt[:200]!r}"
        )

    candidate = response.candidates[0]
    
    # Catch NO_IMAGE explicitly to provide a better error message
    if getattr(candidate.finish_reason, "name", candidate.finish_reason) == "NO_IMAGE":
        raise GenerationBlockedError(
            f"Gemini blocked the image generation (likely due to human face/person safety filters). "
            f"Prompt: {prompt[:200]!r}"
        )

    if candidate.finish_reason not in ("STOP", "MAX_TOKENS", None) or not candidate.content.parts:
        raise GenerationBlockedError(
            f"Gemini finished with reason={candidate.finish_reason!r}, no usable parts. "
            f"Prompt: {prompt[:200]!r}"
        )
    
    for part in response.parts:
        if part.inline_data:
            return ImageResult(image_bytes=part.inline_data.data, mime_type=part.inline_data.mime_type)



# ---------------------------------------------------------------------------
# HuggingFace backend (implemented, not active by default) -- text-only,
# no reference-image conditioning. Fine for scene anchors / rough drafts,
# not for character consistency.
# ---------------------------------------------------------------------------

def _generate_huggingface(prompt: str, config: GenerationConfig) -> ImageResult:

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set -- required for the HuggingFace image backend.")

    client = InferenceClient(model=config.models.huggingface_text2image_model, token=token)
    pil_image = client.text_to_image(prompt)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return ImageResult(image_bytes=buffer.getvalue(), mime_type="image/png")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    config: GenerationConfig,
    reference_image_paths: list[str] | None = None,
) -> ImageResult:
    """Single entry point used by character_reference.py, scene_anchor.py,
    and shot_generator.py. Branches on config.models.text2image_backend --
    that's the only place backend selection should happen."""

    if config.models.text2image_backend == "gemini":
        return _generate_gemini(prompt, config, reference_image_paths)

    if config.models.text2image_backend == "huggingface":
        if reference_image_paths:
            raise ValueError(
                "HuggingFace backend is text-only and can't use reference_image_paths "
                "-- switch config.models.text2image_backend to 'gemini' for "
                "character/scene-consistent generation."
            )
        return _generate_huggingface(prompt, config)

    raise ValueError(f"Unknown text2image_backend: {config.models.text2image_backend!r}")
