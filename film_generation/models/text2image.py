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

    def __init__(self, message: str, reason_code: str = "UNKNOWN"):
        super().__init__(message)
        self.reason_code = reason_code  # e.g. IMAGE_SAFETY, NO_IMAGE, STOP_NO_IMAGE, SAFETY...

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
    response = None

    # Retry loop for transient server errors (500s)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                       model=config.models.gemini_image_model,
                       contents=contents,
                       config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                   )
            break
        except ServerError as e:
            print(f"Attempt {attempt + 1} hit Google 500 error. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    if response is None:
        raise RuntimeError("Gemini returned no response.")

    
    # prompt-side block (no candidates at all)
    if not response.candidates:
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        reason_code = getattr(block_reason, "name", block_reason) or "UNKNOWN"
        raise GenerationBlockedError(
            f"Gemini returned no candidates (block_reason={reason_code}) "
            f"for prompt: {prompt[:200]!r}",
            reason_code=reason_code,  # SAFETY / SPII / BLOCKLIST / OTHER
        )

    candidate = response.candidates[0]

    # ---------------------------------------------------------
    # Check finish reason
    # ---------------------------------------------------------

    finish_reason = getattr(candidate.finish_reason, "name", candidate.finish_reason)

    # Explicit Gemini image-blocking finish reasons -- each gets its own
    # remediation template in prompt_utils.REMEDIATION_TEMPLATES.
    # IMAGE_OTHER is the most common one seen in practice: a catch-all for
    # image generation failing for a reason Gemini doesn't further
    # categorize (as opposed to a specific safety/policy/recitation block).
    BLOCKED_FINISH_REASONS = {
        "NO_IMAGE",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
    }
    if finish_reason in BLOCKED_FINISH_REASONS:
        raise GenerationBlockedError(
            f"Gemini blocked image generation (finish_reason={finish_reason}). "
            f"Prompt: {prompt[:200]!r}",
            reason_code=finish_reason,
        )

    if finish_reason not in ("STOP", "MAX_TOKENS", None):
        raise GenerationBlockedError(
            f"Gemini finished with reason={finish_reason!r}. Prompt: {prompt[:200]!r}",
            reason_code=finish_reason or "UNKNOWN",
        )

    parts = getattr(candidate.content, "parts", None)
    if not parts:
        raise GenerationBlockedError(
            f"Gemini returned no usable parts. Finish reason: {finish_reason!r}. "
            f"Prompt: {prompt[:200]!r}",
            reason_code="STOP_NO_IMAGE",   # STOP + text refusal/empty, no image bytes
        )

    for part in parts:
            if getattr(part, "inline_data", None):
                return ImageResult(
                    image_bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type or "image/png",
                )

    raise GenerationBlockedError(
        f"Gemini response contained no image data. Prompt: {prompt[:200]!r}",
        reason_code="STOP_NO_IMAGE",
    )



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
