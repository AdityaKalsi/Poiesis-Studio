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
def _remediate_prompt(prompt: str, reason_code: str) -> str:
    """
    Modify the image prompt after Gemini refuses or fails
    to produce an image.

    The goal is to make the request clearer, more neutral,
    fictional, and image-generation friendly.
    """

    remediation = {
        "NO_IMAGE": """
        Rewrite the request into a clearly fictional, non-sensitive
        visual description. Remove unnecessary details that could
        cause the image request to be rejected. Keep the character,
        appearance, clothing, environment, composition, and visual
        intent unchanged where possible. Do not introduce real people
        or sensitive real-world context.
        """,

        "IMAGE_SAFETY": """
        Rewrite this as a clearly fictional character image.
        Remove or neutralize any potentially sensitive, violent,
        sexual, graphic, or otherwise policy-sensitive wording.
        Preserve the intended character appearance and visual style.
        """,

                "IMAGE_PROHIBITED_CONTENT": """
        Rewrite the request as a safe fictional image-generation prompt.
        Remove prohibited or potentially problematic details while
        preserving the core visual concept.
        """,

                "IMAGE_RECITATION": """
        Rewrite the description completely in original wording.
        Do not reproduce recognizable copyrighted text or copied
        descriptions. Preserve only the visual characteristics needed
        to generate the intended fictional image.
        """,

                "IMAGE_OTHER": """
        Simplify and rewrite this as a clean, fictional,
        image-generation prompt. Remove unnecessary ambiguity and
        potentially problematic wording while preserving the intended
        visual result.
        """,

                "STOP_NO_IMAGE": """
        Rewrite this as a concise, explicit visual image-generation
        prompt. Clearly describe the subject, appearance, clothing,
        pose, composition, and style without unnecessary or ambiguous
        language.
        """,
            }

    instruction = remediation.get(
                reason_code,
                """
        Rewrite the prompt into a concise, clear, fictional,
        image-generation-friendly visual description while preserving
        the original intent.
        """,
            )

    return f"""Create the image described below.

        IMPORTANT:
        {instruction}

        Original request:
        {prompt}

        Return ONLY the rewritten image-generation prompt.
        """.strip()

def _generate_gemini(
    prompt: str,
    config: GenerationConfig,
    reference_image_paths: list[str] | None = None,
    max_retries: int = 3,
) -> ImageResult:

    print("\n" + "=" * 80 + "\n")
    print("[Gemini] Generating image")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set."
        )

    client = genai.Client(api_key=api_key)

    # ---------------------------------------------------------
    # Reference images + prompt
    # ---------------------------------------------------------
    contents: list = []

    for ref_path in reference_image_paths or []:
        contents.append(
            types.Part.from_bytes(
                data=Path(ref_path).read_bytes(),
                mime_type="image/png",
            )
        )

    # ---------------------------------------------------------
    # Retry state
    # ---------------------------------------------------------
    current_prompt = prompt
    delay = 2

    BLOCKED_FINISH_REASONS = {
        "NO_IMAGE",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "STOP_NO_IMAGE",
    }

    # ---------------------------------------------------------
    # Gemini retry loop
    # ---------------------------------------------------------
    for attempt in range(1, max_retries + 1):

        print(f"[Gemini] Attempt {attempt}/{max_retries}")

        try:
            response = client.models.generate_content(
                model=config.models.gemini_image_model,
                contents=contents + [current_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                ),
            )

        # =====================================================
        # 1. TRANSIENT SERVER ERROR
        # =====================================================
        except ServerError as e:

            if attempt >= max_retries:
                raise RuntimeError(
                    f"Gemini server error after {max_retries} attempts: {e}"
                ) from e

            print(
                f"[Gemini] Server error. "
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)
            delay *= 2
            continue

        # =====================================================
        # 2. NO CANDIDATES / PROMPT BLOCK
        # =====================================================
        if not response.candidates:

            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None)

            reason_code = (
                getattr(block_reason, "name", block_reason)
                or "UNKNOWN"
            )

            if attempt >= max_retries:
                raise GenerationBlockedError(
                    f"Gemini returned no candidates "
                    f"(block_reason={reason_code}) "
                    f"after {max_retries} attempts.",
                    reason_code=reason_code,
                )

            # Modify prompt before retry
            current_prompt = _remediate_prompt(
                original_pr,
                reason_code,
            )

            print(
                f"[Gemini] Prompt blocked: {reason_code}"
            )
            print(
                f"[Gemini] Modifying prompt and "
                f"retrying in {delay} seconds..."
            )

            time.sleep(delay)
            delay *= 2
            continue

        candidate = response.candidates[0]

        # =====================================================
        # 3. CHECK GEMINI FINISH REASON
        # =====================================================
        finish_reason = getattr(
            candidate.finish_reason,
            "name",
            candidate.finish_reason,
        )

        if finish_reason in BLOCKED_FINISH_REASONS:

            if attempt >= max_retries:
                raise GenerationBlockedError(
                    f"Gemini blocked image generation "
                    f"after {max_retries} attempts "
                    f"(finish_reason={finish_reason}).",
                    reason_code=finish_reason,
                )

            # ---------------------------------------------
            # IMPORTANT:
            # Modify the prompt and try Gemini again
            # ---------------------------------------------
            current_prompt = _remediate_prompt(
                current_prompt,
                finish_reason,
            )

            print(
                f"[Gemini] Image generation blocked: "
                f"{finish_reason}"
            )

            print(
                "[Gemini] Modified prompt:"
            )
            print(current_prompt[:500])

            print(
                f"[Gemini] Retrying in {delay} seconds..."
            )

            time.sleep(delay)
            delay *= 2

            continue

        # =====================================================
        # 4. OTHER UNSUCCESSFUL FINISH REASON
        # =====================================================
        if finish_reason not in ("STOP", "MAX_TOKENS", None):

            if attempt >= max_retries:
                raise GenerationBlockedError(
                    f"Gemini finished with reason={finish_reason!r} "
                    f"after {max_retries} attempts.",
                    reason_code=finish_reason or "UNKNOWN",
                )

            current_prompt = _remediate_prompt(
                current_prompt,
                finish_reason,
            )

            print(
                f"[Gemini] Unexpected finish reason: "
                f"{finish_reason}"
            )

            time.sleep(delay)
            delay *= 2
            continue

        # =====================================================
        # 5. EXTRACT IMAGE
        # =====================================================
        parts = getattr(candidate.content, "parts", None)

        if not parts:

            if attempt >= max_retries:
                raise GenerationBlockedError(
                    "Gemini returned no usable image parts.",
                    reason_code="STOP_NO_IMAGE",
                )

            current_prompt = _remediate_prompt(
                current_prompt,
                "STOP_NO_IMAGE",
            )

            print(
                "[Gemini] No image data. "
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)
            delay *= 2
            continue

        for part in parts:

            if getattr(part, "inline_data", None):

                print(
                    f"[Gemini] Image generated successfully "
                    f"on attempt {attempt}"
                )

                return ImageResult(
                    image_bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type
                    or "image/png",
                )

        # No image bytes found
        if attempt >= max_retries:
            raise GenerationBlockedError(
                "Gemini response contained no image data.",
                reason_code="STOP_NO_IMAGE",
            )

        current_prompt = _remediate_prompt(
            current_prompt,
            "STOP_NO_IMAGE",
        )

        print(
            "[Gemini] Response contained no image. "
            f"Retrying in {delay} seconds..."
        )

        time.sleep(delay)
        delay *= 2

    raise RuntimeError("Gemini image generation failed.")



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
