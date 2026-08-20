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
import random
from PIL import Image
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

MAX_REFERENCE_IMAGE_BYTES = 1_000_000  # ~1MB


def _load_and_downscale_image(
    path: str, max_bytes: int = MAX_REFERENCE_IMAGE_BYTES
) -> tuple[bytes, str]:
    """
    Large reference-image payloads increase memory pressure on Gemini's
    generation nodes and are a documented contributor to 500 INTERNAL
    errors, especially with multiple reference images in one call.
    """
    raw = Path(path).read_bytes()
    if len(raw) <= max_bytes:
        return raw, "image/png"

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    quality, scale = 85, 1.0

    while True:
        w, h = img.size
        resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()

        if len(data) <= max_bytes or (quality <= 40 and scale <= 0.3):
            return data, "image/jpeg"

        if quality > 40:
            quality -= 15
        else:
            scale *= 0.8
# ---------------------------------------------------------------------------
# Gemini backend (active) -- supports text + reference image(s) -> image
# ---------------------------------------------------------------------------
def rewrite_image_prompt(
    prompt: str,
    reason_code: str,
    config: GenerationConfig,
) -> str:

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    rewrite_instruction = f"""
Rewrite the following image-generation prompt so that it is clearer,
safer, concise, and more likely to successfully generate an image.

Gemini image failure reason: {reason_code}

Preserve:
- character identity
- clothing
- appearance
- composition
- intended visual style

Remove:
- ambiguous wording
- sensitive wording
- unnecessary names/details
- anything that could trigger image-generation safety filters

Return ONLY the rewritten image prompt.

Original prompt:
{prompt}
"""

    response = client.models.generate_content(
        model=config.models.gemini_text_model,
        contents=rewrite_instruction,
    )

    return response.text.strip()


def _safe_rewrite_prompt(
    prompt: str,
    reason_code: str,
    config: GenerationConfig,
) -> str:
    """
    Wraps rewrite_image_prompt() (an extra network call to the text model)
    so that a transient failure there can never take down the whole
    generation run the way it did before.

    If the API-based rewrite itself fails for any reason (ServerError,
    timeout, malformed response, etc.), fall back to the offline,
    template-based _remediate_prompt() below, which needs no network call
    and therefore can't itself introduce a new point of failure.
    """
    try:
        return rewrite_image_prompt(prompt, reason_code, config)
    except Exception as e:
        print(
            f"[Gemini] Prompt rewrite via API failed ({e!r}); "
            f"falling back to local template rewrite."
        )
        return _remediate_prompt(prompt, reason_code)


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

    # NOTE: this must return a prompt suitable to send DIRECTLY to the
    # image model (it's used as `current_prompt` in the generation call).
    # It must NOT contain meta-instructions like "return only the rewritten
    # prompt" -- that text was previously being rendered/interpreted by the
    # image model itself instead of being stripped out, which likely
    # contributed to repeated NO_IMAGE results.
    return f"{instruction.strip()}\n\n{prompt.strip()}"

def _generate_gemini(
    prompt: str,
    config: GenerationConfig,
    reference_image_paths: list[str] | None = None,
    max_retries: int = 5,
) -> ImageResult:

    print("\n" + "=" * 80 + "\n")
    print("[Gemini] Generating image")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set."
        )

    client = genai.Client(api_key=api_key)
    ...
    # ---------------------------------------------------------
    # Fallback model chain: try the configured model first, then
    # fall forward if it keeps 500ing.
    # ---------------------------------------------------------
    fallback_models = [
        config.models.gemini_image_model,
        "gemini-2.5-flash-image",
        "imagen-3.0-generate-002",
    ]
    seen = set()
    fallback_models = [
        m for m in fallback_models if not (m in seen or seen.add(m))
    ]
    model_idx = 0
    consecutive_server_errors = 0

    # ---------------------------------------------------------
    # Reference images + prompt
    # ---------------------------------------------------------
    contents: list = []

    for ref_path in reference_image_paths or []:
        img_bytes, mime = _load_and_downscale_image(ref_path)
        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type=mime)
        )
        contents.append(
            types.Part.from_bytes(
                data=Path(ref_path).read_bytes(),
                mime_type="image/png",
            )
        )

    # ---------------------------------------------------------
    # Retry state
    # ---------------------------------------------------------
    original_prompt = prompt
    current_prompt = prompt
    delay = 2
    IMAGEN_MODELS = {
        "imagen-3.0-generate-002",
        "imagen-3.0-fast-generate-001",
        "imagen-4.0-generate-001",
    }
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
            active_model = fallback_models[model_idx]

            # ---- Imagen branch: different method, no reference images ----
            if active_model in IMAGEN_MODELS:
                if reference_image_paths:
                    print(
                        f"[Gemini] Note: {active_model!r} does not accept "
                        f"reference images -- generating from text prompt "
                        f"only. Character/scene consistency will be reduced "
                        f"for this shot."
                    )

                img_response = client.models.generate_images(
                    model=active_model,
                    prompt=current_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type="image/png",
                    ),
                )

                if not img_response.generated_images:
                    raise GenerationBlockedError(
                        f"{active_model} returned no images.",
                        reason_code="NO_IMAGE",
                    )

                return ImageResult(
                    image_bytes=img_response.generated_images[0].image.image_bytes,
                    mime_type="image/png",
                )

            # ---- Gemini branch (existing behavior) ----
            response = client.models.generate_content(
                model=active_model,
                contents=contents + [current_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    http_options=types.HttpOptions(
                        retry_options=types.HttpRetryOptions(
                            initial_delay=2.0,
                            attempts=3,
                            exp_base=2.0,
                            max_delay=30.0,
                            http_status_codes=[500, 502, 503, 504],
                        ),
                        timeout=120_000,
                    ),
                ),
            )

        # =====================================================
        # 1. TRANSIENT SERVER ERROR
        # =====================================================
        except ServerError as e:
            consecutive_server_errors += 1

            if (
                consecutive_server_errors >= 2
                and model_idx < len(fallback_models) - 1
            ):
                model_idx += 1
                consecutive_server_errors = 0
                delay = 2
                print(
                    f"[Gemini] Repeated server errors on {active_model!r}; "
                    f"falling back to {fallback_models[model_idx]!r}."
                )

            if attempt >= max_retries:
                raise RuntimeError(
                    f"Gemini server error after {max_retries} attempts "
                    f"across models {fallback_models[:model_idx + 1]}: {e}"
                ) from e

            jittered_delay = delay + random.uniform(0, delay * 0.3)
            print(
                f"[Gemini] Server error on {active_model!r}. "
                f"Retrying in {jittered_delay:.1f}s..."
            )
            time.sleep(jittered_delay)
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
            # NOTE: was `finish_reason` here, which is undefined in this
            # branch (only set later, once candidates exist) -> NameError.
            current_prompt = _safe_rewrite_prompt(
                original_prompt,
                reason_code,
                config,
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
            current_prompt = _safe_rewrite_prompt(
                original_prompt,
                finish_reason,
                config,
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

            current_prompt = _safe_rewrite_prompt(
                original_prompt,
                finish_reason,
                config,
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