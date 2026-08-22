"""
models/image2video.py -- wraps fal.ai's hosted Wan 2.5 image-to-video
endpoint. Same isolation principle as models/text2image.py:
generation/image2video.py calls animate() below and never touches
fal_client directly.

Requires FAL_KEY in the environment (https://fal.ai -- pay-as-you-go,
no local GPU needed since fal runs Wan on their infrastructure).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import fal_client
import urllib.request
from film_generation.config import GenerationConfig


@dataclass
class VideoResult:
    video_url: str
    duration_seconds: float
    local_path: Path | None = None


def _ensure_fal_key() -> None:
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set -- required to call fal.ai's Wan 2.5 endpoint.")

# Default camera behavior per shot type, used when Shot.movement hasn't been
# generated yet (or is empty) upstream in the reasoning pipeline.
_DEFAULT_MOVEMENT_BY_SHOT_TYPE = {
    "wide": "slow, steady push-in, subtle parallax between foreground and background",
    "medium": "gentle handheld sway, natural micro-movement, characters continue their action",
    "close_up": "minimal drift, slow subtle zoom in, focus stays locked on the subject",
    "cutaway": "mostly static frame, very slight handheld drift",
}
_FALLBACK_MOVEMENT = "slow, subtle camera drift, minimal motion"


def build_motion_prompt(shot) -> str:
    """Builds a motion-describing prompt for animate() when Shot.movement is
    missing. Combines a shot_type-based default camera behavior with the
    shot's description, since description carries the actual subject/action
    detail that the raw shot_type alone doesn't."""
    movement = getattr(shot, "movement", None)
    if movement and movement.strip():
        camera_behavior = movement.strip()
    else:
        camera_behavior = _DEFAULT_MOVEMENT_BY_SHOT_TYPE.get(
            getattr(shot, "shot_type", None), _FALLBACK_MOVEMENT
        )

    description = (getattr(shot, "description", "") or "").strip()
    if description:
        return f"{description} Camera: {camera_behavior}."
    return camera_behavior

def animate(
    image_path: str,
    motion_prompt: str,
    config: GenerationConfig,
    duration_seconds: float | None = None,
) -> VideoResult:
    """..."""
    print(f"Animating image with motion prompt: {motion_prompt}")
    _ensure_fal_key()

    if not motion_prompt or not motion_prompt.strip():
        raise ValueError(
            f"motion_prompt is empty for {image_path} -- check that Shot.movement "
            "is populated upstream before calling animate()"
        )

    image_url = fal_client.upload_file(str(image_path))

    requested_duration = duration_seconds or config.defaults.default_clip_seconds
    # Wan 2.5 only accepts '5' or '10' seconds -- snap to the nearest allowed value.
    allowed_durations = (5, 10)
    duration = min(allowed_durations, key=lambda d: abs(d - requested_duration))

    result = fal_client.subscribe(
        config.models.image2video_fal_model,
        arguments={
            "image_url": image_url,
            "prompt": motion_prompt,
            "duration": str(duration),   # API wants a string literal, not a float
            "resolution": "720p",
        },
    )

    video_url = result.get("video", {}).get("url") or result.get("video_url")

    if not video_url:
        raise RuntimeError(f"fal.ai returned no video URL: {result!r}")

    return VideoResult(
        video_url=video_url,
        duration_seconds=float(duration),  # keep the actually-used duration, not the requested one
    )


def download(video_result: VideoResult, dest_path: Path) -> Path:
    """fal.ai URLs are temporary -- download to local disk before handing
    off to asset_manager.py, or the file will be gone by the time
    video_editor.py needs it."""


    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(video_result.video_url, dest_path)
    video_result.local_path = dest_path
    return dest_path
