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

from film_generation.config import GenerationConfig


@dataclass
class VideoResult:
    video_url: str
    duration_seconds: float
    local_path: Path | None = None


def _ensure_fal_key() -> None:
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set -- required to call fal.ai's Wan 2.5 endpoint.")


def animate(
    image_path: str,
    motion_prompt: str,
    config: GenerationConfig,
    duration_seconds: float | None = None,
) -> VideoResult:
    """Uploads a local shot image to fal's storage, then calls Wan 2.5
    image-to-video with a motion-describing prompt (built from Shot.movement
    upstream in generation/image2video.py)."""

    import fal_client

    _ensure_fal_key()

    image_url = fal_client.upload_file(image_path)

    result = fal_client.subscribe(
        config.models.image2video_fal_model,
        arguments={
            "image_url": image_url,
            "prompt": motion_prompt,
        },
    )

    # fal's response shape is {"video": {"url": "...", ...}, ...} for
    # Wan endpoints -- verify against the live API response for the exact
    # model version in use, this covers the common case.
    video_url = result.get("video", {}).get("url") or result.get("video_url")
    if not video_url:
        raise RuntimeError(f"fal.ai returned no video URL: {result!r}")

    return VideoResult(
        video_url=video_url,
        duration_seconds=duration_seconds or config.defaults.default_clip_seconds,
    )


def download(video_result: VideoResult, dest_path: Path) -> Path:
    """fal.ai URLs are temporary -- download to local disk before handing
    off to asset_manager.py, or the file will be gone by the time
    video_editor.py needs it."""

    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(video_result.video_url, dest_path)
    video_result.local_path = dest_path
    return dest_path
