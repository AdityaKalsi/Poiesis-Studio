"""
Organizes generated assets on disk and provides a cache-key scheme so
retries (shot revisions, reruns) don't either waste generation calls or
silently reuse a stale asset.

Cache key is a hash of (prompt + conditioning refs + seed + model
backend) -- NOT scene/shot number alone. Two calls with the same
scene/shot but different prompts (e.g. after a critic-triggered revision)
must produce different cache entries; two calls with identical inputs
should hit the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from film_generation.schemas import ShotPrompt

# asset_manager.py


def compute_cache_key(shot_prompt: ShotPrompt, model_backend: str, seed: int) -> str:
    payload = {
        "positive_prompt": shot_prompt.positive_prompt,
        "negative_prompt": shot_prompt.negative_prompt,
        "character_ref_ids": sorted(shot_prompt.character_ref_ids),
        "scene_anchor_id": shot_prompt.scene_anchor_id,
        "model_backend": model_backend,
        "seed": seed,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


class AssetManager:
    # store the path as path object and create the assets directory if it doesnt exist already
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, scene_number: int, shot_number: str, stage: str, cache_key: str, ext: str) -> Path:
        """stage: 'character_ref' | 'scene_anchor' | 'shot_image' | 'clip'"""
       
        scene_dir = self.output_dir / f"scene_{scene_number:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{stage}_{shot_number}_{cache_key}.{ext}"
        return scene_dir / filename

    def exists(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    def final_movie_path(self, project_title: str) -> Path:
        safe_title = "".join(c if c.isalnum() else "_" for c in project_title)
        return self.output_dir / f"{safe_title}_final.mp4"

    def _cache_meta_path(self, cache_key: str) -> Path:
        meta_dir = self.output_dir / "_cache"
        meta_dir.mkdir(parents=True, exist_ok=True)
        return meta_dir / f"{cache_key}.json"

    def has_cached(self, cache_key: str) -> bool:
        return self._cache_meta_path(cache_key).exists()

    def save_cache(self, cache_key: str, data: dict) -> None:
        """Call this right after a successful generation, alongside whatever
        already writes the binary asset to path_for(...)."""
        self._cache_meta_path(cache_key).write_text(json.dumps(data))

    def load_cached(self, cache_key: str) -> dict:
        return json.loads(self._cache_meta_path(cache_key).read_text())