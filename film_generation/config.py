"""
Central configuration for the generation pipeline.

Deliberately NOT just file paths -- this is where the load-bearing
architectural decisions live (which consistency strategy, which models,
how much retry/spend is allowed) so they're visible and swappable in one
place instead of buried inside individual generation/*.py files.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


ConsistencyStrategy = Literal["reference_image", "ip_adapter", "instantid", "lora", "text_only"]
TransitionType = Literal["cut", "crossfade", "match_cut", "fade_to_black"]


class ModelConfig(BaseModel):
    """Which backend implements each capability. Swappable without
    touching pipeline logic -- generation/*.py files call models/*.py
    wrappers, never a specific model SDK directly.

    Current stack (no local GPU available):
      - image: Gemini (gemini-2.5-flash-image, direct API) is primary --
        it accepts multiple images alongside the text prompt, so character/
        scene reference images can be passed in for consistency, not just
        described in words. HuggingFace Inference API is implemented as an
        alternate in models/text2image.py but not active by default: it's
        text-only, no reference-image conditioning.
      - video: Wan 2.5 via fal.ai. Wan has no simple first-party API for
        external developers (Alibaba's own access is mostly China-region
        cloud accounts), and self-hosting needs a GPU we don't have -- so
        fal.ai (which hosts Wan on its own infra behind one unified API) is
        the practical route, not a stylistic pick over calling Alibaba
        directly.
      - vlm: Gemini (same SDK/key as image gen) -- visual_critic.py uses it
        for both brief-alignment scoring and identity-drift checking, since
        Gemini can directly compare the generated shot against the
        character reference image in one multimodal call. No separate
        embeddings model needed for v1.
    """

    text2image_backend: Literal["gemini", "huggingface"] = "gemini"
    huggingface_text2image_model: str = "black-forest-labs/FLUX.1-dev"
    gemini_image_model: str = "gemini-2.5-flash-image"

    image2video_fal_model: str = "fal-ai/wan-25-preview/image-to-video"

    vlm_backend: str = "gemini-2.5-flash"


class ConsistencyConfig(BaseModel):
    strategy: ConsistencyStrategy = "reference_image"
    # "reference_image": pass the character reference image itself as
    # multimodal input to Gemini alongside the shot prompt. The practical
    # default for a no-GPU, hosted-API setup -- ip_adapter/instantid/lora
    # all require local model weights we can't run here.
    identity_similarity_threshold: float = Field(
        default=0.55,
        description="Min identity-match confidence (0-1, judged by the VLM) for visual_critic to pass a shot",
    )


class RetryConfig(BaseModel):
    max_shot_revisions: int = Field(
        default=2,
        description="Mirrors ProjectState.max_revisions in the reasoning pipeline -- same retry-budget philosophy",
    )


class BudgetConfig(BaseModel):
    """Generation is the expensive stage. Nothing in this pipeline should
    be able to spend without a ceiling."""

    max_generation_calls_per_run: int = 500
    max_estimated_cost_usd: float | None = None  # set when using paid APIs


class ApprovalGates(BaseModel):
    """Human checkpoints. A bad character reference silently propagates
    into every downstream shot -- these gates let a human catch that
    before expensive video generation runs. Default False for an
    unattended v1 run; flip on for production use."""

    approve_character_refs: bool = False
    approve_scene_anchors: bool = False
    approve_shots: bool = False


class GenerationDefaults(BaseModel):
    resolution: tuple[int, int] = (1024, 576)
    fps: int = 24
    default_clip_seconds: float = 3.0
    seed: int | None = None  # None = random per call, recorded for reproducibility


class GenerationConfig(BaseModel):
    models: ModelConfig = Field(default_factory=ModelConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    approvals: ApprovalGates = Field(default_factory=ApprovalGates)
    defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    output_dir: str = "film_generation/outputs"


def load_config() -> GenerationConfig:
    """Single entry point for obtaining config. Keep this the only place
    that knows about env vars / config files, once those exist."""
    return GenerationConfig()
