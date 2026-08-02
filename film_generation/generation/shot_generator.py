"""
Generates the actual shot image: the enriched prompt from
prompt_generator.py, plus reference images for every character in the
shot and the scene anchor, passed together to Gemini's multimodal input.

Note on multi-character shots: Gemini accepts multiple reference images
in one call, but we haven't validated how well identity holds for more
than ~2-3 simultaneous character references in testing -- if a shot with
several characters starts producing weak likeness, that's the first
place to look (may need a per-character inpainting pass instead of one
combined call).
"""

from __future__ import annotations

from film_generation.config import GenerationConfig
from film_generation.generation.asset_manager import AssetManager, compute_cache_key
from film_generation.models import text2image
from film_generation.schemas import CharacterReference, GeneratedImage, SceneAnchor, ShotPrompt


def generate_shot_image(
    shot_prompt: ShotPrompt,
    character_refs: dict[str, CharacterReference],
    scene_anchor: SceneAnchor | None,
    config: GenerationConfig,
    assets: AssetManager,
    seed: int,
) -> GeneratedImage:
    print(
        f"Generating Shot Image for scene {shot_prompt.scene_number}, "
        f"shot {shot_prompt.shot_number}..."
    )

    reference_image_paths: list[str] = []
    for name in shot_prompt.character_ref_ids:
        ref = character_refs.get(name)
        if ref is not None:
            reference_image_paths.append(ref.reference_image_path)

    if scene_anchor is not None:
        reference_image_paths.append(scene_anchor.environment_image_path)

    # HuggingFace backend can't take reference images.
    use_refs = (
        reference_image_paths
        if config.models.text2image_backend == "gemini"
        else None
    )

    result = text2image.generate_image(
        shot_prompt.positive_prompt,
        config,
        reference_image_paths=use_refs,
    )

    cache_key = compute_cache_key(
        shot_prompt,
        config.models.text2image_backend,
        seed,
    )

    image_path = assets.path_for(
        shot_prompt.scene_number,
        shot_prompt.shot_number,
        "shot_image",
        cache_key,
        "png",
    )
    result.save(image_path)

    generated_image = GeneratedImage(
        scene_number=shot_prompt.scene_number,
        shot_number=shot_prompt.shot_number,
        image_path=str(image_path),
        prompt_used=shot_prompt,
        model_backend=config.models.text2image_backend,
        seed=seed,
        cache_key=cache_key,
    )

    # Save metadata
    json_path = image_path.with_suffix(".json")
    json_path.write_text(generated_image.model_dump_json(indent=2))

    print(f"Shot image metadata written to {json_path.resolve()}")

    return generated_image
