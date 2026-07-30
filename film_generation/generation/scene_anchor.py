"""
Generates one environment/style reference image per scene, used as a
composition and lighting reference passed alongside the prompt for every
shot in that scene -- not a pixel-exact layout, since framing varies
shot to shot within a scene.
"""

from __future__ import annotations

from film_agents.schemas import SceneBreakdown
from film_generation.config import GenerationConfig
from film_generation.generation.asset_manager import AssetManager
from film_generation.models import text2image
from film_generation.schemas import SceneAnchor


def generate_scene_anchor(
    scene: SceneBreakdown,
    visual_objective: str,
    config: GenerationConfig,
    assets: AssetManager,
) -> SceneAnchor:
    location_desc = ", ".join(scene.locations) or scene.heading
    prompt = (
        f"An establishing environment reference image for: {scene.heading}. "
        f"Location: {location_desc}. Visual objective: {visual_objective}. "
        f"No characters in frame -- this is a pure environment/lighting/"
        f"palette reference for maintaining consistency across multiple "
        f"camera angles of this same space."
    )

    result = text2image.generate_image(prompt, config)

    image_path = assets.output_dir / f"scene_{scene.scene_number:03d}" / "anchor.png"
    result.save(image_path)

    return SceneAnchor(
        scene_number=scene.scene_number,
        environment_image_path=str(image_path),
        layout_description=location_desc,
        lighting_description=visual_objective,
    )
