"""
Generates a consistent identity reference for one character: a single
portrait image, generated once and reused (as a multimodal reference
input) in every subsequent shot that character appears in.

Uses config.consistency.strategy == "reference_image": with a hosted API
and no local GPU, we can't inject IP-Adapter/LoRA embeddings, so the
reference image itself is the payload -- it gets passed alongside the
prompt to Gemini for every shot featuring this character (see
shot_generator.py).
"""

from __future__ import annotations

from film_generation.config import GenerationConfig
from film_generation.generation.asset_manager import AssetManager
from film_generation.models.text2image import generate_image
from film_generation.schemas import CharacterReference


def generate_character_reference(
    character_name: str,
    description: str,
    config: GenerationConfig,
    assets: AssetManager,
) -> CharacterReference:
    
    print(f"Generating character reference for {character_name}...")

    prompt = (
        f"A single clean character reference portrait of {character_name}. "
        f"{description}. Neutral studio background, front-facing, natural "
        f"lighting, high detail, consistent likeness for reuse as a "
        f"reference image across many shots."
    )
    
    result = generate_image(prompt, config)

    # Save image
    character_dir = assets.output_dir / "characters"
    character_dir.mkdir(parents=True, exist_ok=True)

    image_path = character_dir / f"{character_name.replace(' ', '_')}.png"
    result.save(image_path)

    # Create CharacterReference object
    character_reference = CharacterReference(
        character_name=character_name,
        description=description,
        reference_image_path=str(image_path),
        identity_payload_path=str(image_path),
        identity_payload_kind="image",
    )

    # Save JSON
    json_path = character_dir / f"{character_name.replace(' ', '_')}.json"
    json_path.write_text(character_reference.model_dump_json(indent=2))

    print(f"Character reference written to {json_path.resolve()}")

    return character_reference
