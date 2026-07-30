"""
Reusable prompt-building vocabulary. Kept separate from prompt_generator.py
so the phrase bank can grow (and be tuned per T2I backend) without
touching orchestration logic.
"""

from film_agents.schemas import ShotType

SHOT_TYPE_LANGUAGE: dict[ShotType, str] = {
    "wide": "wide shot, full scene geography visible",
    "medium": "medium shot, waist-up framing",
    "close_up": "close-up, tight emotional framing",
    "insert": "insert shot, macro detail framing",
    "cutaway": "cutaway shot, reaction framing",
    "pov": "point-of-view shot, subjective camera",
    "tracking": "tracking shot, sustained lateral camera movement",
    "crane": "crane shot, elevated sweeping camera movement",
    "dolly": "dolly shot, smooth forward/backward camera movement",
    "handheld": "handheld camera, slight naturalistic shake",
    "steadicam": "steadicam shot, smooth sustained follow movement",
}

TONE_LIGHTING_HINTS: dict[str, str] = {
    "tense": "low-key lighting, hard shadows, desaturated palette",
    "wistful": "soft diffused light, warm golden tones",
    "comedic": "bright even lighting, high-key, saturated colors",
    "melancholy": "cool blue-gray tones, overcast soft light",
    "romantic": "warm soft backlight, gentle bokeh",
    "horror": "harsh underlighting, deep shadow, cold color cast",
}

DEFAULT_NEGATIVE_PROMPT = (
    "extra limbs, distorted hands, warped face, text artifacts, "
    "watermark, low resolution, inconsistent lighting"
)


def lighting_hint_for_tone(tone: str) -> str:
    tone_key = tone.strip().lower()
    for key, hint in TONE_LIGHTING_HINTS.items():
        if key in tone_key:
            return hint
    return "naturalistic motivated lighting"


def lens_hint(lens_mm: int | None) -> str:
    if lens_mm is None:
        return ""
    if lens_mm <= 24:
        return f"{lens_mm}mm wide-angle lens"
    if lens_mm >= 85:
        return f"{lens_mm}mm telephoto lens, compressed background"
    return f"{lens_mm}mm lens"
