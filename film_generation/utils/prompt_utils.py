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

REMEDIATION_TEMPLATES: dict[str, str] = {
    "IMAGE_SAFETY": (
        "The previous attempt was blocked by post-generation safety filters. "
        "Neutralize intense emotional or violent keywords, soften action "
        "descriptions, and avoid explicit, graphic, or gory visual detail."
    ),
    "IMAGE_PROHIBITED_CONTENT": (
        "The previous attempt was blocked for prohibited content. Replace any "
        "real person's name, copyrighted character, logo, or trademarked "
        "visual element with a generic descriptor (e.g. 'a heroic warrior in "
        "dark armor' instead of a named copyrighted character)."
    ),
    "NO_IMAGE": (
        "The previous attempt produced no image payload. Phrase the request "
        "explicitly as 'Generate an image of...' describing one concrete "
        "visual subject."
    ),
    "IMAGE_RECITATION": (
        "The previous attempt too closely resembled existing copyrighted "
        "artwork or stock imagery. Change the art-direction/style keywords "
        "(e.g. photorealistic -> stylized digital painting) and vary the "
        "composition details."
    ),
    "IMAGE_OTHER": (
        "The previous attempt failed to produce an image for an unspecified "
        "reason. Simplify the description into one clear, unambiguous visual "
        "subject, remove any conflicting or overly complex instructions, and "
        "phrase it as a direct 'Generate an image of...' request."
    ),
    "STOP_NO_IMAGE": (
        "The previous attempt returned a text explanation instead of an "
        "image. Rephrase as a direct visual subject description, not a "
        "conversational request or question."
    ),
    "SAFETY": (
        "The previous attempt was blocked before generation began due to "
        "unsafe input text. Remove sensitive, unsafe, or banned terms from "
        "the description."
    ),
    "SPII": (
        "The previous attempt was blocked because the input contained "
        "sensitive personal information. Remove real names or identifying "
        "personal details from the description."
    ),
    "BLOCKLIST": (
        "The previous attempt was blocked for containing blocklisted terms. "
        "Reword the description to avoid banned vocabulary."
    ),
}

_DEFAULT_REMEDIATION = (
    "The previous attempt failed to generate (reason: {reason}). Simplify "
    "and rephrase the description as a direct, safe visual subject."
)


def remediation_feedback_for(reason_code: str) -> str:
    return REMEDIATION_TEMPLATES.get(reason_code, _DEFAULT_REMEDIATION.format(reason=reason_code))


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
