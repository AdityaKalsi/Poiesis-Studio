"""
Assembles a scene's generated clips into a single video according to the
frame-accurate EditDecisionList from cut_planner.py, then concatenates
all scene videos into the final movie.

Uses ffmpeg via subprocess -- no GPU dependency, so unlike the other
generation/ files this one is fully implemented rather than stubbed.
Audio muxing (dialogue/music/sfx) is a known gap -- see the "audio
pipeline" note in the architecture review; this file currently produces
video-only output and is the natural place to add an audio track once
that pipeline exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from film_generation.schemas import EditDecisionList, GeneratedClip


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")


def assemble_scene(
    edl: EditDecisionList,
    clips_by_shot: dict[str, GeneratedClip],
    output_path: Path,
) -> Path:
    """Trims each clip to its planned duration and concatenates in order.
    Hard cuts use the concat demuxer; any shot with a crossfade transition
    falls back to a per-pair xfade filter chain."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_crossfade = any(d.transition_in == "crossfade" for d in edl.decisions)

    trimmed_paths: list[Path] = []
    for decision in edl.decisions:
        clip = clips_by_shot.get(decision.shot_number)
        if clip is None:
            raise ValueError(f"No generated clip for shot {decision.shot_number}")

        duration_seconds = (decision.end_frame - decision.start_frame) / edl.fps
        trimmed_path = output_path.parent / f"_trim_{decision.shot_number}.mp4"
        _run_ffmpeg(
            [
                "-i", clip.video_path,
                "-t", f"{duration_seconds:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-an",
                str(trimmed_path),
            ]
        )
        trimmed_paths.append(trimmed_path)

    if not has_crossfade:
        concat_list_path = output_path.parent / f"_concat_{edl.scene_number}.txt"
        concat_list_path.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in trimmed_paths)
        )
        _run_ffmpeg(
            [
                "-f", "concat", "-safe", "0", "-i", str(concat_list_path),
                "-c", "copy", str(output_path),
            ]
        )
        return output_path

    # Crossfade path: chain xfade filters pairwise across all trimmed clips.
    inputs: list[str] = []
    for p in trimmed_paths:
        inputs += ["-i", str(p)]

    filter_parts = []
    prev_label = "0:v"
    cumulative_offset = 0.0
    for i in range(1, len(trimmed_paths)):
        decision = edl.decisions[i]
        fade_duration = decision.transition_duration_frames / edl.fps
        clip_duration = (decision.end_frame - decision.start_frame) / edl.fps
        cumulative_offset += clip_duration - fade_duration
        out_label = f"v{i}"
        transition = "fade" if decision.transition_in == "crossfade" else "fadeblack"
        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition={transition}:"
            f"duration={max(fade_duration, 0.01):.3f}:offset={cumulative_offset:.3f}[{out_label}]"
        )
        prev_label = out_label

    filter_complex = ";".join(filter_parts)
    _run_ffmpeg(
        [
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{prev_label}]",
            "-c:v", "libx264", "-preset", "fast",
            str(output_path),
        ]
    )
    return output_path


def concatenate_final_movie(scene_video_paths: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list_path = output_path.parent / "_final_concat.txt"
    concat_list_path.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in scene_video_paths)
    )
    _run_ffmpeg(
        [
            "-f", "concat", "-safe", "0", "-i", str(concat_list_path),
            "-c", "copy", str(output_path),
        ]
    )
    return output_path
