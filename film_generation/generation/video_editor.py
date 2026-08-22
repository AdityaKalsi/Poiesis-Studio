"""
Assembles a scene's generated clips into a single video according to the
frame-accurate EditDecisionList from cut_planner.py, then concatenates
all scene videos into the final movie.

Uses ffmpeg via subprocess.
Audio muxing (dialogue/music/sfx) is currently not implemented.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg

from film_generation.schemas import EditDecisionList, GeneratedClip


FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        [FFMPEG_BIN, "-y", *args],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed:\n{result.stderr[-4000:]}"
        )


def assemble_scene(
    edl: EditDecisionList,
    clips_by_shot: dict[str, GeneratedClip],
    output_path: Path,
) -> Path:
    """
    Trims each clip to its planned duration and concatenates them in order.

    Hard cuts use the concat demuxer.
    Crossfades use an xfade filter chain.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_crossfade = any(
        d.transition_in == "crossfade"
        for d in edl.decisions
    )

    trimmed_paths: list[Path] = []

    # ---------------------------------------------------------
    # Trim every generated clip according to the EDL
    # ---------------------------------------------------------

    for decision in edl.decisions:
        clip = clips_by_shot.get(decision.shot_number)

        if clip is None:
            raise ValueError(
                f"No generated clip for shot {decision.shot_number}"
            )

        duration_seconds = (
            decision.end_frame - decision.start_frame
        ) / edl.fps

        trimmed_path = (
            output_path.parent
            / f"_trim_{decision.shot_number}.mp4"
        )

        _run_ffmpeg(
            [
                "-i",
                str(clip.video_path),
                "-t",
                f"{duration_seconds:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-an",
                str(trimmed_path),
            ]
        )

        trimmed_paths.append(trimmed_path)

    if not edl.decisions: 
        raise RuntimeError(
            f"EDL for scene {edl.scene_number} contains no edit decisions"
        )

    if not trimmed_paths:
        raise RuntimeError(
            f"No clips were produced while assembling scene {edl.scene_number}"
        )

    # ---------------------------------------------------------
    # HARD CUT PATH
    # ---------------------------------------------------------

    if not has_crossfade:
        concat_list_path = (
            output_path.parent
            / f"_concat_{edl.scene_number}.txt"
        )

        concat_list_path.write_text(
            "\n".join(
                f"file '{p.resolve().as_posix()}'"
                for p in trimmed_paths
            ),
            encoding="utf-8",
        )

        _run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(output_path),
            ]
        )

        return output_path

    # ---------------------------------------------------------
    # CROSSFADE PATH
    # ---------------------------------------------------------

    inputs: list[str] = []

    for path in trimmed_paths:
        inputs += [
            "-i",
            str(path),
        ]

    filter_parts: list[str] = []

    prev_label = "0:v"

    first_decision = edl.decisions[0]

    cumulative_duration = (
        first_decision.end_frame
        - first_decision.start_frame
    ) / edl.fps

    for i in range(1, len(trimmed_paths)):
        decision = edl.decisions[i]

        fade_duration = (
            decision.transition_duration_frames
            / edl.fps
        )

        current_duration = (
            decision.end_frame
            - decision.start_frame
        ) / edl.fps

        fade_duration = max(
            fade_duration,
            0.01,
        )

        offset = (
            cumulative_duration
            - fade_duration
        )

        out_label = f"v{i}"

        transition = (
            "fade"
            if decision.transition_in == "crossfade"
            else "fadeblack"
        )

        filter_parts.append(
            f"[{prev_label}][{i}:v]"
            f"xfade="
            f"transition={transition}:"
            f"duration={fade_duration:.3f}:"
            f"offset={offset:.3f}"
            f"[{out_label}]"
        )

        prev_label = out_label

        cumulative_duration += (
            current_duration
            - fade_duration
        )

    filter_complex = ";".join(filter_parts)

    _run_ffmpeg(
        [
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{prev_label}]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-an",
            str(output_path),
        ]
    )

    return output_path


def concatenate_final_movie(
    scene_video_paths: list[Path],
    output_path: Path,
) -> Path:

    if not scene_video_paths:
        raise RuntimeError(
            "No scene videos were provided for final concatenation."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    concat_list_path = (
        output_path.parent
        / "_final_concat.txt"
    )

    concat_list_path.write_text(
        "\n".join(
            f"file '{p.resolve().as_posix()}'"
            for p in scene_video_paths
        ),
        encoding="utf-8",
    )

    _run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_path),
        ]
    )

    return output_path