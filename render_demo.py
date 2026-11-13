#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alambda Systems
"""Render measured basketball tracks as a mobile-friendly video and court map.

The input JSON contains observations, not generated player motion. ``time`` is
seconds into the source video and is authoritative; ``frame`` is used as a
fallback source-frame index. Coordinates are source-video pixels and court
metres (28.65 by 15.24). Names/numbers in this renderer are tracking IDs only.
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1080, 1350
VIDEO_TOP, VIDEO_HEIGHT = 76, 608
COURT_LENGTH, COURT_WIDTH = 28.65, 15.24
COURT_X, COURT_Y, COURT_PIXELS = 68, 751, 944
COURT_SCALE = COURT_PIXELS / COURT_LENGTH
COURT_HEIGHT = round(COURT_WIDTH * COURT_SCALE)

# RGB palette. Conversion is explicit for OpenCV drawing operations.
BG = (8, 15, 27)
PANEL = (12, 23, 38)
PAINT = (17, 38, 53)
LINE = (108, 132, 155)
WHITE = (237, 245, 250)
MUTED = (145, 165, 186)
TEAMS = ((83, 221, 198), (189, 157, 255))


def bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(reversed(rgb))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    # Prefer the font used for the published render, then try platform fonts.
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/{filename}",
        filename,
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


FONTS = {n: font(n) for n in (14, 15, 16, 17, 18, 20, 21, 23, 28)}
FONTS_BOLD = {n: font(n, True) for n in (15, 16, 17, 18, 20, 21, 23, 28)}


def court_point(x: float, y: float) -> tuple[int, int]:
    return (
        round(COURT_X + x * COURT_SCALE),
        round(COURT_Y + y * COURT_SCALE),
    )


def polyline(
    image: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int] = LINE,
    thickness: int = 2,
    closed: bool = False,
) -> None:
    pts = np.array([court_point(x, y) for x, y in points], dtype=np.int32)
    cv2.polylines(image, [pts], closed, bgr(color), thickness, cv2.LINE_AA)


def arc_points(
    cx: float, cy: float, radius: float, start: float, end: float, n: int = 96
) -> list[tuple[float, float]]:
    return [
        (cx + radius * math.cos(t), cy + radius * math.sin(t))
        for t in np.linspace(math.radians(start), math.radians(end), n)
    ]


def draw_court(canvas: np.ndarray) -> None:
    """NBA-sized court, including 16-foot paint and corner three-point lines."""
    L, W, mid_y = COURT_LENGTH, COURT_WIDTH, COURT_WIDTH / 2
    cv2.rectangle(
        canvas,
        court_point(0, 0),
        court_point(L, W),
        bgr(PANEL),
        -1,
    )
    for left in (True, False):
        baseline, free_throw = (0.0, 5.7912) if left else (L, L - 5.7912)
        direction = 1 if left else -1
        basket_x = baseline + direction * 1.6002
        board_x = baseline + direction * 1.2192
        half_key = 2.4384
        cv2.rectangle(
            canvas,
            court_point(min(baseline, free_throw), mid_y - half_key),
            court_point(max(baseline, free_throw), mid_y + half_key),
            bgr(PAINT),
            -1,
        )
        polyline(
            canvas,
            [
                (baseline, mid_y - half_key),
                (free_throw, mid_y - half_key),
                (free_throw, mid_y + half_key),
                (baseline, mid_y + half_key),
            ],
        )
        front_start, front_end = (-90, 90) if left else (90, 270)
        polyline(canvas, arc_points(free_throw, mid_y, 1.8288, front_start, front_end))
        back_start = front_end
        for angle in np.arange(back_start, back_start + 180, 24):
            polyline(canvas, arc_points(free_throw, mid_y, 1.8288, angle, angle + 12, 8), thickness=1)
        # Three-point arc meets the straight corner portions 3 feet from each sideline.
        three_radius = 7.239
        corner_y = 0.9144
        dy = mid_y - corner_y
        join_dx = math.sqrt(three_radius**2 - dy**2)
        join_x = basket_x + direction * join_dx
        alpha = math.degrees(math.asin(dy / three_radius))
        polyline(canvas, [(baseline, corner_y), (join_x, corner_y)])
        polyline(canvas, [(baseline, W - corner_y), (join_x, W - corner_y)])
        start, end = (-alpha, alpha) if left else (180 - alpha, 180 + alpha)
        polyline(canvas, arc_points(basket_x, mid_y, three_radius, start, end))
        # Restricted area and backboard.
        start, end = (-90, 90) if left else (90, 270)
        polyline(canvas, arc_points(basket_x, mid_y, 1.2192, start, end), thickness=1)
        polyline(canvas, [(board_x, mid_y - 0.9144), (board_x, mid_y + 0.9144)], WHITE, 2)
        cv2.circle(canvas, court_point(basket_x, mid_y), round(0.2286 * COURT_SCALE), bgr((232, 180, 124)), 2, cv2.LINE_AA)
    polyline(canvas, [(0, 0), (L, 0), (L, W), (0, W)], closed=True)
    polyline(canvas, [(L / 2, 0), (L / 2, W)])
    cv2.circle(canvas, court_point(L / 2, mid_y), round(1.8288 * COURT_SCALE), bgr(LINE), 2, cv2.LINE_AA)
    cv2.circle(canvas, court_point(L / 2, mid_y), round(0.6096 * COURT_SCALE), bgr(LINE), 1, cv2.LINE_AA)


def base_canvas(meta: dict[str, Any]) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), bgr(BG), np.uint8)
    cv2.rectangle(canvas, (0, VIDEO_TOP), (WIDTH, VIDEO_TOP + VIDEO_HEIGHT), (0, 0, 0), -1)
    draw_court(canvas)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(image)
    d.text((28, 10), "ALAMBDA SYSTEMS", font=FONTS_BOLD[15], fill=TEAMS[0])
    d.text((26, 31), "Basketball vision", font=FONTS_BOLD[28], fill=WHITE)
    d.rounded_rectangle((866, 20, 1052, 57), radius=9, fill=(24, 38, 54), outline=(52, 71, 91))
    d.text((890, 27), "CV2 DEMO", font=FONTS_BOLD[18], fill=WHITE)
    d.text((35, 703), "COURT POSITION", font=FONTS_BOLD[21], fill=WHITE)
    d.text((36, 730), "Movement trails · 2 seconds", font=FONTS[16], fill=MUTED)
    d.ellipse((698, 712, 711, 725), fill=TEAMS[0])
    d.text((723, 706), "TEAM A", font=FONTS_BOLD[18], fill=TEAMS[0])
    d.ellipse((872, 712, 885, 725), fill=TEAMS[1])
    d.text((897, 706), "TEAM B", font=FONTS_BOLD[18], fill=TEAMS[1])
    d.line((34, 1277, 1046, 1277), fill=(34, 51, 71), width=1)
    source = str(meta.get("source_label") or "Source: Roboflow tutorial sample")
    # Keep attribution within the frame without wrapping into the disclaimer.
    while d.textbbox((0, 0), source, font=FONTS[15])[2] > 1008 and len(source) > 20:
        source = source[:-2]
    d.text((35, 1290), source, font=FONTS[15], fill=MUTED)
    d.text((35, 1315), "Assisted tracking IDs • Court positions are estimates", font=FONTS[17], fill=WHITE)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def label_for(player: dict[str, Any]) -> str:
    return f"{'A' if int(player.get('team', 0)) == 0 else 'B'}{int(player['id'])}"


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] + 3 <= b[0] or b[2] + 3 <= a[0] or a[3] + 2 <= b[1] or b[3] + 2 <= a[1])


def draw_video_players(
    canvas: np.ndarray,
    players: list[dict[str, Any]],
    source_width: int,
    source_height: int,
    frame_rect: tuple[int, int, int, int],
) -> list[tuple[str, tuple[int, int, int, int], tuple[int, int, int]]]:
    """Draw boxes and return label placements for a single Pillow text pass."""
    ox, oy, fw, fh = frame_rect
    sx, sy = fw / source_width, fh / source_height
    labels = []
    used: list[tuple[int, int, int, int]] = []
    for p in sorted(players, key=lambda v: float(v["bbox"][1])):
        team = int(p.get("team", 0)) % 2
        color = TEAMS[team]
        box = p["bbox"]
        x1 = max(ox, min(ox + fw - 1, round(ox + float(box[0]) * sx)))
        y1 = max(oy, min(oy + fh - 1, round(oy + float(box[1]) * sy)))
        x2 = max(ox, min(ox + fw - 1, round(ox + float(box[2]) * sx)))
        y2 = max(oy, min(oy + fh - 1, round(oy + float(box[3]) * sy)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (5, 9, 15), 4, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr(color), 2, cv2.LINE_AA)
        foot = ((x1 + x2) // 2, y2)
        cv2.circle(canvas, foot, 4, bgr(color), -1, cv2.LINE_AA)
        label = label_for(p)
        tw = round(FONTS_BOLD[18].getlength(label)) + 14
        th = 28
        candidate_rects = []
        for xx, yy in [(x1, y1 - th - 3), (x2 - tw, y1 - th - 3), (x1, y2 + 3), (x2 - tw, y2 + 3)]:
            xx = max(ox + 2, min(ox + fw - tw - 2, xx))
            yy = max(oy + 2, min(oy + fh - th - 2, yy))
            candidate_rects.append((xx, yy, xx + tw, yy + th))
        rect = next((r for r in candidate_rects if not any(_overlaps(r, old) for old in used)), candidate_rects[0])
        used.append(rect)
        labels.append((label, rect, color))
    return labels


def render(
    video: str | Path,
    tracks: str | Path | dict[str, Any],
    output: str | Path,
    *,
    preview: str | Path | None = None,
    fps: float | None = None,
    max_seconds: float | None = None,
    keep_audio: bool = False,
) -> dict[str, Any]:
    """Render observations and encode an H.264, yuv420p, fast-start MP4."""
    video, output = Path(video), Path(output)
    data = tracks if isinstance(tracks, dict) else json.loads(Path(tracks).read_text())
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not src_width or not src_height or not frame_count:
        raise RuntimeError("Video has invalid dimensions or frame count")
    output_fps = float(fps or data.get("fps") or src_fps)
    if not math.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("Output FPS must be positive")
    entries = sorted(data.get("frames", []), key=lambda e: float(e.get("time", e.get("frame", 0) / src_fps)))
    if not entries:
        raise ValueError("Tracking JSON has no frames")
    times = [float(e.get("time", e.get("frame", 0) / src_fps)) for e in entries]
    duration = min(frame_count / src_fps, times[-1] + 1 / output_fps)
    if max_seconds is not None:
        duration = min(duration, max_seconds)
    output_count = max(1, int(math.ceil(duration * output_fps - 1e-7)))
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(output.stem + ".intermediate.mp4")
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the intermediate MP4")
    base = base_canvas(data.get("meta", {}))
    scale = min(WIDTH / src_width, VIDEO_HEIGHT / src_height)
    fw, fh = round(src_width * scale), round(src_height * scale)
    ox, oy = (WIDTH - fw) // 2, VIDEO_TOP + (VIDEO_HEIGHT - fh) // 2
    # JSON pixel geometry can refer to an inference size different from the clip.
    geometry_width = int(data.get("width") or src_width)
    geometry_height = int(data.get("height") or src_height)
    history: dict[int, deque] = defaultdict(deque)
    source_index, current_frame, last_entry = -1, None, -1
    written = 0
    preview_index = min(output_count - 1, round(output_count * 0.45))
    try:
        for out_index in range(output_count):
            t = out_index / output_fps
            target_source = min(frame_count - 1, int(round(t * src_fps)))
            while source_index < target_source:
                ok, f = cap.read()
                if not ok:
                    break
                current_frame = f
                source_index += 1
            if current_frame is None or source_index < target_source:
                break
            canvas = base.copy()
            canvas[oy : oy + fh, ox : ox + fw] = cv2.resize(current_frame, (fw, fh), interpolation=cv2.INTER_AREA)
            ei = max(0, bisect.bisect_right(times, t + 0.5 / src_fps) - 1)
            entry = entries[ei]
            # Avoid presenting long-expired observations as current measurements.
            players = entry.get("players", []) if abs(t - times[ei]) <= max(0.20, 1.5 / output_fps) else []
            if ei != last_entry:
                for p in players:
                    point = p.get("court")
                    if point is not None and len(point) == 2 and all(math.isfinite(float(z)) for z in point):
                        history[int(p["id"])].append((times[ei], float(point[0]), float(point[1])))
                last_entry = ei
            for points in history.values():
                while points and points[0][0] < t - 2.0:
                    points.popleft()
            video_labels = draw_video_players(canvas, players, geometry_width, geometry_height, (ox, oy, fw, fh))
            map_labels = []
            for p in players:
                track_id = int(p["id"])
                team = int(p.get("team", 0)) % 2
                color = TEAMS[team]
                points = list(history[track_id])
                # Break trails on observation gaps, map jumps, or off-map points.
                for a, z in zip(points[:-1], points[1:]):
                    if z[0] - a[0] > 0.22 or math.hypot(z[1] - a[1], z[2] - a[2]) > 1.5:
                        continue
                    if not all(-0.6 <= q[1] <= COURT_LENGTH + 0.6 and -0.6 <= q[2] <= COURT_WIDTH + 0.6 for q in (a, z)):
                        continue
                    age = max(0.0, min(1.0, (t - z[0]) / 2.0))
                    trail_color = tuple(round(PANEL[k] + (color[k] - PANEL[k]) * (0.25 + 0.55 * (1 - age))) for k in range(3))
                    cv2.line(canvas, court_point(a[1], a[2]), court_point(z[1], z[2]), bgr(trail_color), 3, cv2.LINE_AA)
                point = p.get("court")
                if point is None or len(point) != 2:
                    continue
                x, y = map(float, point)
                if not math.isfinite(x + y) or not (-0.6 <= x <= COURT_LENGTH + 0.6 and -0.6 <= y <= COURT_WIDTH + 0.6):
                    continue
                px, py = court_point(x, y)
                cv2.circle(canvas, (px, py), 11, bgr(BG), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 8, bgr(color), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 8, bgr(WHITE), 1, cv2.LINE_AA)
                map_labels.append((label_for(p), px, py, color))
            progress = min(1.0, (out_index + 1) / output_count)
            cv2.rectangle(canvas, (0, VIDEO_TOP + VIDEO_HEIGHT + 1), (WIDTH, VIDEO_TOP + VIDEO_HEIGHT + 3), bgr((29, 46, 62)), -1)
            cv2.rectangle(canvas, (0, VIDEO_TOP + VIDEO_HEIGHT + 1), (round(WIDTH * progress), VIDEO_TOP + VIDEO_HEIGHT + 3), bgr(TEAMS[0]), -1)
            image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(image)
            for label, rect, color in video_labels:
                d.rounded_rectangle(rect, radius=5, fill=color)
                d.text((rect[0] + 7, rect[1] + 2), label, font=FONTS_BOLD[18], fill=BG)
            map_label_rects = []
            for label, px, py, color in map_labels:
                tw = round(FONTS_BOLD[16].getlength(label)) + 10
                candidates = [
                    (px + 11, py - 26),
                    (px + 11, py + 5),
                    (px - tw - 11, py - 26),
                    (px - tw - 11, py + 5),
                ]
                positions = []
                for xx, yy in candidates:
                    xx = max(31, min(WIDTH - 31 - tw, xx))
                    yy = max(COURT_Y - 19, min(1261 - 25, yy))
                    positions.append((xx, yy, xx + tw, yy + 25))
                rect = next((r for r in positions if not any(_overlaps(r, old) for old in map_label_rects)), positions[0])
                map_label_rects.append(rect)
                d.rounded_rectangle(rect, radius=4, fill=BG)
                d.text((rect[0] + 5, rect[1] + 2), label, font=FONTS_BOLD[16], fill=color)
            d.rounded_rectangle((24, VIDEO_TOP + 15, 144, VIDEO_TOP + 49), radius=7, fill=(8, 15, 27, 255))
            time_label = f"{int(t) // 60:02d}:{t % 60:05.2f}"
            d.text((35, VIDEO_TOP + 21), time_label, font=FONTS_BOLD[18], fill=WHITE)
            final_frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            writer.write(final_frame)
            written += 1
            if preview is not None and out_index == preview_index:
                preview_path = Path(preview)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(preview_path, quality=95)
    finally:
        cap.release()
        writer.release()
    if written == 0:
        raise RuntimeError("No video frames were rendered")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(intermediate)]
    if keep_audio:
        command += ["-i", str(video), "-map", "0:v:0", "-map", "1:a?"]
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    if keep_audio:
        command += ["-c:a", "aac", "-b:a", "128k", "-t", str(written / output_fps)]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", str(output)]
    subprocess.run(command, check=True)
    intermediate.unlink(missing_ok=True)
    return {"output": str(output), "frames": written, "fps": output_fps, "duration": written / output_fps, "width": WIDTH, "height": HEIGHT, "preview": str(preview) if preview else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--preview")
    p.add_argument("--fps", type=float)
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--keep-audio", action="store_true")
    args = p.parse_args()
    print(json.dumps(render(args.video, args.tracks, args.output, preview=args.preview, fps=args.fps, max_seconds=args.max_seconds, keep_audio=args.keep_audio), indent=2))


if __name__ == "__main__":
    main()
