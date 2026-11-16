from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

# Import package from project root even when this script is run by filepath.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Thermopile.processing import ThermopilePipeline


THERMAL_HEIGHT = 24
THERMAL_WIDTH = 32
PIR_HISTORY_SECONDS = 4.0
PIR_FUTURE_SECONDS = 0.5
WINDOW_TITLE = "Sensor Fusion Playback"
TIMELINE_MAX = 10000

# Thermopile processing stages shown in the thermal card (name, key)
THERMAL_STAGES = [
    ("Raw temperature + detections", "raw"),
    ("Background model", "background"),
    ("Difference from background", "difference"),
    ("Z score (current frame)", "z"),
    ("Filtered Z score", "smoothed_z"),
    ("Filtered Z of recent-frame average", "smoothed_z_avg"),
    ("Threshold ratio (>1 = foreground)", "ratio"),
    ("Foreground mask, blobs and tracks", "mask"),
]
Z_DISPLAY_MAX = 10.0
RATIO_DISPLAY_MAX = 3.0
DIFFERENCE_DISPLAY_C = 5.0

# Layout
DISPLAY_W = 1280
DISPLAY_H = 800
TOP_H = 440
PLOT_H = 230
CONTROL_H = DISPLAY_H - TOP_H - PLOT_H
PANEL_GAP = 16
PANEL_PAD = 16
LEFT_PANEL_X = 20
LEFT_PANEL_W = 780
RIGHT_PANEL_X = LEFT_PANEL_X + LEFT_PANEL_W + PANEL_GAP
RIGHT_PANEL_W = DISPLAY_W - RIGHT_PANEL_X - 20

# BGR colours
BG = (24, 26, 30)
PANEL = (35, 38, 44)
PANEL_ALT = (43, 47, 54)
BORDER = (65, 70, 78)
TEXT = (235, 238, 242)
MUTED = (165, 172, 182)
ACCENT = (90, 180, 255)
ACCENT_2 = (110, 220, 165)
WARNING = (90, 190, 255)
HUMAN_COLOUR = (95, 225, 130)
HUMAN_ACTIVE_COLOUR = (70, 245, 245)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


BUTTONS: dict[str, tuple[int, int, int, int]] = {}
TIMELINE_RECT = (24, DISPLAY_H - 34, DISPLAY_W - 24, DISPLAY_H - 16)
_ui_state = {
    "paused": False,
    "frame_index": 0,
    "seek_target": None,
    "timeline_drag": False,
    "human_start": None,
    "labels_dirty": False,
    "status_message": "",
    "status_until": 0.0,
}


def _find_video(recording_dir: Path) -> Path:
    for name in ("video.mp4", "video_labelstudio.mp4"):
        path = recording_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No video.mp4 or video_labelstudio.mp4 found in {recording_dir}")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _time_column(df: pd.DataFrame) -> str:
    for column in ("time_s", "pc_timestamp", "rock_timestamp"):
        if column in df.columns:
            return column
    raise ValueError("Expected one of: time_s, pc_timestamp, rock_timestamp")


def _normalise_times(video_frames, pir, thermal):
    video_col = _time_column(video_frames)
    pir_col = _time_column(pir)
    thermal_col = _time_column(thermal)

    video_t = video_frames[video_col].to_numpy(dtype=float)
    pir_t = pir[pir_col].to_numpy(dtype=float)
    thermal_t = thermal[thermal_col].to_numpy(dtype=float)

    if video_col == pir_col == thermal_col == "time_s":
        return video_t, pir_t, thermal_t

    origin = float(video_t[0])
    return video_t - origin, pir_t - origin, thermal_t - origin


def _nearest_index(times: np.ndarray, target: float) -> int:
    if len(times) == 0:
        return 0
    i = int(np.searchsorted(times, target))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - target) < abs(times[i - 1] - target) else i - 1


def _thermal_display_range(thermal: pd.DataFrame) -> tuple[float, float]:
    pixel_cols = [c for c in thermal.columns if c.startswith("pixel_")]
    pixel_cols.sort(key=lambda c: int(c.split("_")[1]))
    values = thermal[pixel_cols].to_numpy(dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 20.0, 35.0

    low = float(np.percentile(values, 2.0))
    high = float(np.percentile(values, 98.0))
    if high - low < 4.0:
        midpoint = 0.5 * (low + high)
        low = midpoint - 2.0
        high = midpoint + 2.0
    return low, high


def _pir_display_range(pir_v: np.ndarray) -> tuple[float, float]:
    values = pir_v[np.isfinite(pir_v)]
    if values.size == 0:
        return 0.0, 3.3

    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high - low < 0.2:
        midpoint = 0.5 * (low + high)
        low = midpoint - 0.1
        high = midpoint + 0.1
    padding = 0.08 * (high - low)
    return low - padding, high + padding


def _load_labels(recording_dir: Path, usable_frames: int, video_t: np.ndarray) -> np.ndarray:
    labels = np.zeros(usable_frames, dtype=np.uint8)

    # Current dataset format used by Fusion/dataset.py
    interval_path = recording_dir / "labels.csv"
    if interval_path.exists():
        try:
            existing = pd.read_csv(interval_path)
        except Exception:
            existing = None

        if existing is not None and {"start_s", "end_s", "human"}.issubset(existing.columns):
            frame_times = video_t[:usable_frames]
            for _, row in existing.iterrows():
                if int(row["human"]) != 1:
                    continue
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
                labels[(frame_times >= start_s) & (frame_times < end_s)] = 1
            return labels

    # Backwards compatibility with the earlier frame-by-frame label file.
    legacy_path = recording_dir / "label.csv"
    if not legacy_path.exists():
        return labels

    try:
        existing = pd.read_csv(legacy_path)
    except Exception:
        return labels

    if "label" not in existing.columns:
        return labels

    if "frame_index" in existing.columns:
        indices = existing["frame_index"].to_numpy(dtype=int)
        values = existing["label"].to_numpy(dtype=int)
        valid = (indices >= 0) & (indices < usable_frames)
        labels[indices[valid]] = (values[valid] > 0).astype(np.uint8)
    else:
        values = existing["label"].to_numpy(dtype=int)
        count = min(usable_frames, len(values))
        labels[:count] = (values[:count] > 0).astype(np.uint8)

    return labels


def _load_predictions(path: Path):
    """Load timestamped OFFLINE decisions, never infer them from the video."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"time_s", "confidence", "state", "human"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: expected columns {sorted(required)}")
    df = df.sort_values("time_s", kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No predictions in {path}")
    if not np.isfinite(df[["time_s", "confidence", "state", "human"]].to_numpy(dtype=float)).all():
        raise ValueError("Prediction CSV contains nonfinite values")
    times = df["time_s"].to_numpy(dtype=float)
    scores = df["confidence"].to_numpy(dtype=float)
    states = df["state"].to_numpy(dtype=int)
    human = df["human"].to_numpy(dtype=int)
    if np.any(~np.isin(states, (0, 1, 2))) or np.any(~np.isin(human, (0, 1))):
        raise ValueError("Prediction state must be 0/1/2 and human must be 0/1")
    # Use the exported project false-alarm flag when present
    if "false_alarm" in df.columns:
        false_mask = df["false_alarm"].to_numpy(dtype=int) == 1
    else:
        false_mask = (states > 0) & (human == 0)
    labelled = df["labelled"].to_numpy(dtype=int) == 1 if "labelled" in df.columns else np.ones(len(df), dtype=bool)
    # One episode is a contiguous run of incorrect decisions. Split long data gaps.
    starts = np.flatnonzero(false_mask & ~np.r_[False, false_mask[:-1]])
    if len(times) > 1:
        gaps = np.flatnonzero(false_mask[1:] & false_mask[:-1] & (np.diff(times) > 1.0)) + 1
        starts = np.sort(np.r_[starts, gaps])
    return {"time": times, "score": scores, "state": states, "human": human, "labelled": labelled,
            "false": false_mask, "starts": times[starts]}


def _review_at(predictions, timestamp):
    """Select latest available causal decision, not a future prediction."""
    if predictions is None:
        return None
    i = int(np.searchsorted(predictions["time"], timestamp, side="right") - 1)
    if i < 0:
        return None
    human = int(predictions["human"][i]) if predictions["labelled"][i] else None
    return (float(predictions["score"][i]), int(predictions["state"][i]),
            human, bool(predictions["false"][i]),
            float(timestamp - predictions["time"][i]))


def _jump_false_alarm(predictions, timestamp, forward=True):
    if predictions is None or not len(predictions["starts"]):
        return None
    starts = predictions["starts"]
    if forward:
        i = int(np.searchsorted(starts, timestamp + 1e-6, side="right"))
    else:
        i = int(np.searchsorted(starts, timestamp - 1e-6, side="left") - 1)
    if not 0 <= i < len(starts):
        return None
    return float(starts[i])


def _save_labels(recording_dir: Path, video_t: np.ndarray, labels: np.ndarray) -> Path:
    path = recording_dir / "labels.csv"
    runs = _label_runs(labels)

    if len(video_t) > 1:
        positive_steps = np.diff(video_t)
        positive_steps = positive_steps[positive_steps > 0]
        frame_step = float(np.median(positive_steps)) if positive_steps.size else 0.0
    else:
        frame_step = 0.0

    rows = []
    for start_index, end_index in runs:
        start_s = float(video_t[start_index])
        if end_index + 1 < len(video_t):
            end_s = float(video_t[end_index + 1])
        else:
            end_s = float(video_t[end_index] + frame_step)

        rows.append({
            "start_s": start_s,
            "end_s": end_s,
            "human": 1,
        })

    output = pd.DataFrame(rows, columns=["start_s", "end_s", "human"])
    output.to_csv(path, index=False)
    return path


def _commit_human_interval(labels: np.ndarray, start_index: int, end_index: int, history: list) -> None:
    start = int(np.clip(min(start_index, end_index), 0, len(labels) - 1))
    end = int(np.clip(max(start_index, end_index), 0, len(labels) - 1))
    previous = labels[start:end + 1].copy()
    labels[start:end + 1] = 1
    history.append((start, end, previous))


def _label_runs(labels: np.ndarray) -> list[tuple[int, int]]:
    if len(labels) == 0 or not np.any(labels):
        return []
    padded = np.pad(labels.astype(np.int8), (1, 1), mode="constant")
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _rounded_rect(img, p1, p2, colour, radius=14, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    if thickness != -1:
        cv2.rectangle(img, p1, p2, colour, thickness, cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), colour, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), colour, -1)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(img, (cx, cy), radius, colour, -1, cv2.LINE_AA)


def _text(img, text, x, y, scale=0.55, colour=TEXT, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def _fit_image(image: np.ndarray, width: int, height: int, bg=PANEL_ALT) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _draw_card(canvas, x, y, w, h, title, subtitle=None):
    _rounded_rect(canvas, (x, y), (x + w, y + h), PANEL, 16)
    cv2.rectangle(canvas, (x, y + 48), (x + w, y + h), PANEL, 1)
    _text(canvas, title, x + 18, y + 31, 0.68, TEXT, 2)
    if subtitle:
        size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        _text(canvas, subtitle, x + w - size[0] - 18, y + 30, 0.48, MUTED, 1)


def _draw_pir_plot(canvas, x, y, w, h, pir_t, pir_v, current_time, v_min, v_max):
    _draw_card(canvas, x, y, w, h, "PIR Signal", f"Window: -{PIR_HISTORY_SECONDS:.1f}s to +{PIR_FUTURE_SECONDS:.1f}s")

    px0, px1 = x + 68, x + w - 24
    py0, py1 = y + 66, y + h - 34
    cv2.rectangle(canvas, (px0, py0), (px1, py1), PANEL_ALT, -1)

    start_t = current_time - PIR_HISTORY_SECONDS
    end_t = current_time + PIR_FUTURE_SECONDS
    mask = (pir_t >= start_t) & (pir_t <= end_t)
    t = pir_t[mask]
    v = pir_v[mask]

    # Grid
    for i in range(5):
        gx = int(px0 + i * (px1 - px0) / 4)
        cv2.line(canvas, (gx, py0), (gx, py1), BORDER, 1)
    for i in range(4):
        gy = int(py0 + i * (py1 - py0) / 3)
        cv2.line(canvas, (px0, gy), (px1, gy), BORDER, 1)

    # Voltage-axis labels aligned with the horizontal grid lines
    for i in range(4):
        voltage = v_max - i * (v_max - v_min) / 3.0
        gy = int(py0 + i * (py1 - py0) / 3)
        _text(canvas, f"{voltage:.3f} V", x + 8, gy + 5, 0.39, MUTED)

    if len(pir_t) == 0:
        message = "No PIR samples recorded in pir.csv"
        size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        _text(canvas, message, px0 + (px1 - px0 - size[0]) // 2, py0 + (py1 - py0) // 2, 0.7, WARNING, 2)

    if len(v) > 0:
        valid_mask = np.isfinite(v)
        if np.any(valid_mask):
            t_plot = t[valid_mask]
            v_plot = np.clip(v[valid_mask], v_min, v_max)
            xs = px0 + ((t_plot - start_t) / max(end_t - start_t, 1e-9) * (px1 - px0))
            ys = py1 - ((v_plot - v_min) / (v_max - v_min) * (py1 - py0))
            pts = np.column_stack((xs, ys)).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, ACCENT_2, 2, cv2.LINE_AA)

    cursor_x = int(px0 + (current_time - start_t) / max(end_t - start_t, 1e-9) * (px1 - px0))
    cv2.line(canvas, (cursor_x, py0), (cursor_x, py1), ACCENT, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cursor_x, py0 + 6), 4, ACCENT, -1, cv2.LINE_AA)

    _text(canvas, f"{max(0.0, start_t):.2f}s", px0, y + h - 10, 0.40, MUTED)
    end_label = f"{end_t:.2f}s"
    label_w = cv2.getTextSize(end_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
    _text(canvas, end_label, px1 - label_w, y + h - 10, 0.40, MUTED)


def _draw_button(canvas, name, label, x, y, w, h, active=False):
    BUTTONS[name] = (x, y, x + w, y + h)
    colour = ACCENT if active else PANEL_ALT
    _rounded_rect(canvas, (x, y), (x + w, y + h), colour, 12)
    label_colour = BLACK if active else TEXT
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    _text(canvas, label, x + (w - size[0]) // 2, y + (h + size[1]) // 2 - 2, 0.5, label_colour, 1)


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def _draw_controls(canvas, current_time, duration, paused, thermal_offset, labels, frame_index, human_start):
    y = TOP_H + PLOT_H + 10
    BUTTONS.clear()

    # Playback controls
    x = 24
    _draw_button(canvas, "restart", "Restart", x, y + 8, 78, 40)
    x += 88
    _draw_button(canvas, "back5", "-5 s", x, y + 8, 60, 40)
    x += 70
    _draw_button(canvas, "back1", "-1 s", x, y + 8, 60, 40)
    x += 70
    _draw_button(canvas, "playpause", "Play" if paused else "Pause", x, y + 8, 84, 40, active=not paused)
    x += 94
    _draw_button(canvas, "forward1", "+1 s", x, y + 8, 60, 40)
    x += 70
    _draw_button(canvas, "forward5", "+5 s", x, y + 8, 60, 40)

    # Label controls. Everything is non-human by default; only marked intervals become human.
    human_active = human_start is not None
    _draw_button(
        canvas, "human", "HUMAN END" if human_active else "HUMAN START",
        620, y + 8, 142, 40, active=human_active
    )
    _draw_button(canvas, "undo", "Undo Human", 772, y + 8, 118, 40)
    _draw_button(canvas, "save", "Save labels", 900, y + 8, 118, 40, active=_ui_state.get("labels_dirty", False))
    if _ui_state.get("review_predictions") is not None:
        _draw_button(canvas, "previous_false", "Prev FA", 1030, y + 58, 104, 30)
        _draw_button(canvas, "next_false", "Next FA", 1140, y + 58, 104, 30)

    _text(canvas, f"{_format_time(current_time)} / {_format_time(duration)}", 1035, y + 25, 0.46, TEXT, 1)
    status = "PAUSED" if paused else "PLAYING"
    status_colour = MUTED if paused else ACCENT_2
    _text(canvas, status, 1035, y + 44, 0.40, status_colour, 1)

    human_frames = int(np.count_nonzero(labels))
    human_seconds = 0.0 if len(labels) <= 1 or duration <= 0 else human_frames / max(len(labels) - 1, 1) * duration
    label_text = f"Human marked: {human_seconds:.1f}s"
    if human_active:
        label_text += "  |  recording human interval..."
    _text(canvas, label_text, 24, y + 67, 0.44, HUMAN_ACTIVE_COLOUR if human_active else MUTED, 1)

    if _ui_state.get("status_message") and time.perf_counter() < _ui_state.get("status_until", 0.0):
        _text(canvas, _ui_state["status_message"], 820, y + 67, 0.44, ACCENT_2, 1)

    # In-window timeline. Human intervals are shown as green bands above the line.
    x1, y1, x2, y2 = TIMELINE_RECT
    timeline_y = (y1 + y2) // 2

    denom = max(len(labels) - 1, 1)
    for start_i, end_i in _label_runs(labels):
        rx1 = int(x1 + (start_i / denom) * (x2 - x1))
        rx2 = int(x1 + (end_i / denom) * (x2 - x1))
        cv2.rectangle(canvas, (rx1, y1 - 7), (max(rx1 + 2, rx2), y1 - 2), HUMAN_COLOUR, -1)

    if human_active:
        start_i = min(int(human_start), int(frame_index))
        end_i = max(int(human_start), int(frame_index))
        rx1 = int(x1 + (start_i / denom) * (x2 - x1))
        rx2 = int(x1 + (end_i / denom) * (x2 - x1))
        cv2.rectangle(canvas, (rx1, y1 - 7), (max(rx1 + 2, rx2), y1 - 2), HUMAN_ACTIVE_COLOUR, -1)

    cv2.line(canvas, (x1, timeline_y), (x2, timeline_y), BORDER, 6, cv2.LINE_AA)
    progress = 0.0 if duration <= 0 else float(np.clip(current_time / duration, 0.0, 1.0))
    knob_x = int(x1 + progress * (x2 - x1))
    cv2.line(canvas, (x1, timeline_y), (knob_x, timeline_y), ACCENT, 6, cv2.LINE_AA)
    cv2.circle(canvas, (knob_x, timeline_y), 9, WHITE, -1, cv2.LINE_AA)
    cv2.circle(canvas, (knob_x, timeline_y), 9, ACCENT, 2, cv2.LINE_AA)



def _set_timeline_from_x(x: int) -> None:
    x1, _, x2, _ = TIMELINE_RECT
    pos = (float(x) - x1) / max(1, x2 - x1)
    _ui_state["timeline_pos"] = int(np.clip(pos, 0.0, 1.0) * TIMELINE_MAX)
    _ui_state["timeline_changed"] = True


def _on_mouse(event, x, y, flags, param):
    x1, y1, x2, y2 = TIMELINE_RECT

    if event == cv2.EVENT_LBUTTONDOWN:
        if x1 <= x <= x2 and y1 - 8 <= y <= y2 + 8:
            _ui_state["timeline_drag"] = True
            _set_timeline_from_x(x)
            return
        for name, (bx1, by1, bx2, by2) in BUTTONS.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                _ui_state["seek_target"] = name
                return

    elif event == cv2.EVENT_MOUSEMOVE and _ui_state.get("timeline_drag", False):
        _set_timeline_from_x(x)

    elif event == cv2.EVENT_LBUTTONUP:
        if _ui_state.get("timeline_drag", False):
            _set_timeline_from_x(x)
        _ui_state["timeline_drag"] = False


def _replay_thermal(recording_dir: Path, pixels: np.ndarray, times: np.ndarray):
    """Run every thermal frame through the real ThermopilePipeline once, keeping each stage."""
    with (recording_dir / "metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    pipeline = ThermopilePipeline((THERMAL_HEIGHT, THERMAL_WIDTH))
    pipeline.load_calibration(meta["thermal_background"], meta["thermal_sigma"])
    shape = (len(pixels), THERMAL_HEIGHT, THERMAL_WIDTH)
    stages = {key: np.zeros(shape, dtype=np.float32) for _, key in THERMAL_STAGES if key != "mask"}
    stages["mask"] = np.zeros(shape, dtype=bool)
    glitch = np.zeros(len(pixels), dtype=bool)
    detections = []
    for i, (frame, t) in enumerate(zip(pixels.astype(np.float64), times)):
        background_before = pipeline.bg_model.bg.copy()
        features = pipeline.process(frame, t)
        glitch[i] = not pipeline.last_frame_valid
        stages["raw"][i] = frame
        stages["background"][i] = background_before
        # Difference the pipeline actually used: background moved by the scene shift
        stages["difference"][i] = frame - (background_before + (0.0 if glitch[i] else pipeline.last_shift))
        stages["z"][i] = pipeline.last_raw_z
        stages["smoothed_z"][i] = pipeline.last_smoothed_z
        stages["smoothed_z_avg"][i] = pipeline.last_smoothed_z_avg
        stages["ratio"][i] = pipeline.last_ratio
        stages["mask"][i] = pipeline.last_mask
        blobs = []
        for d in pipeline.last_detections:
            track = pipeline._tracks.get(d.features.track_id)
            blobs.append((d.blob.bbox, d.features.track_id, d.features.track_id == pipeline.primary_track_id,
                          track.traversed_px if track is not None else 0.0))
        detections.append((blobs, features))
    return stages, glitch, detections


def _colour_map(values, low, high, colormap):
    scaled = np.clip((np.asarray(values, dtype=np.float32) - low) * 255.0 / max(high - low, 1e-9), 0, 255)
    return cv2.applyColorMap(scaled.astype(np.uint8), colormap)


def _diverging(values, limit):
    """Blue below zero, red above zero, dark at zero (BGR)."""
    v = np.clip(np.asarray(values, dtype=np.float32) / limit, -1.0, 1.0)
    img = np.full(v.shape + (3,), 30, dtype=np.float32)
    img[..., 2] += 225 * np.clip(v, 0, 1)
    img[..., 0] += 225 * np.clip(-v, 0, 1)
    return img.astype(np.uint8)


def _draw_colour_bar(canvas, bar, x, y, w, h):
    if bar is None:
        return
    colormap, high_label, low_label = bar
    gradient = np.linspace(255, 0, h, dtype=np.float32).reshape(-1, 1)
    if colormap == "diverging":
        colour_bar = _diverging(gradient / 127.5 - 1.0, 1.0)
    else:
        colour_bar = cv2.applyColorMap(gradient.astype(np.uint8), colormap)
    canvas[y:y + h, x:x + w] = np.repeat(colour_bar, w, axis=1)
    _text(canvas, high_label, x - 10, y - 7, 0.38, MUTED)
    _text(canvas, low_label, x - 10, y + h + 15, 0.38, MUTED)


def _thermal_stage_view(frame_stages, glitch, blobs, features, mode, raw_range, width, height):
    """Render one processing stage of one frame at the display size.

    frame_stages: {stage key: 24x32 array}. Returns image, description and colour bar.
    """
    key = THERMAL_STAGES[mode][1]
    raw_low, raw_high = raw_range
    if key in ("raw", "background"):
        img = _colour_map(frame_stages[key], raw_low, raw_high, cv2.COLORMAP_INFERNO)
        bar = (cv2.COLORMAP_INFERNO, f"{raw_high:.0f}C", f"{raw_low:.0f}C")
    elif key == "difference":
        img = _diverging(frame_stages[key], DIFFERENCE_DISPLAY_C)
        bar = ("diverging", f"+{DIFFERENCE_DISPLAY_C:g}C", f"-{DIFFERENCE_DISPLAY_C:g}C")
    elif key in ("z", "smoothed_z", "smoothed_z_avg"):
        img = _colour_map(frame_stages[key], 0.0, Z_DISPLAY_MAX, cv2.COLORMAP_VIRIDIS)
        bar = (cv2.COLORMAP_VIRIDIS, f"{Z_DISPLAY_MAX:g}", "0")
    elif key == "ratio":
        img = _colour_map(frame_stages[key], 0.0, RATIO_DISPLAY_MAX, cv2.COLORMAP_VIRIDIS)
        bar = (cv2.COLORMAP_VIRIDIS, f"{RATIO_DISPLAY_MAX:g}x", "0")
    else:
        grey = np.clip((frame_stages["raw"] - raw_low) / max(raw_high - raw_low, 1e-9) * 140 + 35, 0, 255)
        img = cv2.cvtColor(grey.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img[frame_stages["mask"]] = (0, 220, 255)
        bar = None
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    if key in ("raw", "mask"):
        sx, sy = width / THERMAL_WIDTH, height / THERMAL_HEIGHT
        for (r0, c0, r1, c1), track_id, primary, traversed in blobs:
            colour = ACCENT_2 if primary else (255, 255, 0)
            p1 = (int(c0 * sx), int(r0 * sy))
            p2 = (int((c1 + 1) * sx) - 1, int((r1 + 1) * sy) - 1)
            cv2.rectangle(img, p1, p2, colour, 3 if primary else 1, cv2.LINE_AA)
            _text(img, f"ID {track_id}  {traversed:.1f}px", p1[0] + 2, max(14, p1[1] - 4), 0.42, colour, 1)

    if glitch:
        description = "Glitch frame: ignored by the pipeline"
    else:
        description = (f"Blobs {int(features['thermalNumberOfBlobs'])}  |  "
                       f"max traversed {features['thermalMaxBlobTraversed']:.1f}px  |  "
                       f"max Z {features['thermalMaxZ']:.1f}")
    return img, description, bar


def main() -> None:
    parser = argparse.ArgumentParser(description="Modern video + PIR + thermopile playback viewer")
    parser.add_argument("--recording", required=True, help="Path to one recording folder")
    parser.add_argument("--predictions", type=Path, help="Optional exported review_predictions.csv")
    args = parser.parse_args()

    recording_dir = Path(args.recording)
    prediction_path = args.predictions or recording_dir / "review_predictions.csv"
    predictions = _load_predictions(prediction_path)
    if predictions is None:
        print(f"No prediction file at {prediction_path}; standard labelling viewer will still work.")
    video_path = _find_video(recording_dir)
    pir = _load_csv(recording_dir / "pir.csv")
    thermal = _load_csv(recording_dir / "thermopile.csv")
    video_frames = _load_csv(recording_dir / "video_frames.csv")

    if "voltage" not in pir.columns:
        raise ValueError("pir.csv does not contain a 'voltage' column")

    video_t, pir_t, thermal_t = _normalise_times(video_frames, pir, thermal)
    pir_v = pir["voltage"].to_numpy(dtype=float)
    pir_display_min, pir_display_max = _pir_display_range(pir_v)
    thermal_display_min, thermal_display_max = _thermal_display_range(thermal)
    thermal_pixel_cols = sorted((c for c in thermal.columns if c.startswith("pixel_")), key=lambda c: int(c.split("_")[1]))
    thermal_pixels = thermal[thermal_pixel_cols].to_numpy(dtype=np.float32).reshape(-1, THERMAL_HEIGHT, THERMAL_WIDTH)
    print("Running the thermopile frames through the processing pipeline...")
    thermal_stages, thermal_glitch, thermal_detections = _replay_thermal(recording_dir, thermal_pixels, thermal_t)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count_video = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    usable_frames = min(frame_count_video, len(video_frames))
    if usable_frames <= 0:
        raise RuntimeError("No usable video frames found")

    duration = float(video_t[usable_frames - 1])

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_TITLE, _on_mouse)

    _ui_state.update({
        "paused": False,
        "frame_index": 0,
        "seek_target": None,
        "timeline_pos": 0,
        "timeline_changed": False,
        "timeline_drag": False,
        "human_start": None,
        "labels_dirty": False,
        "status_message": "",
        "status_until": 0.0,
        "thermal_mode": 0,
        "review_predictions": predictions,
    })

    labels = _load_labels(recording_dir, usable_frames, video_t)
    label_history = []
    frame_index = 0
    play_wall_start = time.perf_counter()
    play_data_start = float(video_t[frame_index])

    while True:
        # Handle timeline slider changes.
        if _ui_state["timeline_changed"]:
            _ui_state["timeline_changed"] = False
            target_time = duration * _ui_state["timeline_pos"] / TIMELINE_MAX
            frame_index = _nearest_index(video_t[:usable_frames], target_time)
            _ui_state["paused"] = True

        # Handle mouse buttons.
        action = _ui_state.get("seek_target")
        _ui_state["seek_target"] = None
        if action:
            current = float(video_t[frame_index])
            if action == "playpause":
                _ui_state["paused"] = not _ui_state["paused"]
                if not _ui_state["paused"]:
                    play_wall_start = time.perf_counter()
                    play_data_start = current
            elif action == "restart":
                frame_index = 0
                _ui_state["paused"] = True
            elif action == "back1":
                frame_index = _nearest_index(video_t[:usable_frames], max(0.0, current - 1.0))
                _ui_state["paused"] = True
            elif action == "forward1":
                frame_index = _nearest_index(video_t[:usable_frames], min(duration, current + 1.0))
                _ui_state["paused"] = True
            elif action == "back5":
                frame_index = _nearest_index(video_t[:usable_frames], max(0.0, current - 5.0))
                _ui_state["paused"] = True
            elif action == "forward5":
                frame_index = _nearest_index(video_t[:usable_frames], min(duration, current + 5.0))
                _ui_state["paused"] = True
            elif action == "human":
                if _ui_state["human_start"] is None:
                    _ui_state["human_start"] = frame_index
                    _ui_state["status_message"] = "Human interval started"
                    _ui_state["status_until"] = time.perf_counter() + 2.0
                else:
                    _commit_human_interval(labels, _ui_state["human_start"], frame_index, label_history)
                    _ui_state["human_start"] = None
                    _ui_state["labels_dirty"] = True
                    _ui_state["status_message"] = "Human interval added"
                    _ui_state["status_until"] = time.perf_counter() + 2.0
            elif action == "undo":
                if _ui_state["human_start"] is not None:
                    _ui_state["human_start"] = None
                    _ui_state["status_message"] = "Current human interval cancelled"
                elif label_history:
                    start_i, end_i, previous = label_history.pop()
                    labels[start_i:end_i + 1] = previous
                    _ui_state["labels_dirty"] = True
                    _ui_state["status_message"] = "Last human interval undone"
                else:
                    _ui_state["status_message"] = "Nothing to undo"
                _ui_state["status_until"] = time.perf_counter() + 2.0
            elif action in ("previous_false", "next_false"):
                target = _jump_false_alarm(predictions, current, forward=action == "next_false")
                if target is not None:
                    frame_index = _nearest_index(video_t[:usable_frames], target)
                    _ui_state["paused"] = True
                else:
                    _ui_state["status_message"] = "No more false alarms in this direction"
                    _ui_state["status_until"] = time.perf_counter() + 2.0
            elif action == "save":
                if _ui_state["human_start"] is not None:
                    _commit_human_interval(labels, _ui_state["human_start"], frame_index, label_history)
                    _ui_state["human_start"] = None
                saved_path = _save_labels(recording_dir, video_t[:usable_frames], labels)
                _ui_state["labels_dirty"] = False
                _ui_state["status_message"] = f"Saved {saved_path.name}"
                _ui_state["status_until"] = time.perf_counter() + 3.0

        frame_index = int(np.clip(frame_index, 0, usable_frames - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break

        current_time = float(video_t[frame_index])
        thermal_index = _nearest_index(thermal_t, current_time)
        thermal_time = float(thermal_t[thermal_index])
        thermal_offset = thermal_time - current_time

        canvas = np.full((DISPLAY_H, DISPLAY_W, 3), BG, dtype=np.uint8)

        # Header
        _text(canvas, "SENSOR FUSION PLAYBACK", 24, 34, 0.82, TEXT, 2)
        _text(canvas, recording_dir.name, 24, 59, 0.48, MUTED, 1)
        status_text = "Paused" if _ui_state["paused"] else "Live playback"
        status_colour = MUTED if _ui_state["paused"] else ACCENT_2
        _text(canvas, status_text, DISPLAY_W - 175, 38, 0.5, status_colour, 2)

        # Video card
        video_y = 76
        video_h = TOP_H - 92
        _draw_card(canvas, LEFT_PANEL_X, video_y, LEFT_PANEL_W, video_h, "Camera", f"t = {current_time:.3f}s")
        video_view = _fit_image(frame, LEFT_PANEL_W - 28, video_h - 66, bg=PANEL_ALT)
        vy0 = video_y + 54
        vx0 = LEFT_PANEL_X + 14
        canvas[vy0:vy0 + video_view.shape[0], vx0:vx0 + video_view.shape[1]] = video_view

        # Model review overlay: show actual exported confidence and state.
        review = _review_at(predictions, current_time)
        if review is not None:
            score, state, actual, false, age = review
            state_name = ("NO INTRUDER", "MAYBE", "INTRUDER")[state]
            colour = (ACCENT_2, WARNING, ACCENT)[state]
            _rounded_rect(canvas, (vx0 + 6, vy0 + 8), (vx0 + 365, vy0 + 64), BG, 8)
            _text(canvas, f"ML: {state_name}  |  confidence {score:.3f}", vx0 + 16, vy0 + 30, .52, colour, 2)
            label_name = "NOT LABELLED" if actual is None else ("HUMAN" if actual else "EMPTY")
            _text(canvas, f"Label: {label_name}  |  {'FALSE ALARM' if false else 'No false alarm'}  |  age {age:.2f}s",
                  vx0 + 16, vy0 + 51, .43, WARNING if false else TEXT)
            if false:
                cv2.rectangle(canvas, (vx0, vy0),
                              (vx0 + video_view.shape[1] - 1, vy0 + video_view.shape[0] - 1),
                              WARNING, 3, cv2.LINE_AA)

        # Thermopile card: one processing stage of the real pipeline
        _draw_card(canvas, RIGHT_PANEL_X, video_y, RIGHT_PANEL_W, video_h, "Thermopile", f"t = {thermal_time:.3f}s")
        mode = _ui_state["thermal_mode"]
        colour_bar_w = 18
        colour_bar_gap = 12
        heat_w = RIGHT_PANEL_W - 28 - colour_bar_gap - colour_bar_w
        heat_h = video_h - 150
        frame_stages = {key: values[thermal_index] for key, values in thermal_stages.items()}
        blobs, features = thermal_detections[thermal_index]
        heat, thermal_description, bar = _thermal_stage_view(
            frame_stages, thermal_glitch[thermal_index], blobs, features, mode,
            (thermal_display_min, thermal_display_max), heat_w, heat_h
        )
        tx0, ty0 = RIGHT_PANEL_X + 14, video_y + 54
        canvas[ty0:ty0 + heat_h, tx0:tx0 + heat_w] = heat

        _draw_colour_bar(canvas, bar, tx0 + heat_w + colour_bar_gap, ty0, colour_bar_w, heat_h)

        frame_values = thermal_pixels[thermal_index]
        _text(canvas, f"Frame {float(frame_values.min()):.1f}-{float(frame_values.max()):.1f} C",
              tx0, video_y + video_h - 18, 0.46, MUTED)
        _text(canvas, f"{mode + 1}/{len(THERMAL_STAGES)} {THERMAL_STAGES[mode][0]} | V/C: stage",
              tx0, video_y + video_h - 61, 0.40, ACCENT_2)
        _text(canvas, thermal_description, tx0, video_y + video_h - 77, 0.37,
              WARNING if thermal_glitch[thermal_index] else MUTED)
        offset_colour = ACCENT_2 if abs(thermal_offset) <= 0.15 else WARNING
        _text(canvas, f"Offset {thermal_offset:+.3f}s", tx0, video_y + video_h - 42, 0.48, offset_colour, 1)

        # PIR plot
        _draw_pir_plot(
            canvas, 20, TOP_H, DISPLAY_W - 40, PLOT_H - 10,
            pir_t, pir_v, current_time, pir_display_min, pir_display_max
        )

        # Controls
        _draw_controls(
            canvas, current_time, duration, _ui_state["paused"], thermal_offset,
            labels, frame_index, _ui_state["human_start"]
        )

        cv2.imshow(WINDOW_TITLE, canvas)

        key = cv2.waitKeyEx(25 if _ui_state["paused"] else 1)
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            _ui_state["paused"] = not _ui_state["paused"]
            if not _ui_state["paused"]:
                play_wall_start = time.perf_counter()
                play_data_start = current_time
        elif key in (ord("a"), 2424832):  # A / Left
            frame_index = _nearest_index(video_t[:usable_frames], max(0.0, current_time - 1.0))
            _ui_state["paused"] = True
        elif key in (ord("d"), 2555904):  # D / Right
            frame_index = _nearest_index(video_t[:usable_frames], min(duration, current_time + 1.0))
            _ui_state["paused"] = True
        elif key == ord("j"):
            frame_index = _nearest_index(video_t[:usable_frames], max(0.0, current_time - 5.0))
            _ui_state["paused"] = True
        elif key == ord("l"):
            frame_index = _nearest_index(video_t[:usable_frames], min(duration, current_time + 5.0))
            _ui_state["paused"] = True
        elif key in (ord("n"), ord("p")):
            target = _jump_false_alarm(predictions, current_time, forward=key == ord("n"))
            if target is not None:
                frame_index = _nearest_index(video_t[:usable_frames], target)
                _ui_state["paused"] = True
            else:
                _ui_state["status_message"] = "No more false alarms in this direction"
                _ui_state["status_until"] = time.perf_counter() + 2.0
        elif key in (ord("v"), ord("c")) or ord("1") <= key <= ord("0") + len(THERMAL_STAGES):
            if key == ord("v"):
                _ui_state["thermal_mode"] = (_ui_state["thermal_mode"] + 1) % len(THERMAL_STAGES)
            elif key == ord("c"):
                _ui_state["thermal_mode"] = (_ui_state["thermal_mode"] - 1) % len(THERMAL_STAGES)
            else:
                _ui_state["thermal_mode"] = key - ord("1")
            _ui_state["status_message"] = THERMAL_STAGES[_ui_state["thermal_mode"]][0]
            _ui_state["status_until"] = time.perf_counter() + 2.0
        elif key == ord("h"):
            if _ui_state["human_start"] is None:
                _ui_state["human_start"] = frame_index
                _ui_state["status_message"] = "Human interval started"
            else:
                _commit_human_interval(labels, _ui_state["human_start"], frame_index, label_history)
                _ui_state["human_start"] = None
                _ui_state["labels_dirty"] = True
                _ui_state["status_message"] = "Human interval added"
            _ui_state["status_until"] = time.perf_counter() + 2.0
        elif key == ord("u"):
            if _ui_state["human_start"] is not None:
                _ui_state["human_start"] = None
                _ui_state["status_message"] = "Current human interval cancelled"
            elif label_history:
                start_i, end_i, previous = label_history.pop()
                labels[start_i:end_i + 1] = previous
                _ui_state["labels_dirty"] = True
                _ui_state["status_message"] = "Last human interval undone"
            else:
                _ui_state["status_message"] = "Nothing to undo"
            _ui_state["status_until"] = time.perf_counter() + 2.0
        elif key == ord("s"):
            if _ui_state["human_start"] is not None:
                _commit_human_interval(labels, _ui_state["human_start"], frame_index, label_history)
                _ui_state["human_start"] = None
            saved_path = _save_labels(recording_dir, video_t[:usable_frames], labels)
            _ui_state["labels_dirty"] = False
            _ui_state["status_message"] = f"Saved {saved_path.name}"
            _ui_state["status_until"] = time.perf_counter() + 3.0
        elif key in (ord("r"), 2359296):  # R / Home on many builds
            frame_index = 0
            _ui_state["paused"] = True

        if not _ui_state["paused"]:
            # Use one continuous wall-clock anchor while playing.  The previous
            # implementation reset its timer every GUI iteration, so the
            # elapsed time rarely became large enough to advance one video
            # frame and playback appeared permanently paused.
            target_time = play_data_start + (time.perf_counter() - play_wall_start)
            if target_time >= duration:
                frame_index = usable_frames - 1
                _ui_state["paused"] = True
            else:
                frame_index = _nearest_index(video_t[:usable_frames], target_time)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
