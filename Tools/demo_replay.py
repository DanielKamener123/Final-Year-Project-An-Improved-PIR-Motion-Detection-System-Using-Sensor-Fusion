"""Demo fallback: replays a recorded session through the real-time system as if it were live.

Run from the project root:
    python -m Tools.demo_replay --recording "Datasets/validation/HomeDadValid"

Use this if the hardware fails during the demo. Every frame is processed while it
plays, exactly as main.py does on the ROCK: the same ThermopilePipeline,
PIRProcessor, first-principles Gradient Boosting and three-state machine
(Fusion/realtimeModel.py). Nothing is pre-computed. The recording supplies the
sensor samples and the background calibration saved during collection.

Keys:
    SPACE       Pause / resume
    R           Restart (from --start)
    V / C       Next / previous thermopile processing stage
    1 - 8       Jump to a thermopile processing stage
    Q / Esc     Quit
"""
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
from Fusion.realtimeModel import RealtimeModel
from PIR.processing import PIRProcessor, threshold as PIR_THRESHOLD
from Thermopile.processing import ThermopilePipeline
from Tools.synchronised_playback import (
    ACCENT, ACCENT_2, BG, BORDER, DISPLAY_H, DISPLAY_W, LEFT_PANEL_W, LEFT_PANEL_X, MUTED, PANEL_ALT,
    RIGHT_PANEL_W, RIGHT_PANEL_X, TEXT, THERMAL_STAGES, WARNING,
    _draw_card, _draw_colour_bar, _draw_pir_plot, _fit_image, _find_video, _nearest_index,
    _pir_display_range, _rounded_rect, _text, _thermal_display_range, _thermal_stage_view,
)

WINDOW_TITLE = "Intrusion Detection - Recorded Demo"
MODEL_PATH = Path("Processed/realtime/fusion_gb.json")
TOP_H = 440
CONFIDENCE_HISTORY_S = 20.0

# Lamp colours (BGR): clear, possible, intruder, off
LAMP_COLOURS = ((113, 204, 46), (15, 196, 241), (60, 76, 231))
LAMP_OFF = (75, 65, 59)


class Replay:
    """Feeds the recording to the real-time system in time order."""

    def __init__(self, recording, model_path):
        self.recording = recording
        self.model_path = model_path
        with (recording / "metadata.json").open(encoding="utf-8") as f:
            self.meta = json.load(f)
        pir = pd.read_csv(recording / "pir.csv").sort_values("time_s")
        thermal = pd.read_csv(recording / "thermopile.csv").sort_values("time_s")
        self.pir_t = pir["time_s"].to_numpy(dtype=float)
        self.pir_v = pir["voltage"].to_numpy(dtype=float)
        pixel_cols = [f"pixel_{i}" for i in range(768)]
        self.thermal_t = thermal["time_s"].to_numpy(dtype=float)
        self.thermal_pixels = thermal[pixel_cols].to_numpy(dtype=np.float64).reshape(-1, 24, 32)
        self.raw_range = _thermal_display_range(thermal)
        self.pir_range = _pir_display_range(self.pir_v)
        self.labels = None
        if (recording / "labels.csv").is_file():
            self.labels = pd.read_csv(recording / "labels.csv")
        self.reset()

    def reset(self):
        self.model = RealtimeModel(self.model_path)
        self.window = self.model.window
        self.pir = PIRProcessor(initialBaseline=self.meta["pir_baseline_v"], windowTime=self.window)
        self.thermal = ThermopilePipeline(window_seconds=self.window)
        self.thermal.load_calibration(self.meta["thermal_background"], self.meta["thermal_sigma"])
        self.next_pir = 0
        self.next_thermal = 0
        self.result = None
        self.status = f"Filling the {self.window:g} s PIR window..."
        self.features = None
        self.background_before = self.thermal.bg_model.bg.copy()
        self.confidence_history = []
        self.process_ms = 0.0

    # Process Every PIR Sample And Thermal Frame Up To Time t (causal, like the live system)
    def advance(self, t):
        while self.next_pir < len(self.pir_t) and self.pir_t[self.next_pir] <= t:
            self.pir.processSample(self.pir_t[self.next_pir], self.pir_v[self.next_pir])
            self.next_pir += 1
        while self.next_thermal < len(self.thermal_t) and self.thermal_t[self.next_thermal] <= t:
            started = time.perf_counter()
            timestamp = float(self.thermal_t[self.next_thermal])
            self.background_before = self.thermal.bg_model.bg.copy()
            self.features = self.thermal.process(self.thermal_pixels[self.next_thermal], timestamp)
            samples = self.pir.getWindowSamples(self.window, endTime=timestamp)
            result, problem = self.model.fuse(self.features, samples, timestamp)
            if result is not None:
                self.result = result
                self.status = "Live decision"
                self.confidence_history.append((timestamp, result["fusion_confidence"]))
            else:
                self.status = problem
            self.process_ms = (time.perf_counter() - started) * 1000.0
            self.next_thermal += 1

    def human_at(self, t):
        if self.labels is None or self.labels.empty:
            return None
        active = (self.labels["start_s"] <= t) & (t < self.labels["end_s"]) & (self.labels["human"] == 1)
        return bool(active.any())

    def frame_stages(self):
        p = self.thermal
        shift = 0.0 if not p.last_frame_valid else p.last_shift
        raw = self.thermal_pixels[max(self.next_thermal - 1, 0)]
        return {
            "raw": raw, "background": self.background_before,
            "difference": raw - (self.background_before + shift),
            "z": p.last_raw_z, "smoothed_z": p.last_smoothed_z, "smoothed_z_avg": p.last_smoothed_z_avg,
            "ratio": p.last_ratio, "mask": p.last_mask,
        }

    def blobs(self):
        p = self.thermal
        out = []
        for d in p.last_detections:
            track = p._tracks.get(d.features.track_id)
            out.append((d.blob.bbox, d.features.track_id, d.features.track_id == p.primary_track_id,
                        track.traversed_px if track is not None else 0.0))
        return out


def draw_lamps(canvas, x, y, title, labels, selected, enabled):
    _text(canvas, title, x, y, 0.58, TEXT, 2)
    for i, label in enumerate(labels):
        cx = x + 34 + 118 * i
        colour = LAMP_COLOURS[i if len(labels) == 3 else (0 if i == 0 else 2)]
        on = enabled and selected == i
        cv2.circle(canvas, (cx, y + 38), 22, colour if on else LAMP_OFF, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, y + 38), 22, BORDER, 2, cv2.LINE_AA)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        _text(canvas, label, cx - size[0] // 2, y + 80, 0.42, TEXT if on else MUTED, 1)


def draw_confidence(canvas, x, y, w, h, history, current_time, thresholds):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_ALT, -1)
    for level, name, lx in ((thresholds["intruder_enter"], "intruder", x + w - 92),
                            (thresholds["maybe_enter"], "maybe", x + 6)):
        ly = int(y + h - level * h)
        cv2.line(canvas, (x, ly), (x + w, ly), BORDER, 1, cv2.LINE_AA)
        _text(canvas, f"{name} {level:.2f}", lx, ly - 4, 0.36, MUTED)
    start = current_time - CONFIDENCE_HISTORY_S
    points = [(t, c) for t, c in history if t >= start]
    if len(points) >= 2:
        pts = np.array([(x + (t - start) / CONFIDENCE_HISTORY_S * w, y + h - c * h) for t, c in points], np.int32)
        cv2.polylines(canvas, [pts], False, ACCENT, 2, cv2.LINE_AA)
    _text(canvas, f"Fused human probability, last {CONFIDENCE_HISTORY_S:g} s", x, y - 8, 0.4, MUTED)


def main():
    parser = argparse.ArgumentParser(description="Replay a recording through the real-time system")
    parser.add_argument("--recording", required=True, type=Path, help="Path to one recording folder")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Exported first-principles model")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start the replay this many seconds in (earlier data is still processed)")
    args = parser.parse_args()

    replay = Replay(args.recording, args.model)
    frames = pd.read_csv(args.recording / "video_frames.csv")
    video_t = frames["time_s"].to_numpy(dtype=float)
    capture = cv2.VideoCapture(str(_find_video(args.recording)))
    if not capture.isOpened():
        raise RuntimeError("Could not open the recording video")
    usable = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or len(video_t), len(video_t))
    duration = float(video_t[usable - 1])

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    paused = False
    mode = 7          # start on the foreground mask with tracks
    video_index = -1
    video_frame = np.zeros((360, 640, 3), np.uint8)
    start_time = float(max(args.start, min(video_t[0], replay.thermal_t[0])))
    play_time = start_time
    last_wall = time.perf_counter()

    while True:
        now = time.perf_counter()
        if not paused:
            play_time = min(duration, play_time + (now - last_wall))
        last_wall = now

        # Sensors and decisions up to the current time, then the matching video frame
        replay.advance(play_time)
        target = _nearest_index(video_t[:usable], play_time)
        while video_index < target:
            if video_index < target - 1:
                capture.grab()            # skip frames without decoding them
            else:
                ok, image = capture.read()
                if ok:
                    video_frame = image
            video_index += 1

        canvas = np.full((DISPLAY_H, DISPLAY_W, 3), BG, dtype=np.uint8)
        _text(canvas, "OUTDOOR INTRUSION DETECTION - RECORDED DEMO", 24, 34, 0.82, TEXT, 2)
        _text(canvas, f"{args.recording.name}  |  replayed through the real-time system  |  "
                      f"{replay.window:g} s fusion window", 24, 59, 0.46, MUTED, 1)
        _text(canvas, "Paused" if paused else "Live replay", DISPLAY_W - 175, 38, 0.5,
              MUTED if paused else ACCENT_2, 2)

        # Video
        card_y, card_h = 76, TOP_H - 92
        _draw_card(canvas, LEFT_PANEL_X, card_y, LEFT_PANEL_W, card_h, "Camera", f"t = {play_time:.2f}s")
        view = _fit_image(video_frame, LEFT_PANEL_W - 28, card_h - 66, bg=PANEL_ALT)
        canvas[card_y + 54:card_y + 54 + view.shape[0], LEFT_PANEL_X + 14:LEFT_PANEL_X + 14 + view.shape[1]] = view
        human = replay.human_at(play_time)
        if human is not None:
            _rounded_rect(canvas, (LEFT_PANEL_X + 22, card_y + 62), (LEFT_PANEL_X + 282, card_y + 94), BG, 8)
            _text(canvas, f"Ground truth: {'PERSON IN ZONE' if human else 'zone empty'}",
                  LEFT_PANEL_X + 32, card_y + 84, 0.5, WARNING if human else MUTED, 1)

        # Thermopile stage
        _draw_card(canvas, RIGHT_PANEL_X, card_y, RIGHT_PANEL_W, card_h, "Thermopile",
                   f"{replay.process_ms:.1f} ms/frame")
        heat_w, heat_h = RIGHT_PANEL_W - 58, card_h - 150
        tx0, ty0 = RIGHT_PANEL_X + 14, card_y + 54
        if replay.features is not None:
            heat, description, bar = _thermal_stage_view(
                replay.frame_stages(), not replay.thermal.last_frame_valid, replay.blobs(), replay.features,
                mode, replay.raw_range, heat_w, heat_h)
            canvas[ty0:ty0 + heat_h, tx0:tx0 + heat_w] = heat
            _draw_colour_bar(canvas, bar, tx0 + heat_w + 12, ty0, 18, heat_h)
            _text(canvas, description, tx0, card_y + card_h - 77, 0.37, MUTED)
        _text(canvas, f"{mode + 1}/{len(THERMAL_STAGES)} {THERMAL_STAGES[mode][0]} | V/C: stage",
              tx0, card_y + card_h - 61, 0.40, ACCENT_2)

        # PIR signal (only samples already received)
        seen = replay.next_pir
        _draw_pir_plot(canvas, 20, TOP_H, 760, DISPLAY_H - TOP_H - 20, replay.pir_t[:seen], replay.pir_v[:seen],
                       play_time, *replay.pir_range)

        # Decisions
        px, py = LEFT_PANEL_X + 776, TOP_H
        pw, ph = DISPLAY_W - px - 20, DISPLAY_H - TOP_H - 20
        _draw_card(canvas, px, py, pw, ph, "Decision", replay.status)
        result = replay.result
        ready = result is not None
        pir_selected = 1 if ready and result["pir_state"] == "INTRUDER" else 0
        fused_selected = ("NO INTRUDER", "MAYBE INTRUDER", "INTRUDER").index(result["fusion_state"]) if ready else -1
        draw_lamps(canvas, px + 18, py + 70, f"PIR ONLY  (|V| >= {PIR_THRESHOLD:g} V)",
                   ("CLEAR", "INTRUDER"), pir_selected, ready)
        draw_lamps(canvas, px + 18, py + 168, "FUSED  (PIR + thermopile)",
                   ("CLEAR", "POSSIBLE", "INTRUDER"), fused_selected, ready)
        draw_confidence(canvas, px + 18, py + 276, pw - 36, ph - 290, replay.confidence_history, play_time,
                        replay.model.fusion.thresholds)
        if ready:
            _text(canvas, f"{result['fusion_confidence']:.0%}", px + pw - 70, py + 182, 0.7, ACCENT, 2)

        _text(canvas, "SPACE pause   R restart   V/C stage   Q quit", 24, DISPLAY_H - 6, 0.4, MUTED)
        cv2.imshow(WINDOW_TITLE, canvas)

        key = cv2.waitKeyEx(15)
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            replay.reset()
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_index = -1
            play_time = start_time
        elif key == ord("v"):
            mode = (mode + 1) % len(THERMAL_STAGES)
        elif key == ord("c"):
            mode = (mode - 1) % len(THERMAL_STAGES)
        elif ord("1") <= key <= ord("0") + len(THERMAL_STAGES):
            mode = key - ord("1")
        if play_time >= duration and not paused:
            paused = True

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
