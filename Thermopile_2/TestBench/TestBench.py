"""
testbench_pc.py

PIR + Thermopile Sensor Fusion -- Intrusion Detection Test Bench (PC side)
===========================================================================

Purpose
-------
Provides an independent, camera-based "ground truth" of when a person is in
frame, and compares it against the detection state reported by the embedded
platform (Rock 4C+ running the PIR/thermopile fusion pipeline) over a serial
link. From the two time-stamped streams it computes:

  - Event-based Probability of Detection (Pd) and false alarms
        Each continuous "person present" period from the camera is treated
        as one intrusion trial. This matches the intended use case: an
        intruder walks past the sensor, they do not loiter in frame.
  - Time-weighted Pd and Pfa
        Classical detection-theory definitions (fraction of "person" time
        detected, fraction of "no person" time falsely flagged), matching
        the Pd >= 95%% / Pfa <= 5%% targets in the project brief.

Embedded -> PC protocol (adjust EmbeddedListener._parse if yours differs)
--------------------------------------------------------------------------
The embedded system is expected to send ASCII lines over USB-serial, one per
fusion decision cycle:
    "1\n"   -> embedded currently believes a person/intruder is present
    "0\n"   -> embedded currently believes no person is present

Usage
-----
1. Install dependencies:  pip install -r requirements.txt
2. Edit the CONFIG block below (camera index, serial port, baud rate).
3. Run:  python testbench_pc.py
4. A live window shows the camera feed with the ground-truth bounding
   box(es) and both states. Press 'q' to stop the run and write results.

Output (written to testbench_results/<timestamp>/)
----------------------------------------------------
  events.csv     - every state transition from both sources, timestamped
  summary.json   - Pd / Pfa / false alarm counts, both metrics
  timeline.png   - plot of both state streams over the run
"""

import os
import sys
import csv
import json
import time
import threading
from collections import deque
from datetime import datetime

import cv2

try:
    import serial
except ImportError:
    serial = None


# ============================== CONFIG ======================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

SERIAL_PORT = "COM5"          # Windows e.g. "COM5"; Linux e.g. "/dev/ttyUSB0"
SERIAL_BAUD = 115200

# If True, no serial connection is made; a synthetic "embedded" signal is
# generated instead (mirrors ground truth with lag + occasional noise) so
# you can test/debug the PC side and stats engine before hardware is ready.
SIMULATE_EMBEDDED = False
SIM_LATENCY_S = 0.4           # simulated detection lag behind ground truth
SIM_MISS_PROB = 0.05          # chance a given gt episode is missed entirely
SIM_FALSE_ALARM_RATE_PER_MIN = 0.5

# Ground-truth detector backend: tries ultralytics YOLO first, falls back
# to OpenCV's built-in HOG person detector if ultralytics isn't installed.
YOLO_MODEL = "yolov8n.pt"     # auto-downloaded by ultralytics on first run
YOLO_CONF_THRESHOLD = 0.5
PERSON_CLASS_ID = 0           # COCO class id for "person"

# Debounce: a frame-by-frame detector flickers. Require MIN_HITS detections
# in the last WINDOW frames before ground truth reports "person present".
GT_DEBOUNCE_WINDOW = 3
GT_DEBOUNCE_MIN_HITS = 2

OUTPUT_DIR = "testbench_results"

# =============================================================================


class SharedState:
    """Thread-safe latest-known state for both sources, for live display."""

    def __init__(self):
        self.lock = threading.Lock()
        self.gt_state = False
        self.gt_boxes = []
        self.gt_frame = None
        self.emb_state = False
        self.emb_connected = False
        self.running = True

    def set_gt(self, state, boxes, frame):
        with self.lock:
            self.gt_state = state
            self.gt_boxes = boxes
            self.gt_frame = frame

    def set_emb(self, state, connected=True):
        with self.lock:
            self.emb_state = state
            self.emb_connected = connected

    def snapshot(self):
        with self.lock:
            return (self.gt_state, list(self.gt_boxes), self.gt_frame,
                    self.emb_state, self.emb_connected)


class EventLogger:
    """Append-only, thread-safe log of state transitions for both sources."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rows = []  # (timestamp, source, state)

    def log(self, source, state, t=None):
        t = t if t is not None else time.time()
        with self.lock:
            self.rows.append((t, source, bool(state)))

    def get_rows(self):
        with self.lock:
            return list(self.rows)

    def save_csv(self, path):
        rows = sorted(self.get_rows(), key=lambda r: r[0])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_unix", "timestamp_iso", "source", "state"])
            for t, source, state in rows:
                w.writerow([f"{t:.4f}", datetime.fromtimestamp(t).isoformat(),
                            source, int(state)])


# ------------------------------ Ground truth --------------------------------

class GroundTruthDetector(threading.Thread):
    """Runs the camera + person detector in its own thread."""

    def __init__(self, shared, logger):
        super().__init__(daemon=True)
        self.shared = shared
        self.logger = logger
        self.hit_window = deque(maxlen=GT_DEBOUNCE_WINDOW)
        self.backend = None
        self.model = None
        self.hog = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL)
            self.backend = "yolo"
            print(f"[GT] Loaded YOLO model: {YOLO_MODEL}")
        except Exception as e:
            print(f"[GT] ultralytics unavailable ({e}); "
                  f"falling back to OpenCV HOG person detector.")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.backend = "hog"

    def _detect(self, frame):
        boxes = []
        if self.backend == "yolo":
            results = self.model.predict(frame, verbose=False,
                                          conf=YOLO_CONF_THRESHOLD,
                                          classes=[PERSON_CLASS_ID])
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    boxes.append((x1, y1, x2, y2, conf))
        else:
            rects, weights = self.hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, w, h), conf in zip(rects, weights):
                boxes.append((x, y, x + w, y + h, float(conf)))
        return boxes

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not cap.isOpened():
            print("[GT] ERROR: could not open camera.")
            self.shared.running = False
            return

        prev_state = False
        try:
            while self.shared.running:
                ret, frame = cap.read()
                if not ret:
                    print("[GT] Camera read failed.")
                    time.sleep(0.05)
                    continue

                boxes = self._detect(frame)
                raw_hit = len(boxes) > 0
                self.hit_window.append(raw_hit)
                state = sum(self.hit_window) >= GT_DEBOUNCE_MIN_HITS

                t = time.time()
                self.shared.set_gt(state, boxes, frame)
                if state != prev_state:
                    self.logger.log("ground_truth", state, t)
                    prev_state = state
        finally:
            cap.release()


# ------------------------------ Embedded link --------------------------------

class EmbeddedListener(threading.Thread):
    """Reads state updates from the embedded platform over serial."""

    def __init__(self, shared, logger, port, baud):
        super().__init__(daemon=True)
        self.shared = shared
        self.logger = logger
        self.port = port
        self.baud = baud
        self.ser = None

    @staticmethod
    def _parse(line):
        line = line.strip()
        if line in ("1", "PERSON", "DETECT"):
            return True
        if line in ("0", "NONE", "CLEAR"):
            return False
        return None

    def run(self):
        if serial is None:
            print("[EMB] pyserial not installed. Run: pip install pyserial")
            self.shared.set_emb(False, connected=False)
            return

        prev_state = False
        while self.shared.running:
            if self.ser is None:
                try:
                    self.ser = serial.Serial(self.port, self.baud, timeout=1)
                    print(f"[EMB] Connected to {self.port} @ {self.baud} baud")
                    self.shared.set_emb(prev_state, connected=True)
                except Exception as e:
                    print(f"[EMB] Could not open {self.port}: {e}. Retrying in 2s...")
                    time.sleep(2)
                    continue
            try:
                raw = self.ser.readline().decode(errors="ignore")
                if not raw:
                    continue
                state = self._parse(raw)
                if state is None:
                    continue
                t = time.time()
                self.shared.set_emb(state, connected=True)
                if state != prev_state:
                    self.logger.log("embedded", state, t)
                    prev_state = state
            except Exception as e:
                print(f"[EMB] Serial error: {e}. Reconnecting...")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                self.shared.set_emb(prev_state, connected=False)
                time.sleep(1)


class SimulatedEmbedded(threading.Thread):
    """
    Stands in for the embedded system when SIMULATE_EMBEDDED = True.
    Mirrors ground-truth "person present" episodes after a fixed latency,
    occasionally misses one entirely, and injects random false alarms.
    Lets you exercise the whole pipeline (and sanity-check the stats
    engine) before hardware is available.
    """

    def __init__(self, shared, logger):
        super().__init__(daemon=True)
        self.shared = shared
        self.logger = logger

    def run(self):
        import random
        prev_gt = False
        prev_emb = False
        self.shared.set_emb(False, connected=True)
        next_false_alarm = time.time() + random.expovariate(
            SIM_FALSE_ALARM_RATE_PER_MIN / 60.0)

        while self.shared.running:
            gt_state, _, _, _, _ = self.shared.snapshot()

            if gt_state and not prev_gt:
                # New ground-truth episode started; decide now whether the
                # simulated embedded system will catch it.
                if random.random() > SIM_MISS_PROB:
                    def fire(delay=SIM_LATENCY_S):
                        time.sleep(delay)
                        if self.shared.running:
                            t = time.time()
                            self.shared.set_emb(True, connected=True)
                            self.logger.log("embedded", True, t)
                    threading.Thread(target=fire, daemon=True).start()
            if not gt_state and prev_gt:
                def clear(delay=SIM_LATENCY_S * 0.5):
                    time.sleep(delay)
                    if self.shared.running:
                        t = time.time()
                        self.shared.set_emb(False, connected=True)
                        self.logger.log("embedded", False, t)
                threading.Thread(target=clear, daemon=True).start()

            if time.time() >= next_false_alarm and not gt_state:
                t = time.time()
                self.shared.set_emb(True, connected=True)
                self.logger.log("embedded", True, t)
                time.sleep(1.0)
                t = time.time()
                self.shared.set_emb(False, connected=True)
                self.logger.log("embedded", False, t)
                next_false_alarm = time.time() + random.expovariate(
                    SIM_FALSE_ALARM_RATE_PER_MIN / 60.0)

            prev_gt = gt_state
            time.sleep(0.05)


# ------------------------------ Stats engine ---------------------------------

def _episodes_from_events(events, run_start, run_end):
    """
    events: sorted list of (t, state) for one source.
    Returns list of (start, end) intervals where state was True, clipped
    to [run_start, run_end].
    """
    episodes = []
    state = False
    ep_start = None
    for t, s in events:
        if s and not state:
            ep_start = t
        elif not s and state and ep_start is not None:
            episodes.append((ep_start, t))
            ep_start = None
        state = s
    if state and ep_start is not None:
        episodes.append((ep_start, run_end))

    clipped = []
    for a, b in episodes:
        a2, b2 = max(a, run_start), min(b, run_end)
        if b2 > a2:
            clipped.append((a2, b2))
    return clipped


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def compute_stats(rows, run_start, run_end):
    gt_events = sorted([(t, s) for t, src, s in rows if src == "ground_truth"])
    emb_events = sorted([(t, s) for t, src, s in rows if src == "embedded"])

    gt_episodes = _episodes_from_events(gt_events, run_start, run_end)
    emb_episodes = _episodes_from_events(emb_events, run_start, run_end)

    # --- Event-based metrics (primary: matches "intruder walks past") -----
    detected = 0
    missed = 0
    for gt_ep in gt_episodes:
        if any(_overlaps(gt_ep, emb_ep) for emb_ep in emb_episodes):
            detected += 1
        else:
            missed += 1
    total_trials = detected + missed
    pd_event = detected / total_trials if total_trials else float("nan")

    false_alarms = [emb_ep for emb_ep in emb_episodes
                     if not any(_overlaps(emb_ep, gt_ep) for gt_ep in gt_episodes)]
    n_false_alarms = len(false_alarms)
    run_hours = max((run_end - run_start) / 3600.0, 1e-9)
    fa_per_hour = n_false_alarms / run_hours

    # --- Time-weighted metrics (secondary, matches Pd/Pfa spec wording) ---
    all_times = sorted(set([run_start, run_end]
                            + [t for t, _ in gt_events]
                            + [t for t, _ in emb_events]))
    gt_state = False
    emb_state = False
    gt_idx = emb_idx = 0
    tp = fn = fp = tn = 0.0

    for i in range(len(all_times) - 1):
        t0, t1 = all_times[i], all_times[i + 1]
        while gt_idx < len(gt_events) and gt_events[gt_idx][0] <= t0:
            gt_state = gt_events[gt_idx][1]
            gt_idx += 1
        while emb_idx < len(emb_events) and emb_events[emb_idx][0] <= t0:
            emb_state = emb_events[emb_idx][1]
            emb_idx += 1
        dt = t1 - t0
        if gt_state and emb_state:
            tp += dt
        elif gt_state and not emb_state:
            fn += dt
        elif not gt_state and emb_state:
            fp += dt
        else:
            tn += dt

    pd_time = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    pfa_time = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "run_duration_s": run_end - run_start,
        "event_based": {
            "gt_episodes": total_trials,
            "detected": detected,
            "missed": missed,
            "pd": pd_event,
            "false_alarms": n_false_alarms,
            "false_alarms_per_hour": fa_per_hour,
        },
        "time_weighted": {
            "true_positive_s": tp,
            "false_negative_s": fn,
            "false_positive_s": fp,
            "true_negative_s": tn,
            "pd": pd_time,
            "pfa": pfa_time,
        },
        "_gt_episodes_raw": gt_episodes,
        "_emb_episodes_raw": emb_episodes,
    }


def print_stats(stats):
    eb = stats["event_based"]
    tw = stats["time_weighted"]
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Run duration: {stats['run_duration_s']:.1f} s")
    print("\n-- Event-based (one trial per intruder pass) --")
    print(f"  Ground-truth episodes : {eb['gt_episodes']}")
    print(f"  Detected              : {eb['detected']}")
    print(f"  Missed                : {eb['missed']}")
    pd_str = f"{eb['pd']*100:.1f}%" if eb['gt_episodes'] else "n/a (no episodes)"
    print(f"  Pd                    : {pd_str}")
    print(f"  False alarms          : {eb['false_alarms']}  "
          f"({eb['false_alarms_per_hour']:.2f}/hour)")
    print("\n-- Time-weighted (fraction of seconds) --")
    print(f"  Pd  (TP/(TP+FN))      : {tw['pd']*100:.1f}%")
    print(f"  Pfa (FP/(FP+TN))      : {tw['pfa']*100:.1f}%")
    print("=" * 60)


def save_outputs(rows, stats, run_start, run_end, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    logger = EventLogger()
    logger.rows = rows
    logger.save_csv(os.path.join(out_dir, "events.csv"))

    summary = {k: v for k, v in stats.items() if not k.startswith("_")}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3))
        for ep in stats["_gt_episodes_raw"]:
            ax.axvspan(ep[0] - run_start, ep[1] - run_start, ymin=0.55, ymax=1.0,
                       color="tab:blue", alpha=0.6)
        for ep in stats["_emb_episodes_raw"]:
            ax.axvspan(ep[0] - run_start, ep[1] - run_start, ymin=0.0, ymax=0.45,
                       color="tab:orange", alpha=0.6)
        ax.set_yticks([0.225, 0.775])
        ax.set_yticklabels(["Embedded", "Ground truth"])
        ax.set_xlabel("Time since run start (s)")
        ax.set_title("Detection timeline: ground truth vs embedded")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "timeline.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[STATS] Could not save timeline plot: {e}")

    print(f"\nResults written to: {os.path.abspath(out_dir)}")


# ------------------------------ Main / display --------------------------------

def draw_overlay(frame, boxes, gt_state, emb_state, emb_connected):
    disp = frame.copy()
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(disp, f"{conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def status_text(label, state, color_ok=(0, 255, 0), color_bad=(0, 0, 255)):
        return label + (" PERSON" if state else " CLEAR"), (color_ok if state else color_bad)

    gt_txt, gt_col = status_text("GROUND TRUTH:", gt_state)
    cv2.putText(disp, gt_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gt_col, 2)

    if emb_connected:
        emb_txt, emb_col = status_text("EMBEDDED:", emb_state)
    else:
        emb_txt, emb_col = "EMBEDDED: NOT CONNECTED", (128, 128, 128)
    cv2.putText(disp, emb_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, emb_col, 2)

    agree = (gt_state == emb_state)
    agree_txt = "AGREE" if agree else "MISMATCH"
    agree_col = (0, 255, 0) if agree else (0, 0, 255)
    cv2.putText(disp, agree_txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, agree_col, 2)

    cv2.putText(disp, "Press 'q' to stop and save results", (10, disp.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return disp


def main():
    shared = SharedState()
    logger = EventLogger()

    gt_thread = GroundTruthDetector(shared, logger)
    gt_thread.start()

    if SIMULATE_EMBEDDED:
        print("[MAIN] SIMULATE_EMBEDDED = True: using synthetic embedded signal.")
        emb_thread = SimulatedEmbedded(shared, logger)
    else:
        emb_thread = EmbeddedListener(shared, logger, SERIAL_PORT, SERIAL_BAUD)
    emb_thread.start()

    run_start = time.time()
    print("[MAIN] Test bench running. Press 'q' in the video window to stop.")

    try:
        while shared.running:
            gt_state, boxes, frame, emb_state, emb_connected = shared.snapshot()
            if frame is not None:
                disp = draw_overlay(frame, boxes, gt_state, emb_state, emb_connected)
                cv2.imshow("Test Bench - Ground Truth vs Embedded", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        run_end = time.time()
        shared.running = False
        cv2.destroyAllWindows()
        time.sleep(0.3)  # let threads notice shared.running == False

        rows = logger.get_rows()
        if len(rows) == 0:
            print("[MAIN] No events recorded; nothing to analyse.")
            return

        stats = compute_stats(rows, run_start, run_end)
        print_stats(stats)

        out_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        save_outputs(rows, stats, run_start, run_end, out_dir)


if __name__ == "__main__":
    main()