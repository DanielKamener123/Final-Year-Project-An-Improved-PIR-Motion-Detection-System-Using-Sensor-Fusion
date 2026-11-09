"""Automatic draft labelling of a recording's video (labelling aid, not part of the system).

A pretrained person detector finds people in the video frames. A person is
INBOUND when the bottom centre of their box (their feet) is inside the detection
zone polygon, and OUTBOUND (visible) whenever a person is detected anywhere in
the frame. The draft intervals are written in the same format as the manual
labels, timed from video_frames.csv. Every draft MUST be checked and corrected
in Tools/synchronised_playback.py before it is used.

Zone: on the first run of a session, click the zone corners on the first video
frame (along the strings: sensor end, the -55 deg string, points along the 10 m
end, the +55 deg string), then press Enter. Saved as zone.json in the recording.
Later recordings from the same (unmoved) camera reuse it with --zone.

    python -m Tools.autoLabel --recording Datasets/testing/Driveway_4m_Night
    python -m Tools.autoLabel --recording Datasets/testing/Driveway_10m_Night --zone Datasets/testing/Driveway_4m_Night/zone.json
    python -m Tools.autoLabel --recording Datasets/testing/Driveway_4m_Day --compare

Outputs (in the recording folder):
    labels_auto.csv, labels(outbound)_auto.csv     always written
    labels.csv, labels(outbound).csv               only if they do not exist yet (or --overwrite)
    --compare: agreement of the auto labels with the existing manual labels

Detectors: "yolo" (default; pip install ultralytics, downloads yolov8n.pt on first
use) or "hog" (OpenCV HOG people detector, no extra install, less accurate).
"""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Config
DEFAULT_STRIDE = 3          # process every 3rd frame (30 fps video -> 0.1 s resolution)
DEFAULT_CONFIDENCE = 0.35   # minimum detector confidence for a person
DEFAULT_GAP_S = 0.6         # gaps shorter than this inside one interval are filled
DEFAULT_MIN_S = 0.4         # intervals shorter than this are dropped
YOLO_WEIGHTS = "yolov8n.pt"
PERSON_CLASS = 0
DISPLAY_WIDTH = 1280


# ---------------------------------------------------------------------------
# Zone polygon
# ---------------------------------------------------------------------------

# Lets The User Click The Zone Corners On The First Frame
def drawZone(frame):
    scale = DISPLAY_WIDTH / frame.shape[1]
    shown = cv2.resize(frame, None, fx=scale, fy=scale)
    points = []
    title = "Click zone corners, Enter = done, Backspace = undo, Esc = cancel"

    def onClick(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x / scale, y / scale))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, onClick)
    while True:
        canvas = shown.copy()
        scaled = [(int(px * scale), int(py * scale)) for px, py in points]
        for p in scaled:
            cv2.circle(canvas, p, 5, (0, 255, 255), -1)
        if len(scaled) > 1:
            cv2.polylines(canvas, [np.array(scaled, np.int32)], len(scaled) > 2, (0, 255, 255), 2)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) >= 3:
            break
        if key == 8 and points:
            points.pop()
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Zone drawing cancelled")
    cv2.destroyAllWindows()
    return points


# Loads A Saved Zone Or Asks For A New One
def loadZone(recording, zonePath, firstFrame):
    if zonePath is not None:
        points = json.loads(Path(zonePath).read_text(encoding="utf-8"))["points"]
    else:
        own = recording / "zone.json"
        if own.is_file():
            points = json.loads(own.read_text(encoding="utf-8"))["points"]
        else:
            points = drawZone(firstFrame)
    (recording / "zone.json").write_text(json.dumps({"points": points}, indent=1), encoding="utf-8")
    return np.array(points, dtype=np.float32)


def insideZone(zone, x, y):
    return cv2.pointPolygonTest(zone, (float(x), float(y)), False) >= 0


# ---------------------------------------------------------------------------
# Person detectors: each returns a list of (x1, y1, x2, y2) boxes
# ---------------------------------------------------------------------------

class YoloDetector:
    def __init__(self, confidence):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("ultralytics is not installed: pip install ultralytics (or use --detector hog)")
        self.model = YOLO(YOLO_WEIGHTS)
        self.confidence = confidence

    def detect(self, frame):
        result = self.model(frame, classes=[PERSON_CLASS], conf=self.confidence, verbose=False)[0]
        return [tuple(box) for box in result.boxes.xyxy.cpu().numpy().tolist()]


class HogDetector:
    def __init__(self, confidence):
        if not hasattr(cv2, "HOGDescriptor"):
            raise SystemExit("This OpenCV build has no HOG people detector; use --detector yolo")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.confidence = confidence

    def detect(self, frame):
        scale = 640 / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        boxes, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return [((x) / scale, (y) / scale, (x + w) / scale, (y + h) / scale)
                for (x, y, w, h), weight in zip(boxes, np.ravel(weights)) if weight >= self.confidence]


# ---------------------------------------------------------------------------
# Frames -> intervals
# ---------------------------------------------------------------------------

# Runs The Detector Over Every Stride-th Frame
def scanVideo(recording, detector, stride, zonePath):
    frameTimes = pd.read_csv(recording / "video_frames.csv").sort_values("frame_index")
    timeOf = dict(zip(frameTimes["frame_index"].astype(int), frameTimes["time_s"].astype(float)))
    capture = cv2.VideoCapture(str(recording / "video.mp4"))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open {recording / 'video.mp4'}")
    ok, first = capture.read()
    if not ok:
        raise SystemExit("Video has no frames")
    zone = loadZone(recording, zonePath, first)

    rows = []
    index = 0
    frame = first
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or len(timeOf)
    while ok:
        if index in timeOf:
            boxes = detector.detect(frame)
            feet = [((x1 + x2) / 2.0, y2) for x1, y1, x2, y2 in boxes]
            rows.append({"frame_index": index, "time_s": timeOf[index],
                         "people": len(boxes),
                         "inbound": int(any(insideZone(zone, fx, fy) for fx, fy in feet)),
                         "visible": int(len(boxes) > 0)})
        if index % 300 == 0:
            print(f"  frame {index}/{total}", end="\r")
        # Skip the frames in between quickly (grab without decoding)
        for _ in range(stride - 1):
            if not capture.grab():
                ok = False
                break
            index += 1
        if not ok:
            break
        ok, frame = capture.read()
        index += 1
    capture.release()
    print()
    return pd.DataFrame(rows)


# Turns A 0/1 Series Sampled At times Into Cleaned (start, end) Intervals
def toIntervals(times, flags, gapSeconds, minSeconds):
    times = np.asarray(times, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    intervals = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = times[i]
        if not flag and start is not None:
            intervals.append([start, times[i]])
            start = None
    if start is not None:
        intervals.append([start, times[-1] + step])
    # Fill short dropouts (detector misses for a few frames)
    merged = []
    for interval in intervals:
        if merged and interval[0] - merged[-1][1] < gapSeconds:
            merged[-1][1] = interval[1]
        else:
            merged.append(interval)
    return [(s, e) for s, e in merged if e - s >= minSeconds]


def writeIntervals(path, intervals):
    pd.DataFrame([{"start_s": s, "end_s": e, "human": 1} for s, e in intervals],
                 columns=["start_s", "end_s", "human"]).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Agreement with existing manual labels
# ---------------------------------------------------------------------------

def readIntervals(path):
    if not path.is_file():
        return None
    data = pd.read_csv(path)
    data = data[data["human"].astype(int) == 1]
    return list(zip(data["start_s"].astype(float), data["end_s"].astype(float)))


# Matches Each Manual Interval To The Auto Interval It Overlaps Most
def compareIntervals(manual, auto):
    matched, startDiff, endDiff = set(), [], []
    missed = 0
    for s, e in manual:
        overlaps = [(min(e, ae) - max(s, as_), k) for k, (as_, ae) in enumerate(auto) if min(e, ae) > max(s, as_)]
        if not overlaps:
            missed += 1
            continue
        _, k = max(overlaps)
        matched.add(k)
        startDiff.append(auto[k][0] - s)
        endDiff.append(auto[k][1] - e)
    extra = len(auto) - len(matched)

    # Time-based agreement on a 0.1 s grid
    end = max([e for _, e in manual] + [e for _, e in auto] + [0.0])
    grid = np.arange(0.0, end, 0.1)
    onManual = np.zeros(len(grid), bool)
    onAuto = np.zeros(len(grid), bool)
    for s, e in manual:
        onManual |= (grid >= s) & (grid < e)
    for s, e in auto:
        onAuto |= (grid >= s) & (grid < e)
    agreement = 100.0 * np.mean(onManual == onAuto) if len(grid) else float("nan")
    return {"manual": len(manual), "auto": len(auto), "missed": missed, "extra": extra,
            "startDiff": np.array(startDiff), "endDiff": np.array(endDiff), "agreement": agreement}


def compareText(name, result):
    lines = [f"{name}: manual {result['manual']}, auto {result['auto']}, manual intervals with no auto match "
             f"{result['missed']}, extra auto intervals {result['extra']}, time agreement {result['agreement']:.1f}%"]
    for key, label in (("startDiff", "start"), ("endDiff", "end")):
        d = result[key]
        if len(d):
            lines.append(f"  {label} (auto - manual): mean {d.mean():+.2f} s, mean |diff| {np.abs(d).mean():.2f} s, "
                         f"max |diff| {np.abs(d).max():.2f} s")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Draft labels from the recording video (check them afterwards)")
    parser.add_argument("--recording", required=True, help="Recording folder (contains video.mp4)")
    parser.add_argument("--zone", help="zone.json to reuse (same camera position)")
    parser.add_argument("--detector", choices=("yolo", "hog"), default="yolo")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP_S, help="Fill gaps shorter than this (s)")
    parser.add_argument("--min", type=float, default=DEFAULT_MIN_S, help="Drop intervals shorter than this (s)")
    parser.add_argument("--overwrite", action="store_true", help="Also overwrite labels.csv / labels(outbound).csv")
    parser.add_argument("--compare", action="store_true", help="Report agreement with the existing manual labels")
    args = parser.parse_args()

    recording = Path(args.recording)
    detector = YoloDetector(args.confidence) if args.detector == "yolo" else HogDetector(args.confidence)
    print(f"Scanning {recording} with {args.detector} (every {args.stride} frames)...")
    frames = scanVideo(recording, detector, max(1, args.stride), args.zone)
    frames.to_csv(recording / "auto_detections.csv", index=False)

    inbound = toIntervals(frames["time_s"], frames["inbound"], args.gap, args.min)
    visible = toIntervals(frames["time_s"], frames["visible"] | frames["inbound"], args.gap, args.min)
    writeIntervals(recording / "labels_auto.csv", inbound)
    writeIntervals(recording / "labels(outbound)_auto.csv", visible)
    print(f"Inbound intervals: {len(inbound)}, outbound (visible) intervals: {len(visible)}")

    lines = []
    if args.compare:
        for manualName, auto, label in (("labels.csv", inbound, "Inbound"),
                                        ("labels(outbound).csv", visible, "Outbound")):
            manual = readIntervals(recording / manualName)
            if manual is not None:
                lines += compareText(label, compareIntervals(manual, auto))
        (recording / "auto_label_agreement.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))

    for name, intervals in (("labels.csv", inbound), ("labels(outbound).csv", visible)):
        path = recording / name
        if path.exists() and not args.overwrite:
            print(f"Kept existing {name} (auto version in {name.replace('.csv', '_auto.csv')})")
        else:
            writeIntervals(path, intervals)
            print(f"Wrote {name}: CHECK IT in Tools/synchronised_playback.py before using it")


if __name__ == "__main__":
    main()
