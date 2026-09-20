"""
PC-side data-collection controller.

Connects to the Rock 4C+ agent (Collection/rock_agent.py), starts a
webcam recording, tells the Rock to start streaming raw PIR samples and
raw thermopile frames, timestamps every sensor message as it arrives
using this PC's clock, and writes the two sensor streams to separate CSV
files. Type stop to finish: this also stops the recording and tells the
Rock to stop streaming.

Every recording belongs to either training or testing data and to a
named run. Re-opening a current run creates a new recording folder
inside that run instead of overwriting earlier data.

Requires opencv-python on the PC (pip install opencv-python) -- not
previously a project dependency, needed here for the webcam capture.

Usage:
    python -m Collection.pc_collector --host <rock ip>
"""

import argparse
import csv
import json
import socket
import sys
import threading
import time
from pathlib import Path

import cv2

from Collection.protocol import (
    CMD_START,
    CMD_STOP,
    DEFAULT_PORT,
    KEY_FRAME,
    KEY_PIR_BASELINE,
    KEY_SEQ,
    KEY_STATUS,
    KEY_THERMAL_BACKGROUND,
    KEY_THERMAL_FRAME_SHAPE,
    KEY_THERMAL_RATE,
    KEY_THERMAL_SIGMA,
    KEY_TYPE,
    KEY_VOLTAGE,
    STATUS_READY,
    STATUS_RECORDING,
    STATUS_STOPPED,
    TYPE_PIR,
    TYPE_THERMAL,
)

DATASET_ROOT = Path("Datasets")


def sendCommand(sock, cmd):
    sock.sendall((json.dumps({"cmd": cmd}) + "\n").encode("utf-8"))


# Keeps trying to connect until the Rock is actually listening -- it
# only opens the port once both sensors have finished warming up, so a
# refused connection here just means "not ready yet", not an error.
def connectWithRetry(host, port):
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            return sock
        except OSError:
            sock.close()
            print("Rock 4C+ not ready yet (still warming up or not started) -- retrying...")
            time.sleep(1.0)


# Blocks until the Rock sends {"status": "ready"}, confirming both the
# thermopile and PIR have finished warming up.
def waitForReady(sock):
    buffer = ""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Rock 4C+ closed the connection before sending ready status")
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get(KEY_STATUS) == STATUS_READY:
                return payload


# Asks which pile this session belongs to, if not already given on the
# command line. Loops until a valid answer is given.
def promptForSplit():
    while True:
        answer = input("Is this training or testing data? [training/testing]: ").strip().lower()
        if answer in ("train", "training"):
            return "training"
        if answer in ("test", "testing"):
            return "testing"
        print("Please type 'training' or 'testing'.")


def sanitiseName(name):
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return cleaned.strip("_")


def chooseRun(split):
    splitDir = DATASET_ROOT / split
    splitDir.mkdir(parents=True, exist_ok=True)

    while True:
        choice = input("Use a new run or a current run? [new/current]: ").strip().lower()
        if choice == "new":
            runName = sanitiseName(input("Enter a name for the new run: "))
            if not runName:
                print("Run name cannot be empty.")
                continue
            runDir = splitDir / runName
            if runDir.exists():
                print("That run already exists. Choose 'current' to add another recording to it.")
                continue
            runDir.mkdir(parents=True)
            return runName, runDir

        if choice == "current":
            runs = sorted(path for path in splitDir.iterdir() if path.is_dir())
            if not runs:
                print("There are no current runs yet. Create a new run first.")
                continue
            print("Available runs:")
            for index, run in enumerate(runs, start=1):
                print(f"  {index}. {run.name}")
            selection = input("Choose the run number or type its name: ").strip()
            if selection.isdigit():
                index = int(selection) - 1
                if 0 <= index < len(runs):
                    return runs[index].name, runs[index]
            else:
                runDir = splitDir / sanitiseName(selection)
                if runDir.is_dir():
                    return runDir.name, runDir
            print("Could not find that run.")
            continue

        print("Please type 'new' or 'current'.")


def openCamera(cameraIndex):
    # On Windows, DirectShow is considerably more reliable for USB webcams
    # than OpenCV's default MSMF backend. A common MSMF failure mode is that
    # VideoCapture reports "opened" but capture.read() never returns a frame.
    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(cameraIndex, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(cameraIndex)

    if not capture.isOpened():
        return capture

    # Logitech USB cameras are generally happiest delivering MJPG at a normal
    # webcam resolution/rate. These are requests, not hard requirements.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def recordVideo(cameraIndex, videoPath, frameTimesPath, stopEvent, readyEvent, videoInfo):
    capture = openCamera(cameraIndex)
    backend = capture.getBackendName() if capture.isOpened() else "unknown"
    videoInfo["camera_index"] = cameraIndex
    videoInfo["camera_backend"] = backend

    if not capture.isOpened():
        videoInfo["error"] = (
            f"Could not open camera index {cameraIndex}. "
            "For the Logitech USB camera, try --camera 1 first; if necessary test --camera 0 or --camera 2."
        )
        readyEvent.set()
        return

    # Do not trust isOpened(): actually require a frame. This also gives a USB
    # webcam a short period to initialise before the VideoWriter is created.
    firstFrame = None
    firstFrameDeadline = time.monotonic() + 5.0
    while not stopEvent.is_set() and time.monotonic() < firstFrameDeadline:
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            firstFrame = frame
            break
        time.sleep(0.05)

    if firstFrame is None:
        capture.release()
        videoInfo["error"] = (
            f"Camera index {cameraIndex} opened with backend {backend}, but no frame was received. "
            "Close Camera/Teams/Zoom/OBS/Logitech Capture and retry. "
            "If this is not the Logitech camera, use --camera 1 (or test 0/2)."
        )
        readyEvent.set()
        return

    height, width = firstFrame.shape[:2]
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(videoPath), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        videoInfo["error"] = f"Could not open video writer for {videoPath}"
        readyEvent.set()
        return

    frameCount = 0
    firstFrameTime = None

    with open(frameTimesPath, "w", newline="", buffering=1) as frameTimesFile:
        frameTimesWriter = csv.writer(frameTimesFile)
        frameTimesWriter.writerow(["frame_index", "pc_timestamp"])

        # Save the frame already captured during camera warm-up.
        frameTime = time.time()
        firstFrameTime = frameTime
        videoInfo["video_start_pc_time"] = firstFrameTime
        videoInfo["fps"] = fps
        videoInfo["width"] = width
        videoInfo["height"] = height
        writer.write(firstFrame)
        frameTimesWriter.writerow([frameCount, frameTime])
        frameCount += 1
        readyEvent.set()

        consecutiveFailures = 0
        while not stopEvent.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                consecutiveFailures += 1
                if consecutiveFailures >= 30:
                    videoInfo["runtime_warning"] = "Camera stopped returning frames during recording"
                    break
                time.sleep(0.01)
                continue

            consecutiveFailures = 0
            frameTime = time.time()
            writer.write(frame)
            frameTimesWriter.writerow([frameCount, frameTime])
            frameCount += 1

    videoInfo["frame_count"] = frameCount
    videoInfo["video_end_pc_time"] = time.time()
    capture.release()
    writer.release()


def receiveLoop(sock, pirWriter, thermalWriter, stopEvent, state):
    buffer = ""
    while not stopEvent.is_set():
        try:
            chunk = sock.recv(65536)
        except OSError:
            break
        if not chunk:
            break
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            pcTimestamp = time.time()
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            status = payload.get(KEY_STATUS)
            if status == STATUS_RECORDING:
                state["recording_status"] = payload
                state["recording_status_pc_time"] = pcTimestamp
                state["recordingEvent"].set()
                continue
            if status == STATUS_STOPPED:
                state["stoppedEvent"].set()
                continue

            messageType = payload.get(KEY_TYPE)
            if messageType == TYPE_PIR:
                pirWriter.writerow([
                    pcTimestamp,
                    payload.get(KEY_SEQ),
                    payload.get(KEY_VOLTAGE),
                ])
                state["pir_rows"] += 1
            elif messageType == TYPE_THERMAL:
                frame = payload.get(KEY_FRAME, [])
                thermalWriter.writerow([
                    pcTimestamp,
                    payload.get(KEY_SEQ),
                    *frame,
                ])
                state["thermal_rows"] += 1


def writeMetadata(path, payload):
    with open(path, "w", encoding="utf-8") as metadataFile:
        json.dump(payload, metadataFile, indent=2)


def main():
    parser = argparse.ArgumentParser(description="PC-side data-collection controller")
    parser.add_argument("--host", required=True, help="Rock 4C+ IP address or hostname")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index (default 1, intended for the Logitech USB camera on this PC)")
    args = parser.parse_args()

    while input("Type 'start' to begin a recording: ").strip().lower() != "start":
        pass

    split = promptForSplit()
    runName, runDir = chooseRun(split)

    recordingName = time.strftime("recording_%Y%m%d_%H%M%S")
    recordingDir = runDir / recordingName
    recordingDir.mkdir(parents=True)

    pirPath = recordingDir / "pir.csv"
    thermalPath = recordingDir / "thermopile.csv"
    videoPath = recordingDir / "video.mp4"
    videoFramesPath = recordingDir / "video_frames.csv"
    metadataPath = recordingDir / "metadata.json"

    print(f"Connecting to Rock 4C+ at {args.host}:{args.port} ...")
    sock = connectWithRetry(args.host, args.port)
    print("Connected. Waiting for both sensors to report ready...")
    readyPayload = waitForReady(sock)
    print("Both sensors are warmed up and ready.")

    frameShape = readyPayload.get(KEY_THERMAL_FRAME_SHAPE, [24, 32])
    frameLength = int(frameShape[0]) * int(frameShape[1])

    pirFile = open(pirPath, "w", newline="", buffering=1)
    thermalFile = open(thermalPath, "w", newline="", buffering=1)
    pirWriter = csv.writer(pirFile)
    thermalWriter = csv.writer(thermalFile)
    pirWriter.writerow(["pc_timestamp", "seq", "voltage"])
    thermalWriter.writerow(["pc_timestamp", "seq"] + [f"pixel_{i}" for i in range(frameLength)])

    videoStopEvent = threading.Event()
    videoReadyEvent = threading.Event()
    videoInfo = {}
    videoThread = threading.Thread(
        target=recordVideo,
        args=(args.camera, videoPath, videoFramesPath, videoStopEvent, videoReadyEvent, videoInfo),
        daemon=True,
    )
    print(f"Opening camera index {args.camera} (Logitech USB camera expected)...")
    videoThread.start()
    cameraReady = videoReadyEvent.wait(timeout=7.0)
    if videoInfo.get("error"):
        pirFile.close()
        thermalFile.close()
        sock.close()
        raise RuntimeError(videoInfo["error"])
    if not cameraReady or videoInfo.get("video_start_pc_time") is None:
        videoStopEvent.set()
        pirFile.close()
        thermalFile.close()
        sock.close()
        raise RuntimeError("Camera opened but no video frame was received within 5 seconds")

    state = {
        "pir_rows": 0,
        "thermal_rows": 0,
        "recording_status": None,
        "recording_status_pc_time": None,
        "recordingEvent": threading.Event(),
        "stoppedEvent": threading.Event(),
    }
    receiveStopEvent = threading.Event()
    receiveThread = threading.Thread(
        target=receiveLoop,
        args=(sock, pirWriter, thermalWriter, receiveStopEvent, state),
        daemon=True,
    )
    receiveThread.start()

    collectionStart = time.time()
    sendCommand(sock, CMD_START)
    state["recordingEvent"].wait(timeout=3.0)

    print(f"Recording {split}/{runName}/{recordingName}")
    while input("Type 'stop' to finish this recording: ").strip().lower() != "stop":
        pass

    collectionStop = time.time()
    sendCommand(sock, CMD_STOP)
    state["stoppedEvent"].wait(timeout=2.0)

    receiveStopEvent.set()
    videoStopEvent.set()
    receiveThread.join(timeout=2.0)
    videoThread.join(timeout=2.0)

    sock.close()
    pirFile.close()
    thermalFile.close()

    recordingStatus = state.get("recording_status") or readyPayload
    metadata = {
        "split": split,
        "run_name": runName,
        "recording_name": recordingName,
        "collection_start_pc_time": collectionStart,
        "collection_stop_pc_time": collectionStop,
        "rock_host": args.host,
        "rock_port": args.port,
        "pir_file": pirPath.name,
        "thermopile_file": thermalPath.name,
        "video_file": videoPath.name,
        "video_frames_file": videoFramesPath.name,
        "pir_rows": state["pir_rows"],
        "thermopile_rows": state["thermal_rows"],
        "video_start_pc_time": videoInfo.get("video_start_pc_time"),
        "video_end_pc_time": videoInfo.get("video_end_pc_time"),
        "video_fps": videoInfo.get("fps"),
        "video_width": videoInfo.get("width"),
        "video_height": videoInfo.get("height"),
        "video_frame_count": videoInfo.get("frame_count"),
        "camera_index": videoInfo.get("camera_index"),
        "camera_backend": videoInfo.get("camera_backend"),
        "camera_runtime_warning": videoInfo.get("runtime_warning"),
        "pir_baseline_v": recordingStatus.get(KEY_PIR_BASELINE),
        "pir_sample_rate_hz": recordingStatus.get("pir_sample_rate_hz"),
        "thermal_rate_hz": recordingStatus.get(KEY_THERMAL_RATE),
        "thermal_frame_shape": recordingStatus.get(KEY_THERMAL_FRAME_SHAPE),
        "thermal_background": recordingStatus.get(KEY_THERMAL_BACKGROUND),
        "thermal_sigma": recordingStatus.get(KEY_THERMAL_SIGMA),
        "recording_status_pc_time": state.get("recording_status_pc_time"),
    }
    writeMetadata(metadataPath, metadata)

    print(f"Saved {state['pir_rows']} PIR samples to {pirPath}")
    print(f"Saved {state['thermal_rows']} thermopile frames to {thermalPath}")
    print(f"Saved video to {videoPath}")
    print(f"Saved per-frame PC timestamps to {videoFramesPath}")
    print(f"Saved session information to {metadataPath}")
    print("After Label Studio annotation, place labels.csv in this recording folder.")


if __name__ == "__main__":
    main()
