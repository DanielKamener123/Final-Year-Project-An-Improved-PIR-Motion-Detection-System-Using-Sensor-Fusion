"""
PC-side data-collection controller.

Connects to the Rock 4C+ agent (Collection/rock_agent.py), starts a
webcam recording, tells the Rock to start streaming combined feature
vectors, timestamps each vector as it arrives (this machine's clock --
the Rock's clock is not assumed to be in sync with it), and writes
everything to CSV. Press Enter to stop: this also stops the recording
and tells the Rock to stop streaming.

Every run is labelled as either training or testing data (--split, or
you'll be prompted for it) and named after the date and time it was
recorded, so repeated sessions never collide and you always know which
pile a given CSV/video belongs to.

Requires opencv-python on the PC (pip install opencv-python) -- not
previously a project dependency, needed here for the webcam capture.

Usage:
    python -m Collection.pc_collector --host <rock ip> --split train
    python -m Collection.pc_collector --host <rock ip>   # prompts for split
"""

import argparse
import csv
import json
import socket
import threading
import time
from pathlib import Path

import cv2

from Collection.protocol import (
    CMD_START, CMD_STOP, KEY_SEQ, KEY_PIR,
    KEY_THERMAL, KEY_THERMAL_DETECTED, DEFAULT_PORT,
    PIR_VECTOR_LENGTH, THERMAL_FEATURE_NAMES,
    KEY_STATUS, STATUS_READY,
)


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
        chunk = sock.recv(1024)
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
                return


# Asks which pile this session belongs to, if not already given on the
# command line. Loops until a valid answer is given.
def promptForSplit():
    while True:
        answer = input("Is this a training or testing run? [train/test]: ").strip().lower()
        if answer in ("train", "test"):
            return answer
        print("Please type 'train' or 'test'.")


# Reads newline-delimited JSON feature-vector lines from sock until
# stopEvent is set or the connection closes, timestamping and writing
# each one to csvWriter as it arrives.
def receiveLoop(sock, csvWriter, csvLock, stopEvent, rowCounter):
    buffer = ""
    while not stopEvent.is_set():
        try:
            chunk = sock.recv(4096)
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
            pc_timestamp = time.time()
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = [
                pc_timestamp,
                payload.get(KEY_SEQ),
                payload.get(KEY_THERMAL_DETECTED),
                *payload.get(KEY_PIR, [0.0] * PIR_VECTOR_LENGTH),
                *payload.get(KEY_THERMAL, [0.0] * len(THERMAL_FEATURE_NAMES)),
            ]
            with csvLock:
                csvWriter.writerow(row)
            rowCounter[0] += 1


# Records webcam video to videoPath and, separately, each frame's own
# wall-clock timestamp to framesCsvPath -- alignment against the
# feature CSV should use those per-frame timestamps, not an assumed
# constant frame rate, since webcam capture timing can jitter.
def recordVideo(cameraIndex, videoPath, framesCsvPath, stopEvent, startedEvent):
    capture = cv2.VideoCapture(cameraIndex)
    if not capture.isOpened():
        print(f"Could not open camera index {cameraIndex} -- no video will be recorded.")
        startedEvent.set()
        return

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(videoPath), fourcc, 30.0, (width, height))

    with open(framesCsvPath, "w", newline="") as framesFile:
        framesWriter = csv.writer(framesFile)
        framesWriter.writerow(["frame_index", "pc_timestamp"])

        startedEvent.set()
        frame_index = 0
        while not stopEvent.is_set():
            ok, frame = capture.read()
            if not ok:
                continue
            writer.write(frame)
            framesWriter.writerow([frame_index, time.time()])
            frame_index += 1

    capture.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="PC-side data-collection controller")
    parser.add_argument("--host", required=True, help="Rock 4C+ IP address or hostname")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--split", choices=["train", "test"], default=None,
                         help="Label this session as training or testing data. "
                              "Prompted for if not given.")
    parser.add_argument("--label", default=None,
                         help="Optional extra tag added to the filename, "
                              "e.g. --label backyard_night")
    args = parser.parse_args()

    split = args.split or promptForSplit()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sessionName = f"{split}_{args.label}_{timestamp}" if args.label else f"{split}_{timestamp}"

    csvPath = Path(f"{sessionName}_features.csv")
    videoPath = Path(f"{sessionName}_video.mp4")
    framesCsvPath = Path(f"{sessionName}_video_frames.csv")
    metaPath = Path(f"{sessionName}_meta.json")

    print(f"Connecting to Rock 4C+ at {args.host}:{args.port} ...")
    sock = connectWithRetry(args.host, args.port)
    print("Connected. Waiting for both sensors to report ready...")
    waitForReady(sock)
    print("Rock 4C+ ready. Starting camera and recording.")

    videoStopEvent = threading.Event()
    videoStartedEvent = threading.Event()
    videoThread = threading.Thread(
        target=recordVideo,
        args=(args.camera, videoPath, framesCsvPath, videoStopEvent, videoStartedEvent),
        daemon=True,
    )
    videoThread.start()
    videoStartedEvent.wait(timeout=5.0)

    csvFile = open(csvPath, "w", newline="")
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(
        ["pc_timestamp", "seq", "thermal_detected"]
        + [f"pir_{i}" for i in range(PIR_VECTOR_LENGTH)]
        + [f"thermal_{name}" for name in THERMAL_FEATURE_NAMES]
    )
    csvLock = threading.Lock()
    rowCounter = [0]

    collectStopEvent = threading.Event()
    receiveThread = threading.Thread(
        target=receiveLoop,
        args=(sock, csvWriter, csvLock, collectStopEvent, rowCounter),
        daemon=True,
    )

    collectionStart = time.time()
    sendCommand(sock, CMD_START)
    receiveThread.start()

    with open(metaPath, "w") as metaFile:
        json.dump({
            "split": split,
            "collection_start_pc_time": collectionStart,
            "rock_host": args.host,
            "rock_port": args.port,
            "csv_file": str(csvPath),
            "video_file": str(videoPath),
            "video_frames_file": str(framesCsvPath),
            "pir_vector_length": PIR_VECTOR_LENGTH,
            "thermal_feature_names": THERMAL_FEATURE_NAMES,
        }, metaFile, indent=2)

    print(f"Recording {split} session '{sessionName}'. Press Enter to stop...")
    input()

    sendCommand(sock, CMD_STOP)
    collectStopEvent.set()
    videoStopEvent.set()
    receiveThread.join(timeout=2.0)
    videoThread.join(timeout=2.0)

    sock.close()
    csvFile.close()

    print(f"Done. {rowCounter[0]} feature rows written to {csvPath}")
    print(f"Video written to {videoPath}, frame timestamps in {framesCsvPath}")
    print(f"Session metadata in {metaPath}")


if __name__ == "__main__":
    main()
