import argparse
import csv
import json
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import cv2

from Collection.protocol import (
    CMD_START_AT,
    CMD_STOP,
    CMD_SYNC,
    DEFAULT_PORT,
    KEY_FRAME,
    KEY_PC_SEND_TIME,
    KEY_PIR_BASELINE,
    KEY_PIR_SAMPLE_RATE,
    KEY_ROCK_START_TIME,
    KEY_ROCK_TIMESTAMP,
    KEY_ROCK_TIME,
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
    TYPE_SYNC,
    TYPE_THERMAL,
)

DATASET_ROOT = Path("Datasets")
VIDEO_FILE_FPS = 30.0

# How Many Clock SYNC Requests
CLOCK_SYNC_PINGS = 10

# How Long To Wait For Both Systems To Start Recording Data
START_DELAY_SECONDS = 2.0

# Sends Commands To The Rock
def sendCommand(sock, command, **fields):
    sock.sendall((json.dumps({"cmd": command, **fields}) + "\n").encode("utf-8"))

# Repeatedly Tries To Connect To The Rock Until It Succeeds
def connectWithRetry(host, port):
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            return sock
        except OSError:
            sock.close()
            print("ROCK not connected, retrying...")
            time.sleep(1.0)

# Waits For Rock To Be Ready (Sensors Calibrated)
def waitForReady(sock):
    buffer = ""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Rock closed the connection before sending ready status")
        buffer += chunk.decode("utf-8")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            dataRecieved = json.loads(line)
            if dataRecieved.get(KEY_STATUS) == STATUS_READY:
                return dataRecieved

# TODO: Look Through - IMPORTANT
def measureClockOffset(sock, state, pings=CLOCK_SYNC_PINGS):
    samples = []
    for _ in range(pings):
        state["syncEvent"].clear()
        pcSend = time.time()
        sendCommand(sock, CMD_SYNC, **{KEY_PC_SEND_TIME: pcSend})
        if state["syncEvent"].wait(timeout=1.0):
            samples.append(state["sync_reply"])

    if not samples:
        return None

    pcSend, rockTime, pcReceive = min(samples, key=lambda sample: sample[2] - sample[0])
    return {
        "offset_s": 0.5 * (pcSend + pcReceive) - rockTime,
        "round_trip_s": pcReceive - pcSend,
        "pings": len(samples),
    }

# Get Data Type
def promptDataType():
    while True:
        answer = input("Is this training or testing data? [training/testing]: ").strip().lower()
        if answer in ("train", "training"):
            return "training"
        if answer in ("test", "testing"):
            return "testing"
        print("Please type 'training' or 'testing'.")

def chooseRun(split):
    splitDir = DATASET_ROOT / split
    splitDir.mkdir(parents=True, exist_ok=True)

    while True:
        choice = input("Use a new run or a current run? [new/current]: ").strip().lower()
        if choice == "new":
            runName = input("Enter a name for the new run: ")
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
                runDir = splitDir / selection
                if runDir.is_dir():
                    return runDir.name, runDir
            print("Could not find that run.")
            continue

        print("Please type 'new' or 'current'.")

# Open Camera With Correct Settings
def openCamera(cameraIndex):
    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(cameraIndex, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(cameraIndex)

    if not capture.isOpened():
        return capture

    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture

# TODO - COMMENTED UP TILL HERE - NEED TO LOOK THROUGH
def recordVideo(cameraIndex, videoPath, frameTimesPath, stopEvent, cameraReadyEvent, startConfiguredEvent, videoInfo):
    # Open Camera
    capture = openCamera(cameraIndex)
    backend = capture.getBackendName() if capture.isOpened() else "unknown"
    videoInfo["camera_index"] = cameraIndex
    videoInfo["camera_backend"] = backend

    if not capture.isOpened():
        videoInfo["error"] = f"Could not open camera index {cameraIndex}. Try --camera 0, 1 or 2."
        cameraReadyEvent.set()
        return

    firstFrame = None
    deadline = time.monotonic() + 5.0
    while not stopEvent.is_set() and time.monotonic() < deadline:
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            firstFrame = frame
            break
        time.sleep(0.05)

    if firstFrame is None:
        capture.release()
        videoInfo["error"] = "Camera opened but did not return a frame within 5 seconds."
        cameraReadyEvent.set()
        return

    height, width = firstFrame.shape[:2]
    videoInfo["camera_reported_fps"] = float(capture.get(cv2.CAP_PROP_FPS))
    videoInfo["width"] = width
    videoInfo["height"] = height
    videoInfo["fps"] = VIDEO_FILE_FPS

    writer = cv2.VideoWriter(
        str(videoPath),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FILE_FPS,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        videoInfo["error"] = f"Could not open video writer for {videoPath}"
        cameraReadyEvent.set()
        return

    cameraReadyEvent.set()

    # Keep grabbing and discarding frames so the webcam is already running when T0 arrives.
    while not stopEvent.is_set() and not startConfiguredEvent.is_set():
        capture.read()

    if stopEvent.is_set():
        capture.release()
        writer.release()
        return

    startPcTime = float(videoInfo["start_pc_time"])
    frameQueue = queue.Queue()

    with open(frameTimesPath, "w", newline="", buffering=1) as frameTimesFile:
        frameTimesWriter = csv.writer(frameTimesFile)
        frameTimesWriter.writerow(["frame_index", "pc_timestamp", "time_s"])

        def writeFrames():
            while True:
                item = frameQueue.get()
                if item is None:
                    return
                index, timestamp, relativeTime, image = item
                writer.write(image)
                frameTimesWriter.writerow([index, timestamp, relativeTime])

        writerThread = threading.Thread(target=writeFrames, daemon=True)
        writerThread.start()

        frameCount = 0
        lastFrameTime = None
        consecutiveFailures = 0

        while not stopEvent.is_set():
            ok, frame = capture.read()
            frameTime = time.time()

            if not ok or frame is None:
                consecutiveFailures += 1
                if consecutiveFailures >= 30:
                    videoInfo["runtime_warning"] = "Camera stopped returning frames during recording"
                    break
                continue

            consecutiveFailures = 0
            if frameTime < startPcTime:
                continue

            relativeTime = frameTime - startPcTime
            frameQueue.put((frameCount, frameTime, relativeTime, frame))
            if frameCount == 0:
                videoInfo["first_frame_pc_time"] = frameTime
                videoInfo["first_frame_time_s"] = relativeTime
            lastFrameTime = frameTime
            frameCount += 1

        frameQueue.put(None)
        writerThread.join()

    videoInfo["frame_count"] = frameCount
    videoInfo["last_frame_pc_time"] = lastFrameTime
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

            pcReceiveTime = time.time()
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            status = payload.get(KEY_STATUS)
            if status == STATUS_RECORDING:
                state["recording_status"] = payload
                state["recordingEvent"].set()
                continue
            if status == STATUS_STOPPED:
                state["stoppedEvent"].set()
                continue

            messageType = payload.get(KEY_TYPE)
            if messageType == TYPE_SYNC:
                state["sync_reply"] = (
                    float(payload[KEY_PC_SEND_TIME]),
                    float(payload[KEY_ROCK_TIME]),
                    pcReceiveTime,
                )
                state["syncEvent"].set()
                continue

            if messageType not in (TYPE_PIR, TYPE_THERMAL):
                continue

            rockTimestamp = float(payload[KEY_ROCK_TIMESTAMP])
            # Convert the ROCK timestamp onto the PC clock using the single
            # offset measured immediately before the recording.
            pcTimestamp = rockTimestamp + float(state["clock_offset_s"])
            relativeTime = pcTimestamp - float(state["start_pc_time"])

            if messageType == TYPE_PIR:
                pirWriter.writerow(
                    [
                        rockTimestamp,
                        pcTimestamp,
                        relativeTime,
                        payload.get(KEY_SEQ),
                        payload.get(KEY_VOLTAGE),
                    ]
                )
                state["pir_rows"] += 1
            else:
                frame = payload.get(KEY_FRAME, [])
                thermalWriter.writerow(
                    [
                        rockTimestamp,
                        pcTimestamp,
                        relativeTime,
                        payload.get(KEY_SEQ),
                        *frame,
                    ]
                )
                state["thermal_rows"] += 1


def writeMetadata(path, payload):
    with open(path, "w", encoding="utf-8") as metadataFile:
        json.dump(payload, metadataFile, indent=2)


def main():
    parser = argparse.ArgumentParser(description="PC-side data-collection controller")
    parser.add_argument("--host", required=True, help="ROCK 4C+ IP address or hostname")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index")
    args = parser.parse_args()

    while input("Type 'start' to prepare a recording: ").strip().lower() != "start":
        pass

    split = promptDataType()
    runName, runDir = chooseRun(split)

    recordingName = time.strftime("recording_%Y%m%d_%H%M%S")
    recordingDir = runDir / recordingName
    recordingDir.mkdir(parents=True)

    pirPath = recordingDir / "pir.csv"
    thermalPath = recordingDir / "thermopile.csv"
    videoPath = recordingDir / "video.mp4"
    videoFramesPath = recordingDir / "video_frames.csv"
    metadataPath = recordingDir / "metadata.json"

    print(f"Connecting to ROCK 4C+ at {args.host}:{args.port} ...")
    sock = connectWithRetry(args.host, args.port)
    readyPayload = waitForReady(sock)
    print("ROCK connected and sensors are ready.")

    frameShape = readyPayload.get(KEY_THERMAL_FRAME_SHAPE, [24, 32])
    frameLength = int(frameShape[0]) * int(frameShape[1])

    pirFile = open(pirPath, "w", newline="", buffering=1)
    thermalFile = open(thermalPath, "w", newline="", buffering=1)
    pirWriter = csv.writer(pirFile)
    thermalWriter = csv.writer(thermalFile)
    pirWriter.writerow(["rock_timestamp", "pc_timestamp", "time_s", "seq", "voltage"])
    thermalWriter.writerow(
        ["rock_timestamp", "pc_timestamp", "time_s", "seq"]
        + [f"pixel_{i}" for i in range(frameLength)]
    )

    state = {
        "pir_rows": 0,
        "thermal_rows": 0,
        "recording_status": None,
        "recordingEvent": threading.Event(),
        "stoppedEvent": threading.Event(),
        "sync_reply": None,
        "syncEvent": threading.Event(),
        "clock_offset_s": None,
        "start_pc_time": None,
    }

    receiveStopEvent = threading.Event()
    receiveThread = threading.Thread(
        target=receiveLoop,
        args=(sock, pirWriter, thermalWriter, receiveStopEvent, state),
        daemon=True,
    )
    receiveThread.start()

    videoStopEvent = threading.Event()
    cameraReadyEvent = threading.Event()
    startConfiguredEvent = threading.Event()
    videoInfo = {}
    videoThread = threading.Thread(
        target=recordVideo,
        args=(
            args.camera,
            videoPath,
            videoFramesPath,
            videoStopEvent,
            cameraReadyEvent,
            startConfiguredEvent,
            videoInfo,
        ),
        daemon=True,
    )
    print(f"Opening and warming camera index {args.camera}...")
    videoThread.start()

    if not cameraReadyEvent.wait(timeout=7.0) or videoInfo.get("error"):
        receiveStopEvent.set()
        videoStopEvent.set()
        sock.close()
        pirFile.close()
        thermalFile.close()
        raise RuntimeError(videoInfo.get("error", "Camera did not become ready"))

    # One simple clock sync immediately before recording.
    clockSync = measureClockOffset(sock, state)
    if clockSync is None:
        receiveStopEvent.set()
        videoStopEvent.set()
        sock.close()
        pirFile.close()
        thermalFile.close()
        raise RuntimeError("ROCK did not answer clock-sync messages")

    clockOffset = float(clockSync["offset_s"])
    state["clock_offset_s"] = clockOffset
    print(
        f"Clock relationship fixed: PC = ROCK {clockOffset:+.4f} s "
        f"(best round trip {clockSync['round_trip_s'] * 1000:.1f} ms)"
    )

    # Give both machines enough time to receive T0 before it occurs.
    startPcTime = time.time() + START_DELAY_SECONDS
    startRockTime = startPcTime - clockOffset
    state["start_pc_time"] = startPcTime
    videoInfo["start_pc_time"] = startPcTime

    sendCommand(sock, CMD_START_AT, **{KEY_ROCK_START_TIME: startRockTime})
    startConfiguredEvent.set()

    print(f"Recording will start automatically in {START_DELAY_SECONDS:.1f} s...")
    if not state["recordingEvent"].wait(timeout=START_DELAY_SECONDS + 3.0):
        receiveStopEvent.set()
        videoStopEvent.set()
        sock.close()
        pirFile.close()
        thermalFile.close()
        raise RuntimeError("ROCK did not start recording at the scheduled time")

    print(f"Recording {split}/{runName}/{recordingName}")
    print("Video, PIR and thermopile now share t = 0 at the scheduled start.")

    while input("Type 'stop' to finish this recording: ").strip().lower() != "stop":
        pass

    stopPcTime = time.time()
    sendCommand(sock, CMD_STOP)
    state["stoppedEvent"].wait(timeout=2.0)

    videoStopEvent.set()
    videoThread.join()

    receiveStopEvent.set()
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
    receiveThread.join(timeout=2.0)

    pirFile.close()
    thermalFile.close()

    recordingStatus = state.get("recording_status") or readyPayload
    frameCount = int(videoInfo.get("frame_count") or 0)
    firstFrameTime = videoInfo.get("first_frame_pc_time")
    lastFrameTime = videoInfo.get("last_frame_pc_time")
    measuredFps = None
    if frameCount > 1 and firstFrameTime is not None and lastFrameTime is not None:
        duration = float(lastFrameTime) - float(firstFrameTime)
        if duration > 0:
            measuredFps = (frameCount - 1) / duration

    metadata = {
        "split": split,
        "run_name": runName,
        "recording_name": recordingName,
        "rock_host": args.host,
        "rock_port": args.port,
        "pir_file": pirPath.name,
        "thermopile_file": thermalPath.name,
        "video_file": videoPath.name,
        "video_frames_file": videoFramesPath.name,
        "common_start_pc_time": startPcTime,
        "common_start_rock_time": startRockTime,
        "stop_pc_time": stopPcTime,
        "rock_to_pc_offset_s": clockOffset,
        "clock_sync_round_trip_s": clockSync["round_trip_s"],
        "clock_sync_pings": clockSync["pings"],
        "pir_rows": state["pir_rows"],
        "thermopile_rows": state["thermal_rows"],
        "video_frame_count": frameCount,
        "first_video_frame_time_s": videoInfo.get("first_frame_time_s"),
        "video_measured_fps": measuredFps,
        "video_file_fps": VIDEO_FILE_FPS,
        "video_width": videoInfo.get("width"),
        "video_height": videoInfo.get("height"),
        "camera_index": videoInfo.get("camera_index"),
        "camera_backend": videoInfo.get("camera_backend"),
        "camera_runtime_warning": videoInfo.get("runtime_warning"),
        "pir_baseline_v": recordingStatus.get(KEY_PIR_BASELINE),
        "pir_sample_rate_hz": recordingStatus.get(KEY_PIR_SAMPLE_RATE),
        "thermal_rate_hz": recordingStatus.get(KEY_THERMAL_RATE),
        "thermal_frame_shape": recordingStatus.get(KEY_THERMAL_FRAME_SHAPE),
        "thermal_background": recordingStatus.get(KEY_THERMAL_BACKGROUND),
        "thermal_sigma": recordingStatus.get(KEY_THERMAL_SIGMA),
        "timing_method": (
            "One PC-ROCK clock offset measured immediately before recording; "
            "both devices start at a scheduled common T0; all CSV time_s values are seconds since T0."
        ),
    }
    writeMetadata(metadataPath, metadata)

    print(f"Saved {state['pir_rows']} PIR samples to {pirPath}")
    print(f"Saved {state['thermal_rows']} thermopile frames to {thermalPath}")
    print(f"Saved {frameCount} video frames to {videoPath}")
    print(f"Saved video frame times to {videoFramesPath}")
    print(f"Saved session information to {metadataPath}")
    print("Next step: label the video, convert the labels, then run main.py.")


if __name__ == "__main__":
    main()
