import argparse
import json
import queue
import socket
import threading
import time

from PIR.interface import PIRAcquisition, sampleRateHz
from PIR.processing import PIRProcessor
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

THERMAL_RATE_HZ = 4.0
THERMAL_FRAME_SHAPE = (24, 32)
THERMAL_FRAME_SIZE = 24 * 32

# Set True when the real MLX90640 is connected and working.
# False keeps the collection path testable by sending dummy 32x24 frames.
USE_THERMOPILE = False
thermopile = None


def sendJson(conn, payload, sendLock):
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    with sendLock:
        conn.sendall(line.encode("utf-8"))


def drainQueue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def calibrationPayload(processor, status):
    payload = {
        KEY_STATUS: status,
        KEY_ROCK_TIME: time.time(),
        KEY_PIR_BASELINE: processor.getBaseline(),
        KEY_PIR_SAMPLE_RATE: float(sampleRateHz),
        KEY_THERMAL_RATE: THERMAL_RATE_HZ,
    }

    if USE_THERMOPILE:
        payload.update(
            {
                KEY_THERMAL_FRAME_SHAPE: list(thermopile.FRAME_SHAPE),
                KEY_THERMAL_BACKGROUND: thermopile.getBackground().reshape(-1).astype(float).tolist(),
                KEY_THERMAL_SIGMA: thermopile.getSigma().reshape(-1).astype(float).tolist(),
            }
        )
    else:
        payload.update(
            {
                KEY_THERMAL_FRAME_SHAPE: list(THERMAL_FRAME_SHAPE),
                KEY_THERMAL_BACKGROUND: [23.0] * THERMAL_FRAME_SIZE,
                KEY_THERMAL_SIGMA: [0.0] * THERMAL_FRAME_SIZE,
            }
        )
    return payload


def pirStreamLoop(conn, processor, stopEvent, sendLock):
    seq = 0
    while not stopEvent.is_set():
        try:
            rockTimestamp, voltage, _relative = processor.outputQueue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            sendJson(
                conn,
                {
                    KEY_TYPE: TYPE_PIR,
                    KEY_SEQ: seq,
                    KEY_ROCK_TIMESTAMP: float(rockTimestamp),
                    KEY_VOLTAGE: float(voltage),
                },
                sendLock,
            )
        except OSError:
            stopEvent.set()
            break
        seq += 1


def dummyThermopileStreamLoop(conn, stopEvent, sendLock):
    seq = 0
    period = 1.0 / THERMAL_RATE_HZ

    while not stopEvent.is_set():
        loopStart = time.monotonic()

        rawFrame = [23.0] * THERMAL_FRAME_SIZE
        # Timestamp after the frame has been obtained.
        rockTimestamp = time.time()

        try:
            sendJson(
                conn,
                {
                    KEY_TYPE: TYPE_THERMAL,
                    KEY_SEQ: seq,
                    KEY_ROCK_TIMESTAMP: rockTimestamp,
                    KEY_FRAME: rawFrame,
                },
                sendLock,
            )
        except OSError:
            stopEvent.set()
            break

        seq += 1
        remaining = period - (time.monotonic() - loopStart)
        if remaining > 0:
            stopEvent.wait(remaining)


def realThermopileStreamLoop(conn, stopEvent, sendLock):
    seq = 0
    while not stopEvent.is_set():
        rawFrame = thermopile.getRawFrame()
        # Timestamp immediately after the complete frame has been read.
        rockTimestamp = time.time()

        try:
            sendJson(
                conn,
                {
                    KEY_TYPE: TYPE_THERMAL,
                    KEY_SEQ: seq,
                    KEY_ROCK_TIMESTAMP: rockTimestamp,
                    KEY_FRAME: [float(value) for value in rawFrame],
                },
                sendLock,
            )
        except OSError:
            stopEvent.set()
            break
        seq += 1


def waitForCommand(conn, expected, sendLock):
    """Wait for one command while still answering PC/ROCK clock-sync pings."""
    buffer = ""
    while True:
        try:
            chunk = conn.recv(1024)
        except OSError:
            return None
        if not chunk:
            return None

        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            message, buffer = buffer.split("\n", 1)
            message = message.strip()
            if not message:
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue

            command = payload.get("cmd")
            if command == CMD_SYNC:
                sendJson(
                    conn,
                    {
                        KEY_TYPE: TYPE_SYNC,
                        KEY_PC_SEND_TIME: payload.get(KEY_PC_SEND_TIME),
                        KEY_ROCK_TIME: time.time(),
                    },
                    sendLock,
                )
                continue

            if command == expected:
                return payload


def waitUntilRockTime(targetTime):
    while True:
        remaining = targetTime - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.02))


def main():
    global thermopile

    parser = argparse.ArgumentParser(description="ROCK 4C+ data-collection agent")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if USE_THERMOPILE:
        print("Warming up thermopile")
        import Thermopile.processing as thermopileModule

        thermopile = thermopileModule
        thermopile.warmUp()
    else:
        print("Thermopile dummy mode enabled")

    print("Starting PIR acquisition")
    acquisitionQueue = queue.Queue(maxsize=4096)
    acquisition = PIRAcquisition(acquisitionQueue)
    processor = PIRProcessor(acquisitionQueue)
    acquisition.start()
    processor.start()

    print("Warming up PIR - keep the scene empty")
    while not processor.initialised:
        time.sleep(0.2)
    print("Sensors ready")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"Listening on port {args.port} for the PC...")

    try:
        while True:
            conn, addr = server.accept()
            sendLock = threading.Lock()
            print(f"PC connected from {addr}")

            with conn:
                sendJson(conn, calibrationPayload(processor, STATUS_READY), sendLock)

                startPayload = waitForCommand(conn, CMD_START_AT, sendLock)
                if startPayload is None:
                    continue

                rockStartTime = float(startPayload[KEY_ROCK_START_TIME])
                print("Start scheduled. Waiting for common T0...")
                waitUntilRockTime(rockStartTime)

                # Remove all PIR samples collected before T0. The next sample placed
                # in the queue is therefore the first recorded PIR sample.
                drainQueue(processor.outputQueue)

                stopEvent = threading.Event()
                sendJson(conn, calibrationPayload(processor, STATUS_RECORDING), sendLock)

                pirThread = threading.Thread(
                    target=pirStreamLoop,
                    args=(conn, processor, stopEvent, sendLock),
                    daemon=True,
                )
                thermalThread = threading.Thread(
                    target=realThermopileStreamLoop if USE_THERMOPILE else dummyThermopileStreamLoop,
                    args=(conn, stopEvent, sendLock),
                    daemon=True,
                )
                pirThread.start()
                thermalThread.start()
                print("Collection started")

                if waitForCommand(conn, CMD_STOP, sendLock) is not None:
                    stopEvent.set()
                    pirThread.join(timeout=1.0)
                    thermalThread.join(timeout=1.0)
                    try:
                        sendJson(conn, {KEY_STATUS: STATUS_STOPPED}, sendLock)
                    except OSError:
                        pass

                print("Collection stopped. Waiting for the next PC connection...")
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop()
        acquisition.stop()
        server.close()
        print("\nAgent shut down")


if __name__ == "__main__":
    main()
