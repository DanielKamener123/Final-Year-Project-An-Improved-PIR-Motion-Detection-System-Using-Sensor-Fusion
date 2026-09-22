import argparse
import json
import queue
import socket
import threading
import time

from PIR.interface import PIRAcquisition, sampleRateHz
from PIR.processing import PIRProcessor
#import Thermopile.processing as thermopile

# Get All Command Variables
from Collection.protocol import (CMD_START, CMD_STOP, DEFAULT_PORT, KEY_FRAME, KEY_PIR_BASELINE, KEY_PIR_SAMPLE_RATE, KEY_SEQ,
                                 KEY_STATUS, KEY_ROCK_TIMESTAMP, KEY_ROCK_TIME, KEY_THERMAL_BACKGROUND, KEY_THERMAL_FRAME_SHAPE, KEY_THERMAL_RATE, KEY_THERMAL_SIGMA, 
                                 KEY_TYPE, KEY_VOLTAGE, STATUS_READY, STATUS_RECORDING, STATUS_STOPPED, TYPE_PIR, TYPE_THERMAL)

THERMAL_RATE_HZ = 4.0
THERMAL_FRAME_SHAPE = (24, 32)
THERMAL_FRAME_SIZE = 24 * 32


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

# Gets The Calibration Data For Both Sensors
#def buildCalibrationPayload(processor, status):
    #return {
    #    KEY_STATUS: status,
    #    KEY_PIR_BASELINE: processor.getBaseline(),
    #    KEY_PIR_SAMPLE_RATE: float(sampleRateHz),
    #    KEY_THERMAL_RATE: THERMAL_RATE_HZ,
    #    KEY_THERMAL_FRAME_SHAPE: list(thermopile.FRAME_SHAPE),
    #    KEY_THERMAL_BACKGROUND: thermopile.getBackground().reshape(-1).astype(float).tolist(),
    #    KEY_THERMAL_SIGMA: thermopile.getSigma().reshape(-1).astype(float).tolist(),
    #}

# Gets The Calibration Data For Both Sensors
def buildCalibrationPayload(processor, status):
    return {
        KEY_STATUS: status,
        KEY_ROCK_TIME: time.time(),
        KEY_PIR_BASELINE: processor.getBaseline(),
        KEY_PIR_SAMPLE_RATE: float(sampleRateHz),

        KEY_THERMAL_RATE: THERMAL_RATE_HZ,
        KEY_THERMAL_FRAME_SHAPE: list(THERMAL_FRAME_SHAPE),

        # Dummy thermopile calibration data
        KEY_THERMAL_BACKGROUND: [0.0] * THERMAL_FRAME_SIZE,
        KEY_THERMAL_SIGMA: [0.0] * THERMAL_FRAME_SIZE,
    }

# Streams The PIR Data To The PC
def pirStreamLoop(conn, processor, stopEvent, sendLock):
    seq = 0
    while not stopEvent.is_set():
        try:
            rockMonotonicTs, voltage, _relative = processor.outputQueue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Convert the acquisition-time monotonic timestamp to the ROCK wall-clock
        # timeline. This preserves the true sampling instant instead of timestamping
        # the sample later when it happens to be transmitted over TCP.
        rockTimestamp = rockMonotonicTs + (time.time() - time.monotonic())

        try:
            # Sends Sequency and Voltage Data to the PC - Sequence Ensures No Data Loss
            sendJson(
                conn,
                {
                    KEY_TYPE: TYPE_PIR,
                    KEY_SEQ: seq,
                    KEY_ROCK_TIMESTAMP: float(rockTimestamp),
                    KEY_VOLTAGE: float(voltage),
                },
                # Makes Sure Data Is Finished Sending Before The Next Data Is Sent
                sendLock,
            )
        except OSError:
            stopEvent.set()
            break
        seq += 1

# Streams The Thermopile Data To The PC
#def thermopileStreamLoop(conn, stopEvent, sendLock):
    #seq = 0
    #while not stopEvent.is_set():
    #    rawFrame = thermopile.getFrame()
        #rawFrame = [0 for i in range(768)]
    #    try:
    #        sendJson(
    #            conn,
    #            {
    #                KEY_TYPE: TYPE_THERMAL,
    #                KEY_SEQ: seq,
    #                KEY_FRAME: [float(value) for value in rawFrame],
    #            },
    #            sendLock,
    #        )
    #    except OSError:
    #        stopEvent.set()
    #        break
    #    seq += 1

# Streams The Thermopile Data To The PC - TODO: REMOVE THIS DUMMY FUNCTION AND REPLACE WITH REAL THERMOPILE STREAMING
def thermopileStreamLoop(conn, stopEvent, sendLock):
    seq = 0
    period = 1.0 / THERMAL_RATE_HZ

    while not stopEvent.is_set():
        startTime = time.monotonic()

        # Timestamp the measurement on the ROCK before the frame is sent.
        # With the ROCK system clock synced to the PC, this timestamp is on the
        # same wall-clock timeline as the webcam timestamps recorded on the PC.
        rockTimestamp = time.time()

        # Dummy 32x24 thermopile frame
        rawFrame = [0.0] * THERMAL_FRAME_SIZE

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

        elapsed = time.monotonic() - startTime
        remaining = period - elapsed

        if remaining > 0:
            stopEvent.wait(remaining)

# Blocks until a JSON line {"cmd": expected} is received on conn.
# Returns False if the connection closes first.
def waitForCommand(conn, expected):
    buffer = ""
    while True:
        chunk = conn.recv(1024)
        if not chunk:
            return False
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            msg = msg.strip()
            if not msg:
                continue
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if payload.get("cmd") == expected:
                return True


def main():
    parser = argparse.ArgumentParser(description="Rock Agent")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print("Warming up thermopile")
    #thermopile.warmUp()

    print("Starting PIR Acquisition")
    acq_queue = queue.Queue(maxsize=4096)

    acquisition = PIRAcquisition(acq_queue)
    processor = PIRProcessor(acq_queue)

    acquisition.start()
    processor.start()

    print("Warming Up PIR")
    while not processor.initialised:
        time.sleep(0.2)
    print("Both sensors warmed up")

    # Rocks Port Opens For Connection From PC
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"Listening on port {args.port} for the PC to connect...")

    try:
        while True:
            # PC Connected, Waiting For Ready Status
            conn, addr = server.accept()
            sendLock = threading.Lock()
            print(f"PC connected from {addr}. Sending ready status...")
            with conn:
                sendJson(conn, buildCalibrationPayload(processor, STATUS_READY), sendLock)
                if not waitForCommand(conn, CMD_START):
                    continue

                print("Collection started.")
                drainQueue(processor.outputQueue)
                stopEvent = threading.Event()

                sendJson(conn, buildCalibrationPayload(processor, STATUS_RECORDING), sendLock)

                pirThread = threading.Thread(
                    target=pirStreamLoop,
                    args=(conn, processor, stopEvent, sendLock),
                    daemon=True,
                )
                thermalThread = threading.Thread(
                    target=thermopileStreamLoop,
                    args=(conn, stopEvent, sendLock),
                    daemon=True,
                )
                pirThread.start()
                thermalThread.start()

                waitForCommand(conn, CMD_STOP)
                stopEvent.set()
                pirThread.join(timeout=1.0)
                thermalThread.join(timeout=1.0)

                try:
                    sendJson(conn, {KEY_STATUS: STATUS_STOPPED}, sendLock)
                except OSError:
                    pass
                print("Collection stopped. Waiting for the next connection...")
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop()
        acquisition.stop()
        server.close()
        print("\nAgent shut down.")


if __name__ == "__main__":
    main()
