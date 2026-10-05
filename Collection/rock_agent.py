import json
import queue
import socket
import threading
import time

from PIR.interface import PIRAcquisition, sampleRateHz
from PIR.processing import PIRProcessor
import Thermopile.processing as thermopile
from Thermopile.interface import REFRESH_RATE_HZ
from Collection.protocol import (
    CMD_START_AT,
    CMD_STOP,
    CMD_SYNC,
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

THERMAL_RATE_HZ = REFRESH_RATE_HZ

# Converts Dictionary to JSON and Sends
def sendJson(conn, payload, sendLock):
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    # Makes Sure Only One Thread Sends At A Time
    with sendLock:
        conn.sendall(line.encode("utf-8"))

# Removes All PIR Samples From Queue To Start Fresh Recording
def drainQueue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def calibrationData(processor, status):
    payload = {
        KEY_STATUS: status,
        KEY_ROCK_TIME: time.time(),
        KEY_PIR_BASELINE: processor.getBaseline(),
        KEY_PIR_SAMPLE_RATE: float(sampleRateHz),
        KEY_THERMAL_RATE: THERMAL_RATE_HZ,
    }

    payload.update(
        {
            KEY_THERMAL_FRAME_SHAPE: list(thermopile.FRAME_SHAPE),
            KEY_THERMAL_BACKGROUND: thermopile.getBackground().reshape(-1).astype(float).tolist(),
            KEY_THERMAL_SIGMA: thermopile.getSigma().reshape(-1).astype(float).tolist(),
        }
    )
    return payload

# Send PIR Data
def pirStreamLoop(conn, processor, stopEvent, sendLock):
    seqIndex = 0

    # While Stop Condition Is Not Set
    while not stopEvent.is_set():
        try:
            # Get Timestamp and Data From PIR Queue
            rockTimestamp, voltage = processor.outputQueue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # Send PIR Data To PC
            sendJson(
                conn,
                {
                    KEY_TYPE: TYPE_PIR,
                    KEY_SEQ: seqIndex,
                    KEY_ROCK_TIMESTAMP: float(rockTimestamp),
                    KEY_VOLTAGE: float(voltage),
                },
                sendLock,
            )

        # Stop Sending Frames If The PC Disconnects Or If Error Occurs
        except OSError:
            stopEvent.set()
            break
        seqIndex += 1

def thermopileStreamLoop(conn, stopEvent, sendLock):
    seqIndex = 0

    # Keep Sending Frames Until Stop Condition Is Set
    while not stopEvent.is_set():
        # Get Raw Frame Data
        rawFrame = thermopile.getRawFrame()

        # Timestamp immediately after the complete frame has been read.
        rockTimestamp = time.time()

        try:
            # Send Thermopile Data To PC
            sendJson(
                conn,
                {
                    KEY_TYPE: TYPE_THERMAL,
                    KEY_SEQ: seqIndex,
                    KEY_ROCK_TIMESTAMP: rockTimestamp,
                    KEY_FRAME: [float(value) for value in rawFrame],
                },
                sendLock,
            )

        # Stop Sending Frames If The PC Disconnects Or If Error Occurs
        except OSError:
            stopEvent.set()
            break
        seqIndex += 1


# Wait For Start At Or Stop Command From PC
def waitForCommand(conn, expected, sendLock):
    # TCP Stream Buffer
    buffer = ""

    while True:
        try:
            # Read Incoming Bytes
            chunk = conn.recv(1024)
        # Stop If The PC Disconnects Or If Error Occurs
        except OSError:
            return None
        if not chunk:
            return None

        # Add Chunk To Buffer
        buffer += chunk.decode("utf-8")

        # Go Through Each Line In Buffer
        while "\n" in buffer:
            # Split And Strip Buffer
            message, buffer = buffer.split("\n", 1)
            message = message.strip()
            if not message:
                continue

            try:
                dataRecieved = json.loads(message)
            except json.JSONDecodeError:
                continue

            # Look For Command Type
            command = dataRecieved.get("cmd")

            # If The PC Sends SYNC Command, Reply With SYNC Response
            if command == CMD_SYNC:
                sendJson(
                    conn,
                    {
                        KEY_TYPE: TYPE_SYNC,
                        KEY_PC_SEND_TIME: dataRecieved.get(KEY_PC_SEND_TIME),
                        KEY_ROCK_TIME: time.time(),
                    },
                    sendLock,
                )
                continue

            # Otherwise Return The Data If The Command Matches The Expected Command
            if command == expected:
                return dataRecieved

# Wait Until The Rock Time Reaches The Target Time To Start Capturing Data
def waitUntilRockTime(targetTime):
    while True:
        remaining = targetTime - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.02))

def main():
    print("Warming up thermopile")
    thermopile.warmUp()


    print("Starting PIR")
    acquisitionQueue = queue.Queue(maxsize=4096)
    acquisition = PIRAcquisition(acquisitionQueue)
    processor = PIRProcessor(acquisitionQueue)
    acquisition.start()
    processor.start()

    print("Warming up PIR")
    while not processor.initialised:
        time.sleep(0.2)
    print("Sensors ready")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 5050))
    server.listen(1)
    print(f"Listening for PC...")

    try:
        while True:
            conn, _ = server.accept()
            sendLock = threading.Lock()
            print(f"PC connected")

            with conn:
                sendJson(conn, calibrationData(processor, STATUS_READY), sendLock)

                startPayload = waitForCommand(conn, CMD_START_AT, sendLock)
                if startPayload is None:
                    continue

                rockStartTime = float(startPayload[KEY_ROCK_START_TIME])
                print("Start scheduled")
                waitUntilRockTime(rockStartTime)

                drainQueue(processor.outputQueue)
                stopEvent = threading.Event()
                sendJson(conn, calibrationData(processor, STATUS_RECORDING), sendLock)

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
