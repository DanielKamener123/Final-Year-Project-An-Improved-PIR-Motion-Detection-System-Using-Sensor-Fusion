"""
Data-collection agent -- run this on the Rock 4C+.

Runs the PIR and thermopile pipelines together, waits for a start
command from Collection/pc_collector.py (run on the PC) over TCP, then
streams raw PIR samples continuously and one raw thermopile frame per
thermopile update until told to stop.

The PC does the timestamping, not this script -- the two machines'
clocks are not assumed to be in sync.

Usage:
    python -m Collection.rock_agent --port 5050
"""

import argparse
import json
import queue
import socket
import threading
import time

from PIR.interface import PIRAcquisition, sampleRateHz
from PIR.processing import PIRProcessor
import Thermopile.processing as thermopile

from Collection.protocol import (
    CMD_START,
    CMD_STOP,
    DEFAULT_PORT,
    KEY_FRAME,
    KEY_PIR_BASELINE,
    KEY_PIR_SAMPLE_RATE,
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

THERMAL_RATE_HZ = 4.0


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


def buildCalibrationPayload(processor, status):
    return {
        KEY_STATUS: status,
        KEY_PIR_BASELINE: processor.getBaseline(),
        KEY_PIR_SAMPLE_RATE: float(sampleRateHz),
        KEY_THERMAL_RATE: THERMAL_RATE_HZ,
        KEY_THERMAL_FRAME_SHAPE: list(thermopile.FRAME_SHAPE),
        KEY_THERMAL_BACKGROUND: thermopile.getBackground().reshape(-1).astype(float).tolist(),
        KEY_THERMAL_SIGMA: thermopile.getSigma().reshape(-1).astype(float).tolist(),
    }


def pirStreamLoop(conn, processor, stopEvent, sendLock):
    seq = 0
    while not stopEvent.is_set():
        try:
            _rock_ts, voltage, _relative = processor.outputQueue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            sendJson(
                conn,
                {
                    KEY_TYPE: TYPE_PIR,
                    KEY_SEQ: seq,
                    KEY_VOLTAGE: float(voltage),
                },
                sendLock,
            )
        except OSError:
            stopEvent.set()
            break
        seq += 1


def thermopileStreamLoop(conn, stopEvent, sendLock):
    seq = 0
    while not stopEvent.is_set():
        rawFrame = thermopile.getFrame()
        try:
            sendJson(
                conn,
                {
                    KEY_TYPE: TYPE_THERMAL,
                    KEY_SEQ: seq,
                    KEY_FRAME: [float(value) for value in rawFrame],
                },
                sendLock,
            )
        except OSError:
            stopEvent.set()
            break
        seq += 1


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
    parser = argparse.ArgumentParser(description="Rock 4C+ data-collection agent")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print("Warming up thermopile background model - keep the scene empty...")
    thermopile.warmUp()

    print("Starting PIR acquisition (self-warms over its own ~10 s window)...")
    acq_queue = queue.Queue(maxsize=4096)

    acquisition = PIRAcquisition(acq_queue)
    processor = PIRProcessor(acq_queue)

    acquisition.start()
    processor.start()

    print("Waiting for PIR baseline calibration (~10 s)...")
    while not processor.initialised:
        time.sleep(0.2)
    print("Both sensors warmed up.")

    # The port only opens now -- a PC trying to connect any earlier
    # simply has nothing to connect to, rather than being accepted and
    # fed data from sensors that are still calibrating.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"Listening on port {args.port} for the PC to connect...")

    try:
        while True:
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
