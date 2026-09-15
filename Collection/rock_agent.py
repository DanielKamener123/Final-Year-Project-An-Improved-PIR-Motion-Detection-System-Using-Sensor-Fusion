"""
Data-collection agent -- run this on the Rock 4C+.

Runs the PIR and thermopile pipelines together, waits for a start
command from Collection/pc_collector.py (run on the PC) over TCP, then
streams one combined feature vector per thermopile frame (~4 Hz) until
told to stop.

Both halves of each vector represent the signal as it currently is:
the thermopile side is this frame's detection, and the PIR side is a
sliding-window Haar feature recomputed fresh every tick from the last
~128 raw samples (see PIR/sliding_window.py) -- neither one is a value
held over from an earlier moment.

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

from PIR.interface import PIRAcquisition
from PIR.processing import PIRProcessor
from PIR.sliding_window import PIRSlidingWindow
import Thermopile.processing as thermopile

from Collection.protocol import (
    CMD_START, CMD_STOP, KEY_SEQ, KEY_PIR,
    KEY_THERMAL, KEY_THERMAL_DETECTED, DEFAULT_PORT,
    THERMAL_VECTOR_LENGTH, KEY_STATUS, STATUS_READY,
)


# Flattens a Thermopile.thermopile_pipeline.BlobFeatures instance into a
# plain list of floats, in the order Collection/protocol.py declares.
def flattenBlobFeatures(features):
    return [
        float(features.track_id),
        float(features.centroid[0]),
        float(features.centroid[1]),
        float(features.pixel_count),
        float(features.width),
        float(features.height),
        float(features.row_displacement),
        float(features.col_displacement),
        float(features.displacement_px),
        float(features.normalised_displacement),
        float(features.track_age_frames),
        float(features.cumulative_displacement_px),
        float(features.mean_temperature_difference),
        float(features.max_temperature_difference),
        float(features.virtual_position_x),
        float(features.horizontal_pixels_traversed),
        float(features.vertical_pixels_traversed),
        float(features.horizontal_pixels_covered),
        float(features.vertical_pixels_covered),
        float(features.horizontal_direction),
        float(features.vertical_direction),
    ]


# Picks the most prominent detection this frame (largest blob) as "the"
# thermopile reading. Multiple simultaneous people are not merged --
# that is out of scope here, flagging it rather than quietly picking one.
def buildThermalRow():
    detections = thermopile.getDetections()
    if not detections:
        return [0.0] * THERMAL_VECTOR_LENGTH, False
    best = max(detections, key=lambda d: d.blob.pixel_count)
    return flattenBlobFeatures(best.features), True


# Copies every item from source_q into both dest_a and dest_b, so the
# event-based PIRProcessor and the continuous PIRSlidingWindow can each
# see every raw sample without stealing it from one another.
def tee(source_q, dest_a, dest_b, stop_event):
    while not stop_event.is_set():
        try:
            item = source_q.get(timeout=0.5)
        except queue.Empty:
            continue
        dest_a.put(item)
        dest_b.put(item)


def collectionLoop(conn, slidingWindow, stopEvent):
    seq = 0
    while not stopEvent.is_set():
        thermopile.getFrame()  # blocks until the next ~4 Hz frame is ready
        thermal_vec, detected = buildThermalRow()
        pir_vec = slidingWindow.snapshot()

        line = json.dumps({
            KEY_SEQ: seq,
            KEY_PIR: pir_vec,
            KEY_THERMAL: thermal_vec,
            KEY_THERMAL_DETECTED: detected,
        }) + "\n"

        try:
            conn.sendall(line.encode("utf-8"))
        except OSError:
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
    processor_queue = queue.Queue(maxsize=4096)
    window_queue = queue.Queue(maxsize=4096)

    acquisition = PIRAcquisition(acq_queue)
    processor = PIRProcessor(processor_queue)
    slidingWindow = PIRSlidingWindow(window_queue, processor.segmenter)

    tee_stop = threading.Event()
    tee_thread = threading.Thread(
        target=tee, args=(acq_queue, processor_queue, window_queue, tee_stop),
        daemon=True,
    )

    acquisition.start()
    tee_thread.start()
    processor.start()
    slidingWindow.start()

    print("Waiting for PIR baseline calibration (~10 s)...")
    while not processor.segmenter.initialised:
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
            print(f"PC connected from {addr}. Sending ready status...")
            with conn:
                conn.sendall((json.dumps({KEY_STATUS: STATUS_READY}) + "\n").encode("utf-8"))
                if not waitForCommand(conn, CMD_START):
                    continue
                print("Collection started.")
                stopEvent = threading.Event()

                def watchForStop():
                    waitForCommand(conn, CMD_STOP)
                    stopEvent.set()

                stopWatcher = threading.Thread(target=watchForStop, daemon=True)
                stopWatcher.start()

                collectionLoop(conn, slidingWindow, stopEvent)
                print("Collection stopped. Waiting for the next connection...")
    except KeyboardInterrupt:
        pass
    finally:
        tee_stop.set()
        slidingWindow.stop()
        processor.stop()
        acquisition.stop()
        server.close()
        print("\nAgent shut down.")


if __name__ == "__main__":
    main()
