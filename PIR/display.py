"""
pir_display.py

Live diagnostic display for bench-testing the PIR interface + processing
pipeline. Not part of the deployed system -- for verifying wiring, signal
levels, threshold behaviour, and feature vectors during development.

Shows:
  - Scrolling raw waveform with the background baseline and detection
    thresholds overlaid.
  - Live Haar-transform bin energies from the sliding window -- these
    are recomputed on every plot update, so the bars always reflect the
    signal right now, not just the most recently completed passage.
  - Console printout of each completed passage's feature vector, plus
    its duration and direction.
  - Diagnostics: achieved sample rate, and warnings if the signal is
    clipping near the 0 V / 3.3 V rails.

Acquisition produces one sample stream; a tee thread duplicates it into
three independent queues (the real PIRProcessor, the live sliding
window, and this display) so plotting never steals samples the others
need.
"""

import queue
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from interface import PIRAcquisition, i2cBusNumber, i2cAddress, sampleRateHz
from processing import PIRProcessor, threshold
from sliding_window import PIRSlidingWindow

# --- signal rail limits for the 0-3.3 V PIR frontend output ---
RAIL_LOW_V = 0.05
RAIL_HIGH_V = 3.25

# --- how much history to show on the scrolling plot ---
DISPLAY_WINDOW_S = 15.0


def tee(source_q: queue.Queue, dest_queues, stop_event):
    """Copies every item from source_q into every queue in dest_queues."""
    while not stop_event.is_set():
        try:
            item = source_q.get(timeout=0.5)
        except queue.Empty:
            continue
        for dest in dest_queues:
            dest.put(item)


def main():
    acq_queue = queue.Queue(maxsize=4096)
    processor_queue = queue.Queue(maxsize=4096)
    window_queue = queue.Queue(maxsize=4096)
    display_queue = queue.Queue(maxsize=4096)

    acquisition = PIRAcquisition(acq_queue)
    processor = PIRProcessor(processor_queue)
    slidingWindow = PIRSlidingWindow(window_queue, processor.segmenter)

    tee_stop = threading.Event()
    tee_thread = threading.Thread(
        target=tee,
        args=(acq_queue, [processor_queue, window_queue, display_queue], tee_stop),
        daemon=True,
    )

    acquisition.start()
    tee_thread.start()
    processor.start()
    slidingWindow.start()

    print(f"Requested {sampleRateHz} Hz -> ADS1115 set to {acquisition.actualADCRateHz} SPS")
    print(f"I2C bus {i2cBusNumber}, address 0x{i2cAddress:02X}")
    print("Live Haar bars: 8 frequency-band energies, recomputed every update")
    print("Console printout: 8-value Haar vector for each completed passage\n")

    # --- rolling buffers for plotting ---
    maxlen = int(DISPLAY_WINDOW_S * sampleRateHz * 1.5) + 10
    t_buf = deque(maxlen=maxlen)
    v_buf = deque(maxlen=maxlen)
    passage_count = 0

    # --- rate/clipping diagnostics ---
    recent_sample_times = deque(maxlen=200)
    clip_warned = False
    t_start = time.monotonic()

    fig, (ax_wave, ax_haar) = plt.subplots(2, 1, figsize=(10, 7))

    wave_line, = ax_wave.plot([], [], lw=1, color="tab:blue", label="PIR signal")
    base_line, = ax_wave.plot([], [], lw=1, color="gray", linestyle="--", label="baseline")
    hi_line, = ax_wave.plot([], [], lw=1, color="tab:green", linestyle=":", label="+threshold")
    lo_line, = ax_wave.plot([], [], lw=1, color="tab:red", linestyle=":", label="-threshold")
    ax_wave.set_ylim(-0.2, 3.5)
    ax_wave.set_xlim(0, DISPLAY_WINDOW_S)
    ax_wave.set_xlabel("time (s)")
    ax_wave.set_ylabel("volts")
    ax_wave.legend(loc="upper right", fontsize=8)
    ax_wave.set_title("PIR raw signal")

    haar_bars = ax_haar.bar(range(8), [0] * 8, color="tab:purple")
    ax_haar.set_ylim(0, 1)
    ax_haar.set_xlabel("Haar frequency bin")
    ax_haar.set_ylabel("energy")
    ax_haar.set_title("Live Haar-bin energy vector (sliding window)")

    status_text = ax_wave.text(
        0.01, 0.95, "", transform=ax_wave.transAxes, va="top", fontsize=8,
        family="monospace",
    )

    def update(_frame):
        nonlocal clip_warned, passage_count

        while True:
            try:
                ts, v = display_queue.get_nowait()
            except queue.Empty:
                break
            t_buf.append(ts)
            v_buf.append(v)
            recent_sample_times.append(ts)

        while not processor.movementQueue.empty():
            metadata = processor.movementQueue.get()
            passage_count += 1
            duration = metadata["end_ts"] - metadata["start_ts"]
            direction = metadata["direction"]
            np.set_printoptions(precision=4, suppress=True)
            print(f"[passage {passage_count}] duration={duration:.3f}s direction={direction}")
            print(f"  vector = {metadata['features']}")

        if t_buf:
            t_arr = np.array(t_buf)
            v_arr = np.array(v_buf)
            t0 = t_arr[-1] - DISPLAY_WINDOW_S
            mask = t_arr >= t0
            if mask.any():
                rel_t = t_arr[mask] - t_arr[mask][0]
                wave_line.set_data(rel_t, v_arr[mask])

            baseline = processor.segmenter.baselineVoltage
            xs = [0, DISPLAY_WINDOW_S]
            base_line.set_data(xs, [baseline, baseline])
            hi_line.set_data(xs, [baseline + threshold, baseline + threshold])
            lo_line.set_data(xs, [baseline - threshold, baseline - threshold])

            if (v_arr[-50:] < RAIL_LOW_V).any() or (v_arr[-50:] > RAIL_HIGH_V).any():
                clip_warned = True

        e_bins = np.array(slidingWindow.snapshot())
        top = max(e_bins.max(), 1e-6)
        ax_haar.set_ylim(0, top * 1.2)
        for bar, e in zip(haar_bars, e_bins):
            bar.set_height(e)

        if len(recent_sample_times) >= 2:
            dt = recent_sample_times[-1] - recent_sample_times[0]
            achieved_hz = (len(recent_sample_times) - 1) / dt if dt > 0 else 0.0
        else:
            achieved_hz = 0.0

        clip_flag = "CLIPPING NEAR RAIL" if clip_warned else "ok"
        status_text.set_text(
            f"requested rate : {sampleRateHz:6.2f} Hz\n"
            f"achieved rate  : {achieved_hz:6.2f} Hz\n"
            f"baseline       : {processor.segmenter.baselineVoltage:6.3f} V\n"
            f"signal range   : {clip_flag}\n"
            f"passages seen  : {passage_count}\n"
            f"uptime         : {time.monotonic() - t_start:6.1f} s"
        )

        return wave_line, base_line, hi_line, lo_line, status_text, *haar_bars

    ani = animation.FuncAnimation(fig, update, interval=200, blit=False)
    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop()
        acquisition.stop()
        slidingWindow.stop()
        tee_stop.set()


if __name__ == "__main__":
    main()
