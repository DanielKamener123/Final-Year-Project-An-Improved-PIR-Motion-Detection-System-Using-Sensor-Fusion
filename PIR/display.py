"""
pir_display.py

Live diagnostic display for bench-testing the PIR interface + processing
pipeline. Not part of the deployed system -- for verifying wiring, signal
levels, threshold behaviour, and feature vectors during development.

Shows:
  - Scrolling raw waveform with the adaptive noise mean (Am) and detection
    thresholds (Ap, An) overlaid.
  - Latest Haar-Transform bin energies as a bar chart.
  - Console printout of each completed passage's feature vector.
  - Diagnostics: achieved sample rate, and warnings if the signal is
    clipping near the 0 V / 3.3 V rails.

Acquisition produces one sample stream; a tee thread duplicates it into two
independent queues (one for the real PIRProcessor, one for this display)
so plotting never steals samples the processor needs.
"""

import queue
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from interface import PIRInterfaceConfig, PIRAcquisition
from processing import PIRProcessingConfig, PIRProcessor

# --- signal rail limits for the 0-3.3 V PIR frontend output ---
RAIL_LOW_V = 0.05
RAIL_HIGH_V = 3.25

# --- how much history to show on the scrolling plot ---
DISPLAY_WINDOW_S = 15.0


def tee(source_q: queue.Queue, dest_a: queue.Queue, dest_b: queue.Queue, stop_event):
    """Copies every item from source_q into both dest_a and dest_b."""
    while not stop_event.is_set():
        try:
            item = source_q.get(timeout=0.5)
        except queue.Empty:
            continue
        dest_a.put(item)
        dest_b.put(item)


def main():
    interface_cfg = PIRInterfaceConfig()
    processing_cfg = PIRProcessingConfig()

    acq_queue = queue.Queue(maxsize=4096)
    processor_queue = queue.Queue(maxsize=4096)
    display_queue = queue.Queue(maxsize=4096)

    acquisition = PIRAcquisition(interface_cfg, acq_queue)
    processor = PIRProcessor(processing_cfg, processor_queue)

    tee_stop = threading.Event()
    tee_thread = threading.Thread(
        target=tee, args=(acq_queue, processor_queue, display_queue, tee_stop),
        daemon=True,
    )

    acquisition.start()
    tee_thread.start()
    processor.start()

    print(f"Requested {interface_cfg.sample_rate_hz} Hz -> "
          f"ADS1115 set to {acquisition.actual_adc_rate_hz} SPS")
    print(f"I2C bus {interface_cfg.i2c_bus_number}, address "
          f"0x{interface_cfg.i2c_address:02X}")
    print("Feature vector layout:")
    print("[A1, T1, A2, T2, A3, T3, D, direction, E0, E1, E2, E3, E4, E5, E6, E7]\n")

    # --- rolling buffers for plotting ---
    maxlen = int(DISPLAY_WINDOW_S * interface_cfg.sample_rate_hz * 1.5) + 10
    t_buf = deque(maxlen=maxlen)
    v_buf = deque(maxlen=maxlen)
    passage_count = 0

    # --- rate/clipping diagnostics ---
    recent_sample_times = deque(maxlen=200)
    clip_warned = False
    t_start = time.monotonic()

    fig, (ax_wave, ax_haar) = plt.subplots(2, 1, figsize=(10, 7))

    wave_line, = ax_wave.plot([], [], lw=1, color="tab:blue", label="PIR signal")
    am_line, = ax_wave.plot([], [], lw=1, color="gray", linestyle="--", label="Am")
    ap_line, = ax_wave.plot([], [], lw=1, color="tab:green", linestyle=":", label="Ap")
    an_line, = ax_wave.plot([], [], lw=1, color="tab:red", linestyle=":", label="An")
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
    ax_haar.set_title("Latest Haar-bin energy vector")

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

        while not processor.feature_queue.empty():
            vec = processor.feature_queue.get()
            passage_count += 1
            direction = int(vec[7])
            duration = vec[6]
            np.set_printoptions(precision=4, suppress=True)
            dir_str = {1: "positive-first", -1: "negative-first", 0: "unknown"}[direction]
            print(f"[passage {passage_count}] duration={duration:.3f}s direction={dir_str}")
            print(f"  vector = {vec}")

        if t_buf:
            t_arr = np.array(t_buf)
            v_arr = np.array(v_buf)
            t0 = t_arr[-1] - DISPLAY_WINDOW_S
            mask = t_arr >= t0
            if mask.any():
                rel_t = t_arr[mask] - t_arr[mask][0]
                wave_line.set_data(rel_t, v_arr[mask])

            Am = processor.segmenter.Am
            delta = processing_cfg.threshold_delta_v
            xs = [0, DISPLAY_WINDOW_S]
            am_line.set_data(xs, [Am, Am])
            ap_line.set_data(xs, [Am + delta, Am + delta])
            an_line.set_data(xs, [Am - delta, Am - delta])

            if (v_arr[-50:] < RAIL_LOW_V).any() or (v_arr[-50:] > RAIL_HIGH_V).any():
                clip_warned = True

        if processor.haar is not None and processor.haar.latest is not None:
            _, e_bins = processor.haar.latest
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
            f"requested rate : {interface_cfg.sample_rate_hz:6.2f} Hz\n"
            f"achieved rate  : {achieved_hz:6.2f} Hz\n"
            f"Am (noise mean): {processor.segmenter.Am:6.3f} V\n"
            f"signal range   : {clip_flag}\n"
            f"passages seen  : {passage_count}\n"
            f"uptime         : {time.monotonic() - t_start:6.1f} s"
        )

        return wave_line, am_line, ap_line, an_line, status_text, *haar_bars

    ani = animation.FuncAnimation(fig, update, interval=200, blit=False)
    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop()
        acquisition.stop()
        tee_stop.set()


if __name__ == "__main__":
    main()