import queue
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIR.interface import PIRAcquisition, i2cBusNumber, i2cAddress, sampleRateHz
from PIR.processing import (
    PIRProcessor,
    threshold,
    rollingWindow,
    getPIRFeatureNames,
)

RAIL_LOW_V = 0.05
RAIL_HIGH_V = 3.25
DISPLAY_WINDOW_S = 15.0


def main():
    acq_queue = queue.Queue(maxsize=4096)

    acquisition = PIRAcquisition(acq_queue)
    processor = PIRProcessor(acq_queue)

    acquisition.start()
    processor.start()

    feature_names = getPIRFeatureNames()

    # --- rolling buffers for plotting ---
    maxlen = int(DISPLAY_WINDOW_S * sampleRateHz * 1.5) + 10
    t_buf = deque(maxlen=maxlen)
    v_buf = deque(maxlen=maxlen)
    relative_buf = deque(maxlen=maxlen)

    # --- rate/clipping diagnostics ---
    recent_sample_times = deque(maxlen=200)
    clip_warned = False
    t_start = time.monotonic()

    fig, (ax_wave, ax_relative, ax_features) = plt.subplots(3, 1, figsize=(10, 9))

    wave_line, = ax_wave.plot([], [], lw=1, label="PIR signal")
    base_line, = ax_wave.plot([], [], lw=1, linestyle="--", label="baseline")
    hi_line, = ax_wave.plot([], [], lw=1, linestyle=":", label="+threshold")
    lo_line, = ax_wave.plot([], [], lw=1, linestyle=":", label="-threshold")
    ax_wave.set_ylim(-0.2, 3.5)
    ax_wave.set_xlim(0, DISPLAY_WINDOW_S)
    ax_wave.set_xlabel("time (s)")
    ax_wave.set_ylabel("volts")
    ax_wave.legend(loc="upper right", fontsize=8)
    ax_wave.set_title("PIR raw signal")

    relative_line, = ax_relative.plot([], [], lw=1, label="relative PIR signal")
    ax_relative.axhline(0.0, lw=1, linestyle="--")
    ax_relative.axhline(threshold, lw=1, linestyle=":")
    ax_relative.axhline(-threshold, lw=1, linestyle=":")
    ax_relative.set_xlim(0, DISPLAY_WINDOW_S)
    ax_relative.set_ylim(-1.0, 1.0)
    ax_relative.set_xlabel("time (s)")
    ax_relative.set_ylabel("volts from baseline")
    ax_relative.set_title("PIR zero-centred signal")

    feature_bars = ax_features.bar(range(len(feature_names)), [0.0] * len(feature_names))
    ax_features.set_xticks(range(len(feature_names)), feature_names, rotation=15, ha="right")
    ax_features.set_ylim(0, 1)
    ax_features.set_ylabel("feature value")
    ax_features.set_title(f"Live PIR temporal features ({rollingWindow:.2f} s rolling window)")

    status_text = ax_wave.text(
        0.01, 0.95, "", transform=ax_wave.transAxes, va="top", fontsize=8,
        family="monospace",
    )

    def update(_frame):
        nonlocal clip_warned

        while True:
            try:
                ts, v, relative = processor.outputQueue.get_nowait()
            except queue.Empty:
                break
            t_buf.append(ts)
            v_buf.append(v)
            relative_buf.append(relative)
            recent_sample_times.append(ts)

        if t_buf:
            t_arr = np.array(t_buf)
            v_arr = np.array(v_buf)
            relative_arr = np.array(relative_buf)
            t0 = t_arr[-1] - DISPLAY_WINDOW_S
            mask = t_arr >= t0
            if mask.any():
                rel_t = t_arr[mask] - t_arr[mask][0]
                wave_line.set_data(rel_t, v_arr[mask])
                relative_line.set_data(rel_t, relative_arr[mask])

                relative_limit = max(float(np.max(np.abs(relative_arr[mask]))) * 1.2, threshold * 1.5, 0.1)
                ax_relative.set_ylim(-relative_limit, relative_limit)

            baseline = processor.getBaseline()
            xs = [0, DISPLAY_WINDOW_S]
            base_line.set_data(xs, [baseline, baseline])
            hi_line.set_data(xs, [baseline + threshold, baseline + threshold])
            lo_line.set_data(xs, [baseline - threshold, baseline - threshold])

            if (v_arr[-50:] < RAIL_LOW_V).any() or (v_arr[-50:] > RAIL_HIGH_V).any():
                clip_warned = True

        feature_values = np.asarray(processor.snapshot(rollingWindow), dtype=np.float64)
        top = max(float(feature_values.max()), 1e-6)
        ax_features.set_ylim(0, top * 1.2)
        for bar, value in zip(feature_bars, feature_values):
            bar.set_height(value)

        if len(recent_sample_times) >= 2:
            dt = recent_sample_times[-1] - recent_sample_times[0]
            achieved_hz = (len(recent_sample_times) - 1) / dt if dt > 0 else 0.0
        else:
            achieved_hz = 0.0

        clip_flag = "CLIPPING NEAR RAIL" if clip_warned else "ok"
        warmup_flag = "ready" if processor.initialised else "warming up"
        status_text.set_text(
            f"requested rate : {sampleRateHz:6.2f} Hz\n"
            f"achieved rate  : {achieved_hz:6.2f} Hz\n"
            f"baseline       : {processor.getBaseline():6.3f} V\n"
            f"PIR state      : {warmup_flag}\n"
            f"signal range   : {clip_flag}\n"
            f"uptime         : {time.monotonic() - t_start:6.1f} s"
        )

        return wave_line, relative_line, base_line, hi_line, lo_line, status_text, *feature_bars

    # Keep A Reference - Otherwise The Animation Is Garbage Collected And The Plot Never Updates
    ani = animation.FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
