import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from Thermopile.interface import getFrame, getCalibrationFrames
from Thermopile.processing import ThermopilePipeline

WIDTH = 32
HEIGHT = 24

# Colour Scale
DISPLAY_MIN_TEMP = 15
DISPLAY_MAX_TEMP = 40

# Fixed Color Scale For Z Score 
Z_DISPLAY_MAX = 10.0

# Background Calibration
N_WARMUP_FRAMES = 20

pipeline = ThermopilePipeline()

# Captures frames to use as initialised background model
def warmUp():
    print(f"Warm up Starting - Make Sure No One Is In Frame")
    pipeline.calibrate(getCalibrationFrames(N_WARMUP_FRAMES))
    print("Warm Up Complete")

# One View At A Time, Same Stages And Keys As The main.py Sensor View (keys 1-5)
# (key, title, stage, colour map, minimum, maximum)
STAGES = [
    ("1", "Raw temperature (detections boxed)", "raw", "inferno", DISPLAY_MIN_TEMP, DISPLAY_MAX_TEMP),
    ("2", "Background model", "background", "inferno", DISPLAY_MIN_TEMP, DISPLAY_MAX_TEMP),
    ("3", "Z score", "raw_z", "viridis", 0, Z_DISPLAY_MAX),
    ("4", "Filtered Z score", "smoothed_z", "viridis", 0, Z_DISPLAY_MAX),
    ("5", "Foreground mask, blobs and tracks", "mask", "gray", 0, 1),
]

HELP_TEXT = (
    "Keys:  [1-5] view   [p] pause/resume   "
    "[w] re-warm background   [s] save frame snapshot   [q] quit"
)

# Converts 1D Array To 2D Array
def flat_to_grid(flat_frame):
    return [flat_frame[row * WIDTH:(row + 1) * WIDTH] for row in range(HEIGHT)]

class DebugDashboard:
    def __init__(self):
        self.paused = False
        self.stage = 0
        self.frame_count = 0
        self._last_time = time.time()
        self.fps = 0.0
        self.grid = [[0] * WIDTH for _ in range(HEIGHT)]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Thermopile Debug Dashboard")
        self.image = self.ax.imshow(self.grid, cmap="inferno", vmin=DISPLAY_MIN_TEMP, vmax=DISPLAY_MAX_TEMP)
        self.colourbar = self.fig.colorbar(self.image, ax=self.ax, fraction=0.035, pad=0.02)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.stats_text = self.fig.text(0.01, 0.01, "", fontsize=9, family="monospace")
        self.fig.suptitle(HELP_TEXT, fontsize=9)

        # Overlay artists (bounding boxes / labels)
        self._overlay_artists = []
        self._apply_stage()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # --- interaction ---------------------------------------------------

    def _on_key(self, event):
        if event.key == "p":
            self.paused = not self.paused
        elif event.key == "w":
            print("Re-running warm-up - make sure no one is in frame...")
            warmUp()
        elif event.key == "s":
            self._save_snapshot()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)
        else:
            for index, stage in enumerate(STAGES):
                if event.key == stage[0]:
                    self.stage = index
                    self._apply_stage()
                    self._draw()
                    self.fig.canvas.draw_idle()
                    break

    # Colour Map, Range And Title Of The Selected View
    def _apply_stage(self):
        key, title, _, cmap, low, high = STAGES[self.stage]
        self.image.set_cmap(cmap)
        self.image.set_clim(low, high)
        self.ax.set_title(f"[{key}] {title}", fontsize=12)

    def _save_snapshot(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"thermopile_snapshot_{timestamp}.npz"
        np.savez(
            filename,
            background=pipeline.bg_model.bg,
            sigma=pipeline.bg_model.sigma,
            raw_z=pipeline.last_raw_z,
            smoothed_z=pipeline.last_smoothed_z,
            mask=pipeline.last_mask,
            threshold=pipeline.last_threshold,
            scene_temp=pipeline.last_scene_temp,
        )
        print(f"Saved snapshot to {filename}")

    # --- drawing ---------------------------------------------------------

    def _clear_overlays(self):
        for artist in self._overlay_artists:
            artist.remove()
        self._overlay_artists.clear()

    def _draw_detections(self, detections):
        for result in detections:
            min_row, min_col, max_row, max_col = result.blob.bbox
            rect = patches.Rectangle(
                (min_col - 0.5, min_row - 0.5),
                max_col - min_col + 1,
                max_row - min_row + 1,
                linewidth=1.5,
                edgecolor="cyan",
                facecolor="none",
            )
            self.ax.add_patch(rect)
            self._overlay_artists.append(rect)
            track_id = result.features.track_id
            primary = "P " if track_id == pipeline.primary_track_id else ""
            label = self.ax.text(min_col - 0.5, min_row - 1.0, f"{primary}ID {track_id}",
                                 color="cyan", fontsize=9, fontweight="bold")
            self._overlay_artists.append(label)

    # Shows The Selected Stage Of The Latest Frame
    def _draw(self):
        stage = STAGES[self.stage][2]
        if stage == "raw":
            data = self.grid
        elif stage == "background":
            data = pipeline.bg_model.bg
        elif stage == "raw_z":
            data = pipeline.last_raw_z
        elif stage == "smoothed_z":
            data = pipeline.last_smoothed_z
        else:
            data = np.asarray(pipeline.last_mask, dtype=float)
        self.image.set_data(data)
        self._clear_overlays()
        if stage in ("raw", "mask"):
            self._draw_detections(pipeline.last_detections)

    def update(self, _frame_number):
        if self.paused:
            return []

        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self.fps = 1.0 / dt
        self.frame_count += 1

        rawFrame = getFrame()
        pipeline.process(rawFrame, time.time())
        self.grid = flat_to_grid(rawFrame)
        self._draw()

        self.stats_text.set_text(
            f"Frame {self.frame_count}   FPS {self.fps:.1f}   Scene temp {pipeline.last_scene_temp:.2f} C   "
            f"Detections {len(pipeline.last_detections)}   Primary {pipeline.primary_track_id}"
        )
        return [self.image, self.stats_text] + self._overlay_artists


def main():
    print("Warming up background model - make sure no one is in frame...")
    warmUp()

    dashboard = DebugDashboard()
    print(HELP_TEXT)

    ani = animation.FuncAnimation(
        dashboard.fig,
        dashboard.update,
        interval=100,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    return ani  # keep a reference alive for interactive use


if __name__ == "__main__":
    main()
