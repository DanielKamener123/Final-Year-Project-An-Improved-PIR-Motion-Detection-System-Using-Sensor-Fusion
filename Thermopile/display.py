import time

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

# Panels, in the order they occupy the 2x3 grid. 'key' is the number
# key that focuses/unfocuses that panel to fill the whole figure.
PANEL_DEFS = [
    {"key": "1", "kind": "raw", "title": "Raw Frame And Detections)"},
    {"key": "2", "kind": "background", "title": "Background Model"},
    {"key": "3", "kind": "mask", "title": "Foreground Mask"},
    {"key": "4", "kind": "raw_z", "title": "Z Score"},
    {"key": "5", "kind": "smoothed_z", "title": "Filtered Z Score"},
    {"key": "6", "kind": "stats", "title": "Stats"},
]

HELP_TEXT = (
    "Keys:  [1-6] focus panel   [g] grid view   [p] pause/resume   "
    "[w] re-warm background   [s] save frame snapshot   [q] quit"
)

# Converts 1D Array To 2D Array
def flat_to_grid(flat_frame):
    return [flat_frame[row * WIDTH:(row + 1) * WIDTH] for row in range(HEIGHT)]

class DebugDashboard:
    def __init__(self):
        self.paused = False
        self.focused_kind = None  # None => grid view
        self.frame_count = 0
        self._last_time = time.time()
        self.fps = 0.0

        self.fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Thermopile Debug Dashboard")
        self.axes_by_kind = {}
        self.images_by_kind = {}
        self._orig_positions = {}

        flat_axes = axes.flatten()
        for ax, panel in zip(flat_axes, PANEL_DEFS):
            kind = panel["kind"]
            ax.set_title(f"[{panel['key']}] {panel['title']}", fontsize=10)
            self.axes_by_kind[kind] = ax

            if kind == "stats":
                ax.axis("off")
                self.stats_text = ax.text(
                    0.02, 0.98, "", transform=ax.transAxes,
                    fontsize=9, family="monospace",
                    verticalalignment="top",
                )
                continue

            zeros = [[0] * WIDTH for _ in range(HEIGHT)]
            if kind == "raw" or kind == "background":
                image = ax.imshow(
                    zeros,
                    cmap="inferno",
                    vmin=DISPLAY_MIN_TEMP,
                    vmax=DISPLAY_MAX_TEMP,
                )
            elif kind == "mask":
                image = ax.imshow(zeros, cmap="gray", vmin=0, vmax=1)
            else:  # raw_z, smoothed_z
                image = ax.imshow(
                    zeros,
                    cmap="viridis",
                    vmin=0,
                    vmax=Z_DISPLAY_MAX,
                )

            self.images_by_kind[kind] = image
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig.suptitle(HELP_TEXT, fontsize=9)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Remember each axis's normal grid position so focus mode can
        # restore it afterwards.
        for kind, ax in self.axes_by_kind.items():
            self._orig_positions[kind] = ax.get_position().bounds

        # Overlay artists (bounding boxes / labels) drawn on the raw panel
        self._overlay_artists = []

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # --- interaction ---------------------------------------------------

    def _on_key(self, event):
        if event.key == "p":
            self.paused = not self.paused
        elif event.key == "g":
            self.focused_kind = None
            self._apply_layout()
        elif event.key == "w":
            print("Re-running warm-up - make sure no one is in frame...")
            warmUp()
        elif event.key == "s":
            self._save_snapshot()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)
        else:
            for panel in PANEL_DEFS:
                if event.key == panel["key"]:
                    kind = panel["kind"]
                    self.focused_kind = None if self.focused_kind == kind else kind
                    self._apply_layout()
                    break

    def _apply_layout(self):
        for kind, ax in self.axes_by_kind.items():
            if self.focused_kind is None:
                ax.set_visible(True)
                ax.set_position(self._orig_positions[kind])
            elif kind == self.focused_kind:
                ax.set_visible(True)
                ax.set_position([0.05, 0.05, 0.9, 0.85])
            else:
                ax.set_visible(False)
        self.fig.canvas.draw_idle()

    def _save_snapshot(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"thermopile_snapshot_{timestamp}.npz"
        import numpy as np
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

    def _draw_detections(self, ax, detections):
        for result in detections:
            min_row, min_col, max_row, max_col = result.blob.bbox
            box_width = max_col - min_col + 1
            box_height = max_row - min_row + 1

            rect = patches.Rectangle(
                (min_col - 0.5, min_row - 0.5),
                box_width,
                box_height,
                linewidth=1.5,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)
            self._overlay_artists.append(rect)

            # Use confidence if it exists, otherwise use track ID if available.
            confidence = getattr(result, "confidence", None)
            track_id = getattr(getattr(result, "features", None), "track_id", None)

            if track_id is not None and confidence is not None:
                label_text = f"ID {track_id}  {confidence:.2f}"
            elif track_id is not None:
                label_text = f"ID {track_id}"
            elif confidence is not None:
                label_text = f"{confidence:.2f}"
            else:
                label_text = "Blob"

            label = ax.text(
                min_col - 0.5,
                min_row - 1.0,
                label_text,
                color="cyan",
                fontsize=8,
                fontweight="bold",
            )
            self._overlay_artists.append(label)

    def _update_stats_text(self, detections):
        lines = [
            f"Frame:       {self.frame_count}",
            f"FPS:         {self.fps:.1f}",
            f"Scene temp:  {pipeline.last_scene_temp:.2f} C",
            f"Threshold:   {pipeline.last_threshold:.2f}",
            f"Detections:  {len(detections)}",
            f"Primary:     {pipeline.primary_track_id}",
            "",
        ]

        for i, result in enumerate(detections):
            lines.append(f"Blob {i}:")

            confidence = getattr(result, "confidence", None)
            if confidence is not None:
                lines.append(f"  confidence={confidence:.2f}")

            blob = getattr(result, "blob", None)
            if blob is not None:
                centroid = getattr(blob, "centroid", None)
                pixel_count = getattr(blob, "pixel_count", None)
                bbox = getattr(blob, "bbox", None)

                if centroid is not None:
                    lines.append(
                        f"  centroid=({centroid[0]:.1f},{centroid[1]:.1f})"
                    )
                if pixel_count is not None:
                    lines.append(f"  pixels={pixel_count}")
                if bbox is not None:
                    lines.append(f"  bbox={bbox}")

            features = getattr(result, "features", None)
            if features is None:
                continue

            # Display whatever fields the current BlobFeatures class contains.
            track_id = getattr(features, "track_id", None)
            track_age_frames = getattr(features, "track_age_frames", None)
            mean_temperature_difference = getattr(features, "mean_temperature_difference", None)
            max_temperature_difference = getattr(features, "max_temperature_difference", None)

            if track_id is not None:
                lines.append(f"  track_id={track_id}")

            if track_age_frames is not None:
                lines.append(f"  track_age_frames={track_age_frames}")

            if mean_temperature_difference is not None:
                lines.append(
                    f"  mean_temperature_difference={mean_temperature_difference:.2f}"
                )

            if max_temperature_difference is not None:
                lines.append(
                    f"  max_temperature_difference={max_temperature_difference:.2f}"
                )

            lines.append("")

        self.stats_text.set_text("\n".join(lines))

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
        grid = flat_to_grid(rawFrame)
        detections = pipeline.last_detections

        if "raw" in self.images_by_kind:
            self.images_by_kind["raw"].set_data(grid)
        if "background" in self.images_by_kind:
            self.images_by_kind["background"].set_data(pipeline.bg_model.bg)
        if "mask" in self.images_by_kind:
            self.images_by_kind["mask"].set_data(pipeline.last_mask)
        if "raw_z" in self.images_by_kind:
            self.images_by_kind["raw_z"].set_data(pipeline.last_raw_z)
        if "smoothed_z" in self.images_by_kind:
            self.images_by_kind["smoothed_z"].set_data(pipeline.last_smoothed_z)

        self._clear_overlays()
        if "raw" in self.axes_by_kind:
            self._draw_detections(self.axes_by_kind["raw"], detections)

        self._update_stats_text(detections)

        artists = (
            list(self.images_by_kind.values())
            + self._overlay_artists
            + [self.stats_text]
        )
        return artists


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
