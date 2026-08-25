import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from processing import getFrame, getDetections

WIDTH = 32
HEIGHT = 24

# Fixed color scale range in degrees C -- adjust to suit your
# expected background/target temperature range.
DISPLAY_MIN_TEMP = 15
DISPLAY_MAX_TEMP = 40


def flat_to_grid(flat_frame):
    """Convert a flat 768-element list into a 24x32 list of lists."""
    return [flat_frame[row * WIDTH:(row + 1) * WIDTH] for row in range(HEIGHT)]


fig, ax = plt.subplots()
initial_grid = flat_to_grid(getFrame())
image = ax.imshow(initial_grid, cmap="inferno", vmin=DISPLAY_MIN_TEMP, vmax=DISPLAY_MAX_TEMP)
fig.colorbar(image, ax=ax, label="Temperature (C)")
ax.set_title("Live Thermopile Frame")

# Bounding boxes / confidence labels are recreated every frame since
# the number of detected blobs changes frame to frame.
_overlay_artists = []


def _clear_overlays():
    for artist in _overlay_artists:
        artist.remove()
    _overlay_artists.clear()


def _draw_detections(detections):
    for result in detections:
        min_row, min_col, max_row, max_col = result.blob.bbox
        box_width = max_col - min_col + 1
        box_height = max_row - min_row + 1

        rect = patches.Rectangle(
            (min_col - 0.5, min_row - 0.5), box_width, box_height,
            linewidth=1.5, edgecolor="cyan", facecolor="none",
        )
        ax.add_patch(rect)
        _overlay_artists.append(rect)

        label = ax.text(
            min_col - 0.5, min_row - 1.0,
            f"{result.confidence:.2f}",
            color="cyan", fontsize=8, fontweight="bold",
        )
        _overlay_artists.append(label)


def update(_frame_number):
    grid = flat_to_grid(getFrame())
    image.set_data(grid)

    _clear_overlays()
    _draw_detections(getDetections())

    return [image, *_overlay_artists]


ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
plt.show()
