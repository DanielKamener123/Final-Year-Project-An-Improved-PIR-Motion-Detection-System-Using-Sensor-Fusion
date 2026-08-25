import matplotlib.pyplot as plt
import matplotlib.animation as animation

from processing import getFrame

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


def update(_frame_number):
    grid = flat_to_grid(getFrame())
    image.set_data(grid)
    return [image]


ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
plt.show()
