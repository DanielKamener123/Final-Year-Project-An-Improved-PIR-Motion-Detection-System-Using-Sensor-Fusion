import board
import busio
import adafruit_mlx90640

# Config
I2C_FREQUENCY = 1000000  # 100 kHz
REFRESH_RATE = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

# I2C bus
i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = REFRESH_RATE

frame_buffer = [0.0] * 768  # 32 x 24


def getFrame():
    """
    Block until a new frame is available, then return it as a flat
    list of 768 temperatures in degrees Celsius (row-major, row 0 =
    top of the sensor's field of view).
    """
    while True:
        try:
            mlx.getFrame(frame_buffer)
            break
        except ValueError:
            # Read glitch
            continue

    return list(frame_buffer)


if __name__ == "__main__":
    # Simple standalone smoke test.
    frame = getFrame()
    print(f"Got frame with {len(frame)} pixels. min={min(frame):.2f}C max={max(frame):.2f}C")
