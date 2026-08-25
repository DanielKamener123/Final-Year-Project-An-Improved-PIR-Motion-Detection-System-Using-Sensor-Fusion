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

# Reads Frame From Thermopile
def getFrame():
    while True:
        try:
            mlx.getFrame(frame_buffer)
            break
        except ValueError:
            # Read glitch
            continue

    return list(frame_buffer)


if __name__ == "__main__":
    # Tests To Ensure Frame Is Being Read
    frame = getFrame()
    print(frame)
