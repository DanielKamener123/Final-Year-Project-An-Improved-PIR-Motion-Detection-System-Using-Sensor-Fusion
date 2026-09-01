import adafruit_mlx90640
import adafruit_extended_bus

# Config
I2C_BUS = 7
REFRESH_RATE = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

i2c = adafruit_extended_bus.ExtendedI2C(I2C_BUS)

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