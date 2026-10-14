import adafruit_mlx90640
import adafruit_extended_bus

# Config
I2C_BUS = 7
I2C_FREQUENCY = 400_000
REFRESH_RATE_HZ = 8.0

REFRESH_RATES = {
    0.5: adafruit_mlx90640.RefreshRate.REFRESH_0_5_HZ,
    1.0: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
    2.0: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
    4.0: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
    8.0: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
    16.0: adafruit_mlx90640.RefreshRate.REFRESH_16_HZ,
    32.0: adafruit_mlx90640.RefreshRate.REFRESH_32_HZ,
    64.0: adafruit_mlx90640.RefreshRate.REFRESH_64_HZ,
}

i2c = adafruit_extended_bus.ExtendedI2C(I2C_BUS, frequency=I2C_FREQUENCY)

mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = REFRESH_RATES[REFRESH_RATE_HZ]

frame_buffer = [0.0] * 768  # 32 x 24

# Reads Frame From Thermopile
def getFrame():
    for attempt in range(5):
        try:
            mlx.getFrame(frame_buffer)
            return list(frame_buffer)
        except ValueError:
            # Transient MLX90640 frame error; do not retry forever.
            if attempt == 4:
                raise
    raise RuntimeError("Unable to acquire MLX90640 frame")


if __name__ == "__main__":
    # Tests To Ensure Frame Is Being Read
    frame = getFrame()
    print(frame)