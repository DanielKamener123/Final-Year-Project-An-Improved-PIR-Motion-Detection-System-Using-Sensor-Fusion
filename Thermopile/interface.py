import adafruit_mlx90640
import adafruit_extended_bus

# Config
I2C_BUS = 7
REFRESH_RATE_HZ = 4.0

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

# The Sensor Is Opened On The First Read (not on import), so a missing or
# unplugged thermopile no longer stops main.py from starting. After an I2C
# failure the connection is dropped and re-opened on the next read.
i2c = None
mlx = None

frame_buffer = [0.0] * 768  # 32 x 24

# Opens The Thermopile On The I2C Bus
def connect():
    global i2c, mlx
    i2c = adafruit_extended_bus.ExtendedI2C(I2C_BUS)
    sensor = adafruit_mlx90640.MLX90640(i2c)
    sensor.refresh_rate = REFRESH_RATES[REFRESH_RATE_HZ]
    mlx = sensor

# Forgets The Connection So The Next Read Re-Opens The Sensor
def disconnect():
    global i2c, mlx
    mlx = None
    if i2c is not None:
        try:
            i2c.deinit()
        except Exception:
            pass
    i2c = None

# Reads Frame From Thermopile
def getFrame():
    if mlx is None:
        connect()
    for attempt in range(5):
        try:
            mlx.getFrame(frame_buffer)
            return list(frame_buffer)
        except ValueError:
            # Transient MLX90640 frame error; do not retry forever.
            if attempt == 4:
                raise
        except OSError:
            # I2C failure (e.g. sensor unplugged): re-open on the next call
            disconnect()
            raise
    raise RuntimeError("Unable to acquire thermopile frame")

# Reads Frames For Background Calibration - Make Sure No One Is In Frame
def getCalibrationFrames(frameCount):
    return [getFrame() for _ in range(int(frameCount))]


if __name__ == "__main__":
    # Tests To Ensure Frame Is Being Read
    frame = getFrame()
    print(frame)
