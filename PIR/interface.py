import time
import threading
import queue

import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_extended_bus


# Sampling Rates Supported By ADC (this library takes the real Hz value
# directly, not an index into a register table like the old ADS1x15-ADC
# library did)
SUPPORTED_RATES = [8, 16, 32, 64, 128, 250, 475, 860]

# Gets The Most Suitable Sampling Rate For Requested Rate
def selectDataRate(requestedHz):
    for rate in SUPPORTED_RATES:
        if rate >= requestedHz:
            return rate
    return SUPPORTED_RATES[-1]


# I2C 2 - PINS 27/28
i2cBusNumber = 2
i2cAddress = 0x48
sampleRateHz = 25


class PIRAcquisition(threading.Thread):
    def __init__(self, outQueue: queue.Queue):
        super().__init__(daemon=True)
        self.outQueue = outQueue
        self.stopEvent = threading.Event()

        # ADC Setup
        self.actualADCRateHz = selectDataRate(sampleRateHz)

        # adafruit_ads1x15 needs an actual I2C bus object, not a bare bus
        # number -- ExtendedI2C is the same tool Thermopile/interface.py
        # already uses to address a non-default Linux I2C bus.
        i2c = adafruit_extended_bus.ExtendedI2C(i2cBusNumber)

        self.adc = ADS.ADS1115(
            i2c,
            address=i2cAddress,
            gain=1,                          # +/- 4.096 V full-scale
            data_rate=self.actualADCRateHz,
            mode=Mode.CONTINUOUS,
        )

        # Persistent channel object -- AnalogIn handles the raw-to-volts
        # conversion itself, so there is no separate getValue()/
        # toVoltage() step with this library.
        self.channel = AnalogIn(self.adc, ADS.P0)

    def stop(self):
        self.stopEvent.set()

    def run(self):
        # Convert Sampling Frequency Into Time Steps
        period = 1.0 / sampleRateHz

        # Gets Time For Next Sample
        next_t = time.monotonic()

        while not self.stopEvent.is_set():
            # Get Voltage
            v = self.channel.voltage

            # Get Timestamp
            ts = time.monotonic()

            # Add (Time, Voltage) To Queue
            self.outQueue.put((ts, v))

            # Set Time For Next Sample
            next_t += period
            sleep_time = next_t - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.monotonic()


# Debug To Test PIR Is Working
if __name__ == "__main__":
    q = queue.Queue(maxsize=2048)
    acq = PIRAcquisition(q)
    acq.start()

    print(f"Requested {sampleRateHz} Hz -> ADS1115 set to {acq.actualADCRateHz} SPS")
    try:
        while True:
            ts, v = q.get()
            print(f"{ts:.4f}\t{v:+.5f} V")
    except KeyboardInterrupt:
        acq.stop()
        print("\nStopped.")
