import time
import threading
import queue
from dataclasses import dataclass

import adafruit_ads1x15.ads1115 as ADS


# Sampling Rates Supported By ADC
SUPPORTED_RATES = [
    (8, 0), (16, 1), (32, 2), (64, 3),
    (128, 4), (250, 5), (475, 6), (860, 7),
]

# Gets The Most Suitable Sampling Rate For Requested Rate
def selectDataRate(requestedHz):
    for rate, index in SUPPORTED_RATES:
        if rate >= requestedHz:
            return rate, index
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
        self.actualADCRateHz, sampleRateIndex = selectDataRate(sampleRateHz)
        self.adc = ADS.ADS1115(i2cBusNumber, i2cAddress)

        # Gets Voltage Scale Of 125 uV
        self.adc.setGain(self.adc.PGA_4_096V)      
        self.adc.setMode(self.adc.MODE_CONTINUOUS)

        # Sets Sampling Rate
        self.adc.setDataRate(sampleRateIndex)
        self.adc.readADC(0)                       

    def stop(self):
        self.stopEvent.set()

    def run(self):
        # Convert Sampling Frequency Into Time Steps
        period = 1.0 / sampleRateHz

        # Gets Time For Next Sample
        next_t = time.monotonic()

        while not self.stopEvent.is_set():
            # Get RAW Data
            raw = self.adc.getValue()

            #Covert ADC To Voltage
            v = self.adc.toVoltage(raw)

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