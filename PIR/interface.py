import time
import threading
import queue
from dataclasses import dataclass

import ADS1x15


# Sampling Rates Supported By ADC
_SUPPORTED_RATES = [
    (8, 0), (16, 1), (32, 2), (64, 3),
    (128, 4), (250, 5), (475, 6), (860, 7),
]

# Gets The Most Suitable Sampling Rate For Requested Rate
def _select_data_rate(requested_hz: float):
    for rate, index in _SUPPORTED_RATES:
        if rate >= requested_hz:
            return rate, index
    return _SUPPORTED_RATES[-1]


@dataclass
class PIRInterfaceConfig:
    # I2C 2 - PINS 27/28
    i2c_bus_number = 2 
    i2c_address = 0x48
    sample_rate_hz = 25


class PIRAcquisition(threading.Thread):
    def __init__(self, config: PIRInterfaceConfig, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.out_queue = out_queue
        self.stop_event = threading.Event()

        # ADC Setup
        self.actual_adc_rate_hz, dr_index = _select_data_rate(config.sample_rate_hz)
        self.adc = ADS1x15.ADS1115(config.i2c_bus_number, config.i2c_address)

        # Gets Voltage Scale Of 125 uV
        self.adc.setGain(self.adc.PGA_4_096V)      
        self.adc.setMode(self.adc.MODE_CONTINUOUS)

        # Sets Sampling Rate
        self.adc.setDataRate(dr_index)
        self.adc.readADC(0)                       

    def stop(self):
        self.stop_event.set()

    def run(self):
        # Convert Sampling Frequency Into Time Steps
        period = 1.0 / self.config.sample_rate_hz

        # Gets Time For Next Sample
        next_t = time.monotonic()

        while not self.stop_event.is_set():
            # Get RAW Data
            raw = self.adc.getValue()

            #Covert ADC To Voltage
            v = self.adc.toVoltage(raw)

            # Get Timestamp
            ts = time.monotonic()

            # Add (Time, Voltage) To Queue
            self.out_queue.put((ts, v))

            # Set Time For Next Sample
            next_t += period
            sleep_time = next_t - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.monotonic()


# Debug To Test PIR Is Working
if __name__ == "__main__":
    cfg = PIRInterfaceConfig()
    q = queue.Queue(maxsize=2048)
    acq = PIRAcquisition(cfg, q)
    acq.start()

    print(f"Requested {cfg.sample_rate_hz} Hz -> ADS1115 set to {acq.actual_adc_rate_hz} SPS")
    try:
        while True:
            ts, v = q.get()
            print(f"{ts:.4f}\t{v:+.5f} V")
    except KeyboardInterrupt:
        acq.stop()
        print("\nStopped.")