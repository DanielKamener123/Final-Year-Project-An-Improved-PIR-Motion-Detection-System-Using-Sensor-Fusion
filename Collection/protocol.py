# Shared message-format constants for the PC <-> Rock 4C+ data-collection
# link, so both ends agree on the port and JSON field names without
# duplicating the values in two separate files. Deliberately has no
# hardware/sensor dependencies so the PC side can import it without
# needing the PIR/Thermopile packages installed.

DEFAULT_PORT = 5050

# Control messages, sent PC -> Rock as one JSON object per line:
# {"cmd": "start"} / {"cmd": "stop"}
CMD_START = "start"
CMD_STOP = "stop"

# Status message, sent Rock -> PC once as soon as a connection is
# accepted: {"status": "ready"}. The Rock does not send this (and does
# not even open the port at all) until both the thermopile background
# model and the PIR baseline calibration have finished warming up.
KEY_STATUS = "status"
STATUS_READY = "ready"
STATUS_RECORDING = "recording"
STATUS_STOPPED = "stopped"

KEY_TYPE = "type"
TYPE_PIR = "pir"
TYPE_THERMAL = "thermal"

KEY_SEQ = "seq"
KEY_VOLTAGE = "voltage"
KEY_FRAME = "frame"

KEY_PIR_BASELINE = "pir_baseline_v"
KEY_THERMAL_BACKGROUND = "thermal_background"
KEY_THERMAL_SIGMA = "thermal_sigma"
KEY_THERMAL_FRAME_SHAPE = "thermal_frame_shape"
KEY_PIR_SAMPLE_RATE = "pir_sample_rate_hz"
KEY_THERMAL_RATE = "thermal_rate_hz"
