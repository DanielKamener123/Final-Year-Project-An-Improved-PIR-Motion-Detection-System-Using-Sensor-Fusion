# Shared protocol constants used by the PC and ROCK collection programs.
DEFAULT_PORT = 5050

# Commands
CMD_START_AT = "start_at"
CMD_STOP = "stop"
CMD_SYNC = "sync"

# Command fields
KEY_PC_SEND_TIME = "pc_send_time"
KEY_ROCK_START_TIME = "rock_start_time"

# Status messages
KEY_STATUS = "status"
STATUS_READY = "ready"
STATUS_RECORDING = "recording"
STATUS_STOPPED = "stopped"

# Data types
KEY_TYPE = "type"
TYPE_SYNC = "sync"
TYPE_PIR = "pir"
TYPE_THERMAL = "thermal"

# Common data fields
KEY_SEQ = "seq"
KEY_ROCK_TIMESTAMP = "rock_timestamp"
KEY_ROCK_TIME = "rock_time"

# PIR fields
KEY_VOLTAGE = "voltage"
KEY_PIR_BASELINE = "pir_baseline_v"
KEY_PIR_SAMPLE_RATE = "pir_sample_rate_hz"

# Thermopile fields
KEY_FRAME = "frame"
KEY_THERMAL_BACKGROUND = "thermal_background"
KEY_THERMAL_SIGMA = "thermal_sigma"
KEY_THERMAL_FRAME_SHAPE = "thermal_frame_shape"
KEY_THERMAL_RATE = "thermal_rate_hz"
