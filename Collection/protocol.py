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

# Keys used in each streamed feature-vector line, sent Rock -> PC as one
# JSON object per line.
KEY_SEQ = "seq"
KEY_PIR = "pir"                            # list[float], length PIR_VECTOR_LENGTH.
                                            # A live sliding-window Haar
                                            # feature, recomputed fresh
                                            # every tick -- always "now",
                                            # never a held-over value.
KEY_THERMAL = "thermal"                    # list[float], length len(THERMAL_FEATURE_NAMES)
KEY_THERMAL_DETECTED = "thermal_detected"  # bool -- False if no blob was
                                            # detected this frame, in which
                                            # case KEY_THERMAL is all zeros.

PIR_VECTOR_LENGTH = 8  # Haar-transform frequency-bin energies

# Flattened Thermopile.thermopile_pipeline.BlobFeatures fields, in the
# fixed order both ends must agree on. `centroid` (a 2-tuple in
# BlobFeatures) is unpacked into centroid_row/centroid_col here.
THERMAL_FEATURE_NAMES = [
    "track_id", "centroid_row", "centroid_col",
    "pixel_count", "width", "height",
    "row_displacement", "col_displacement",
    "displacement_px", "normalised_displacement",
    "track_age_frames", "cumulative_displacement_px",
    "mean_temperature_difference", "max_temperature_difference",
    "virtual_position_x",
    "horizontal_pixels_traversed", "vertical_pixels_traversed",
    "horizontal_pixels_covered", "vertical_pixels_covered",
    "horizontal_direction", "vertical_direction",
]
THERMAL_VECTOR_LENGTH = len(THERMAL_FEATURE_NAMES)
