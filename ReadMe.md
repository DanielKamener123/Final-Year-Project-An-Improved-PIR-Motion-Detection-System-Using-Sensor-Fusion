# Project Data Collection and Fusion

This version uses a deliberately simple timing system.

The **video is timestamped on the PC**, while the **PIR and thermopile are timestamped on the ROCK 4C+**. Immediately before each recording, the PC measures the fixed time offset between the two clocks. The PC then chooses a start time `T0` about 2 seconds in the future and tells the ROCK the equivalent ROCK-clock start time.

Both machines are already initialised and waiting before `T0`. At `T0` they begin **saving** data. Every saved file also contains `time_s`, which is simply:

```text
time_s = seconds since T0
```

All later labelling and fusion uses `time_s`, so the rest of the project does not need to care whether the original timestamp came from the PC or the ROCK.

---

## 1. Start the ROCK 4C+

On the ROCK, open a terminal in the project directory and run:

```bash
python -m Collection.rock_agent --port 5050
```

The ROCK will:

1. Warm up the thermopile if the real thermopile is enabled.
2. Start the PIR acquisition.
3. Wait for the PIR baseline warm-up to finish.
4. Open port `5050`.
5. Wait for the PC.

The PIR acquisition continues running in the background, but samples collected before the scheduled recording start are discarded.

---

## 2. Start Data Collection on the PC

On the PC, run:

```bash
python -m Collection.pc_collector --host <ROCK_IP>
```

Example:

```bash
python -m Collection.pc_collector --host 192.168.1.147 --camera 1
```

The PC then asks whether the recording is training or testing data and which run it belongs to.

---

## 3. What Happens Before Recording Starts

The collector automatically performs the following sequence:

```text
PC                                      ROCK 4C+
 |                                          |
 |---- connect ---------------------------->|
 |                                          |
 |  open camera and keep grabbing frames   | sensors already warmed up
 |                                          |
 |---- clock sync ------------------------->|
 |<--- ROCK time ---------------------------|
 |                                          |
 | calculate: PC time - ROCK time          |
 |                                          |
 | choose common start time T0             |
 | about 2 seconds in the future           |
 |                                          |
 |---- START_AT equivalent ROCK time ------>|
 |                                          |
 | wait for T0                             | wait for T0
 |                                          |
 |========== SAME REAL MOMENT ==============|
 |                                          |
 | save video frames                       | save PIR samples
 |                                          | save thermopile frames
```

The network delay of the start command does not determine the start time. The command is sent **before** recording starts and tells both machines which future time to use.

---

## 4. How Timestamps Are Stored

### PIR

Each PIR voltage is timestamped on the ROCK immediately after the ADC value has been read.

`pir.csv` contains:

```text
rock_timestamp, pc_timestamp, time_s, seq, voltage
```

- `rock_timestamp`: original ROCK clock timestamp.
- `pc_timestamp`: the same measurement converted onto the PC clock using the measured fixed offset.
- `time_s`: seconds since the common recording start `T0`.
- `seq`: sample sequence number.
- `voltage`: raw PIR voltage.

### Thermopile

Each complete thermopile frame is timestamped on the ROCK immediately after the frame has been read.

`thermopile.csv` contains:

```text
rock_timestamp, pc_timestamp, time_s, seq, pixel_0, ...
```

### Video

The camera is opened and warmed **before** `T0`. Frames are continuously grabbed before the recording starts so the camera is already running. Those warm-up frames are discarded.

Only frames completed at or after `T0` are saved.

`video_frames.csv` contains:

```text
frame_index, pc_timestamp, time_s
```

The important column for later processing is `time_s`.

---

## 5. Stop the Recording

While the recording is running, type:

```text
stop
```

The PC stops the ROCK sensor streams and finishes writing the video.

Each recording folder contains approximately:

```text
recording_YYYYMMDD_HHMMSS/
├── pir.csv
├── thermopile.csv
├── video.mp4
├── video_frames.csv
└── metadata.json
```

`metadata.json` stores the measured PC/ROCK clock offset, the common start times, sensor configuration and recording information.

---

# Video Labelling

## 6. Convert the Video for Label Studio

The video file is written with a fixed declared frame rate of 30 fps so that each saved frame remains one video frame.

Use:

```bash
ffmpeg -i "video.mp4" -c:v libx264 -pix_fmt yuv420p -r 30 -fps_mode cfr -c:a aac -movflags +faststart "video_labelstudio.mp4"
```

In the Label Studio configuration, use:

```text
frameRate="30"
```

This lets Label Studio frame numbers map back to the rows in `video_frames.csv`.

---

## 7. Label Human / No Human

Label only the intervals where a human is present.

For example:

```text
0.0 - 5.2 s      no human
5.2 - 14.7 s     human
14.7 - 22.0 s    no human
```

The exact sensor and video timestamps do not have to be perfect to the millisecond. They only need to share the same recording timeline closely enough that a human visible in the video corresponds to the same period in the PIR and thermopile data.

Export the Label Studio task as JSON and run:

```bash
python -m Collection.labelstudio_labels \
    --export <LABEL_EXPORT.json> \
    --recording <RECORDING_FOLDER>
```

Example:

```bash
python -m Collection.labelstudio_labels --export "Datasets/training/training_outdoors_2/recording_20260924_121936/labels.json" --recording "Datasets/training/training_outdoors_2/recording_20260924_121936"
```

This creates:

```text
labels.csv
```

with:

```text
start_s, end_s, human
```

These times are on the same `time_s` timeline as the sensors.

---

# Offline Processing and Sensor Fusion

## 8. Configure the Fusion Window

Open `main.py`.

The main settings are:

```python
WINDOW_SECONDS = 1.0
WINDOW_STEP_SECONDS = 0.25

TRAIN_DATASET = Path("Datasets/training")
TEST_DATASET = Path("Datasets/testing")
```

### `WINDOW_SECONDS`

This is how much previous PIR history is used for one PIR feature vector.

With:

```python
WINDOW_SECONDS = 1.0
```

and a PIR sampling rate of 25 Hz, approximately 25 PIR samples contribute to each window.

### `WINDOW_STEP_SECONDS`

This controls how often one combined fusion vector is created.

With:

```python
WINDOW_STEP_SECONDS = 0.25
```

one new feature vector is generated every 250 ms.

The windows therefore overlap.

---

## 9. How One Fusion Row Is Produced

At a fusion time such as:

```text
t = 10.00 s
```

the offline processing does approximately this:

```text
PIR samples from 9.00 s to 10.00 s
             ↓
       PIR features

latest thermopile frame at/before 10.00 s
             ↓
    thermopile features

PIR features + thermopile features
             ↓
     combined feature vector

labels.csv at 10.00 s
             ↓
      human / no human
```

Everything uses the same `time_s` timeline.

---

## 10. Run Processing and Machine Learning

After the training and testing recordings have been labelled, run:

```bash
python main.py
```

The program will:

1. Load the raw PIR recordings.
2. Load the raw thermopile recordings.
3. Use their `time_s` timestamps.
4. Re-run PIR processing and extract PIR features.
5. Re-run thermopile processing and extract thermopile features.
6. Match the features at common fusion times.
7. Apply the human/no-human video label for that time.
8. Build the training and testing feature datasets.
9. Train the machine-learning models.
10. Test the models and save their results.

Outputs are stored under:

```text
Processed/
└── window_1.00s_step_0.25s/
    ├── train_features.csv
    ├── test_features.csv
    └── model_comparison_results.csv
```

---

# Simple Mental Model

The entire collection process can be understood as:

```text
              COMMON START T0
                    |
        +-----------+-----------+
        |           |           |
       PIR      THERMOPILE    VIDEO
        |           |           |
      time_s      time_s      time_s
        |           |           |
        +-----------+-----------+
                    |
              raw recordings
                    |
              label video
                    |
               labels.csv
                    |
        process sensor data
                    |
        combined feature vector
                    |
            machine learning
```

The important rule is simply:

> **All three streams share the same `t = 0`, and all later processing uses seconds since that start.**

No camera-latency calibration, end-of-recording clock re-sync, or millisecond-level correction is required for this version.
