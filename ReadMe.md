# Running the Project

This project uses the **ROCK 4C+** to collect data from the PIR sensor and MLX90640 thermopile, while a **PC** controls the recording session, timestamps the received sensor data and records the reference video.

All commands should be run from the root `Project` directory.

---

## 1. Start the ROCK 4C+

On the ROCK 4C+, open a terminal in the project directory and run:

```bash
python -m Collection.rock_agent --port 5050
```

The ROCK will first:

1. Warm up the thermopile background model.
2. Start the PIR acquisition.
3. Wait approximately 10 seconds for the PIR baseline calibration.
4. Wait until both sensors are ready.
5. Open port `5050` and wait for the PC to connect.

The scene should remain empty while the sensors are warming up.

The terminal should eventually show:

```text
Both sensors warmed up.
Listening on port 5050 for the PC to connect...
```

Leave this program running.

---

## 2. Start Data Collection on the PC

On the PC, open a terminal in the project directory.

Run:

```bash
python -m Collection.pc_collector --host <ROCK_IP>
```

For example:

```bash
python -m Collection.pc_collector --host 192.168.1.100
```

Replace `192.168.1.100` with the actual IP address of the ROCK 4C+.

The PC will connect to the ROCK and confirm that both sensors have completed their warm-up before recording can begin.

---

## 3. Select Training or Testing Data

The program will ask:

```text
Is this training or testing data? [training/testing]:
```

Enter either:

```text
training
```

or:

```text
testing
```

Training and testing recordings are stored separately.

The testing recordings should contain experiments that are not used to train the machine-learning models.

---

## 4. Select a Run

The program will then allow a new run to be created or an existing run to be used.

A run represents a continuous experimental group.

For example:

```text
Datasets/
├── training/
│   ├── run_01/
│   ├── run_02/
│   └── run_03/
└── testing/
    ├── run_01/
    └── run_02/
```

Multiple recordings can be stored inside the same run without overwriting previous recordings.

---

## 5. Record the Experiment

Once the run has been selected:

1. The PC camera starts recording.
2. The PC sends the start command to the ROCK.
3. PIR samples are continuously transmitted from the ROCK.
4. Raw thermopile frames are continuously transmitted from the ROCK.
5. The PC timestamps all received sensor data using the **PC clock**.
6. The video and sensor data are recorded simultaneously.

Perform the complete experiment while the system continues recording.

A recording can contain multiple movements through the field of view. It is not necessary to create a new recording for every individual movement.

When the experiment is complete, type:

```text
stop
```

The PC will stop:

* PIR data collection
* thermopile data collection
* video recording

and save the recording.

---

## 6. Recorded Data

Each recording is stored in a folder similar to:

```text
Datasets/
└── training/
    └── run_01/
        └── recording_YYYYMMDD_HHMMSS/
            ├── pir.csv
            ├── thermopile.csv
            ├── video.mp4
            ├── video_frames.csv
            └── metadata.json
```

### `pir.csv`

Contains the raw PIR voltage samples:

```text
pc_timestamp, seq, voltage
```

### `thermopile.csv`

Contains every raw 32 × 24 thermopile frame:

```text
pc_timestamp, seq, pixel_0, pixel_1, ... pixel_767
```

### `video.mp4`

Reference video used to determine whether a human is present.

### `video_frames.csv`

Stores the PC timestamp associated with each recorded video frame.

This is used to accurately align Label Studio labels with the sensor measurements.

### `metadata.json`

Stores information required to reproduce the processing, including sensor calibration information and recording configuration.

---

# Labelling the Recorded Data

The video is labelled using **Label Studio**.

The required label is binary:

```text
Human Present
No Human
```

Only the periods where a human is present need to be explicitly labelled as **Human Present**.

All remaining video time is treated as:

```text
No Human
```

Use timeline/video interval labels so that the start and end of each period containing a human are identified.

---

## 7. Export Labels from Label Studio

Export the labelled task from Label Studio as JSON.

Convert the Label Studio export into the format used by the project with:

```bash
python -m Collection.labelstudio_labels \
    --export <LABEL_EXPORT.json> \
    --recording <RECORDING_FOLDER>
```

Example:

```bash
python -m Collection.labelstudio_labels \
    --export labels.json \
    --recording Datasets/training/run_01/recording_20260917_140000
```

This creates:

```text
labels.csv
```

inside the recording folder.

The complete recording folder will then contain:

```text
pir.csv
thermopile.csv
video.mp4
video_frames.csv
metadata.json
labels.csv
```

If the Label Studio export contains multiple tasks, specify the required task:

```bash
python -m Collection.labelstudio_labels \
    --export labels.json \
    --recording Datasets/training/run_01/recording_20260917_140000 \
    --task-id <TASK_ID>
```

---

# Processing the Dataset and Running Machine Learning

The raw sensor data is intentionally stored rather than only storing extracted features.

This allows different:

* processing methods
* feature sets
* window lengths
* window overlaps

to be tested later without collecting the sensor data again.

---

## 8. Configure the Fusion Window

Open:

```text
main.py
```

The main configuration is at the top of the file:

```python
WINDOW_SECONDS = 1.0
WINDOW_STEP_SECONDS = 0.25

TRAIN_DATASET = Path("Datasets/training")
TEST_DATASET = Path("Datasets/testing")

OUTPUT_ROOT = Path("Processed")
```

### `WINDOW_SECONDS`

Defines how much previous sensor history is used to produce each feature vector.

For example:

```python
WINDOW_SECONDS = 1.0
```

means that the PIR feature extractor receives the previous **1 second of PIR measurements**.

At 25 Hz this corresponds to approximately:

```text
25 PIR samples
```

### `WINDOW_STEP_SECONDS`

Defines how often a new fusion feature vector is produced.

For example:

```python
WINDOW_STEP_SECONDS = 0.25
```

produces a new feature vector every:

```text
250 ms
```

Therefore a 1-second window with a 0.25-second step produces overlapping windows.

---

## 9. Run the Offline Processing and Machine Learning

Once the training and testing recordings have been labelled, run:

```bash
python main.py
```

The program will:

1. Load all raw training recordings.
2. Reconstruct the PIR features using the feature extraction currently implemented in:

```text
PIR/processing.py
```

3. Reconstruct the thermopile features using the current thermopile processing pipeline.
4. Combine the PIR and thermopile features.
5. Create the training feature dataset.
6. Repeat the same processing for the testing recordings.
7. Train the machine-learning algorithms on the training dataset.
8. Evaluate them using the separate testing dataset.
9. Save the model comparison results.

---

## 10. Processed Results

For:

```python
WINDOW_SECONDS = 1.0
WINDOW_STEP_SECONDS = 0.25
```

the output will be stored in:

```text
Processed/
└── window_1.00s_step_0.25s/
    ├── train_features.csv
    ├── test_features.csv
    └── model_comparison_results.csv
```

### `train_features.csv`

Feature vectors generated from the training recordings.

### `test_features.csv`

Feature vectors generated from the testing recordings.

### `model_comparison_results.csv`

Performance results for the machine-learning algorithms.

---

# Testing Different Processing Methods

Because the original raw sensor measurements are preserved, the feature extraction can be modified without repeating the experiments.

For example, change:

```python
WINDOW_SECONDS = 1.0
```

to:

```python
WINDOW_SECONDS = 0.5
```

and run:

```bash
python main.py
```

again.

A new dataset will be generated from exactly the same recordings.

Similarly, PIR features can be changed inside:

```text
PIR/processing.py
```

and the program can simply be rerun.

The offline processing will use whatever PIR feature extraction is currently implemented in the processing file.

The same principle applies to the thermopile processing.

---

# Typical Workflow

```text
ROCK 4C+
    ↓
Start rock_agent
    ↓
Sensors warm up
    ↓
PC
    ↓
Start pc_collector
    ↓
Choose training/testing
    ↓
Choose new/current run
    ↓
Start recording
    ↓
Perform experiments
    ↓
Type "stop"
    ↓
Raw PIR + thermal data + video saved
    ↓
Label video in Label Studio
    ↓
Convert labels to labels.csv
    ↓
Configure main.py
    ↓
Run python main.py
    ↓
Generate fusion feature vectors
    ↓
Train models
    ↓
Test models
    ↓
Compare performance
```

# Important

Always run the Python commands from the root `Project` directory.

For example:

```text
Project/
├── Collection/
├── Fusion/
├── PIR/
├── Thermopile/
└── main.py
```

The terminal should be inside `Project/` before running commands such as:

```bash
python -m Collection.rock_agent
```

or:

```bash
python main.py
```

This is required so Python can correctly find packages such as:

```python
from PIR.interface import ...
from Thermopile.processing import ...
```
