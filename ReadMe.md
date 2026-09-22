# Running the Project

## 1. Start the ROCK 4C+

On ROCK, open a terminal in the project directory and run:

```bash
python -m Collection.rock_agent --port 5050
```

The ROCK will:

1. Warm up the thermopile background model.
2. Start the PIR acquisition.
3. Wait approximately 10 seconds for the PIR baseline calibration.
4. Wait until both sensors are ready.
5. Open port `5050` and wait for the PC to connect.

## 2. Start Data Collection on the PC

On the PC, open a terminal in the project directory.

Run:

```bash
python -m Collection.pc_collector --host <ROCK_IP>
```

For example:

```bash
python -m Collection.pc_collector --host 192.168.1.147 --camera 1
```


## 3. Select Training or Testing Data
## 4. Select a Run
## 5. Record the Experiment

Once the run has been selected:

1. The PC camera starts recording.
2. The PC sends the start command to the ROCK.
3. PIR samples are continuously transmitted from the ROCK.
4. Raw thermopile frames are continuously transmitted from the ROCK.
5. The PC timestamps all received sensor data using the **PC clock**.
6. The video and sensor data are recorded simultaneously.

```text
stop
```

## 6. Recorded Data
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

If the Label Studio export contains multiple tasks, specify the required task:

```bash
python -m Collection.labelstudio_labels \
    --export labels.json \
    --recording Datasets/training/run_01/recording_20260917_140000 \
    --task-id <TASK_ID>
```

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