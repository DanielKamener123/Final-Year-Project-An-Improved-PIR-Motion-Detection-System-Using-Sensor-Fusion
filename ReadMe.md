# Project Data Collection and Sensor Fusion

## 1. Start the ROCK 4C+

On the ROCK, open a terminal in the project directory and run:

```bash
python -m Collection.rock_agent
```

## 2. Start data collection on the PC

On the PC, run:

```bash
python -m Collection.pc_agent --host <ROCK_IP> --camera 1
```

For example:

```bash
python -m Collection.pc_agent --host 192.168.1.147 --camera 1
```

Change the camera index if necessary (`--camera 0`, `1`, etc.). The program immediately asks:

```text
Is this training, validation or testing data? [training/validation/testing]: train
Recording name: HomeTamTrain
```

Accepted data-type shortcuts are `train`/`training`, `val`/`validation` and `test`/`testing`.

**Every collection is a new recording.** There is no `start` command and no `new/current` selection. The recording name is used as its folder name. If that folder already exists in the selected data-type directory, the program requests a different name instead of overwriting or adding to it.

After you enter a unique recording name, the PC connects to the ROCK, waits for sensor readiness, warms up the camera, synchronises the clocks and schedules recording automatically. Keep the monitored area empty during sensor calibration. Once recording begins, type `stop` to finish.

## 3. Recording folder layout

A recording is saved directly inside the folder you named, under its data type:

```text
Datasets/
├── training/
│   └── HomeTamTrain/
│       ├── pir.csv
│       ├── thermopile.csv
│       ├── video.mp4
│       ├── video_frames.csv
│       └── metadata.json
├── validation/
│   └── HomeTamValidate/
│       ├── pir.csv
│       ├── thermopile.csv
│       ├── video.mp4
│       ├── video_frames.csv
│       └── metadata.json
└── testing/
    └── HomeTamTest/
        ├── pir.csv
        ├── thermopile.csv
        ├── video.mp4
        ├── video_frames.csv
        └── metadata.json
```

There is **no extra `recording_YYYYMMDD_HHMMSS` subfolder**. The `metadata.json` file preserves the recording name, split, sensor settings, time synchronisation information and file names.

## 4. How recording synchronisation works

```text
PC                                      ROCK 4C+
 |                                          |
 |---- connect ---------------------------->|
 |  warm up camera                         | warm up / calibrate sensors
 |---- clock sync ------------------------->|
 |<--- ROCK time ---------------------------|
 |  calculate PC minus ROCK clock offset   |
 |  schedule common start T0               |
 |---- START_AT (ROCK-clock T0) ----------->|
 |  wait for T0                            | wait for T0
 |============== COMMON T0 =================|
 |  save video frames                      | stream PIR samples and thermal frames
```

The network delay for the start command does not define the start time, because the command is sent ahead of `T0`.

- `pir.csv`: `rock_timestamp, pc_timestamp, time_s, seq, voltage`
- `thermopile.csv`: `rock_timestamp, pc_timestamp, time_s, seq, pixel_0, ...`
- `video_frames.csv`: `frame_index, pc_timestamp, time_s`
- `video.mp4`: recorded camera frames
- `metadata.json`: common start, clock offset, sensor settings and collection statistics

The important alignment field is `time_s`. Video frames are timestamped individually in `video_frames.csv`; do not assume frame number divided by a nominal frame rate is the exact capture timestamp.

## 5. Label recordings using synchronised playback

**Label Studio is no longer used.** Use the project's synchronised playback and labelling tool instead:

```bash
python -m Tools.synchronised_playback --recording "Datasets/training/HomeTamTrain"
```

The viewer displays the recorded video alongside the PIR and thermopile signals on their shared timeline. Use **HUMAN START** when a person enters the defined Inbound monitored area, and **HUMAN END** when no person remains inside it. Use **Save labels** (or press `S`) to save:

```text
Datasets/training/HomeTamTrain/labels.csv
```

The saved `labels.csv` contains interval labels:

```text
start_s,end_s,human
5.2,14.7,1
```

Intervals not labelled as human are treated as no human by the offline dataset loader. Check that all human-presence intervals are correctly marked before saving. The viewer provides play/pause, timeline navigation, undo and other review controls; press `Q` or `Esc` to exit.

**Inbound labels** indicate that a human is physically inside the monitored area. If you are also maintaining `labels(outbound).csv` for your wider-field-of-view false-alarm evaluation, keep that separate from the training `labels.csv`. The current playback label-save operation writes `labels.csv` only; it does not automatically create the Outbound label file.

There is no video conversion, Label Studio export or `Collection.labels` conversion step in this workflow.

## 6. Offline feature extraction and fusion

The dataset loader (`Fusion/dataset.py`) finds recordings recursively by `metadata.json`, so it supports the new folder structure directly. It reads each recording's `labels.csv`, `pir.csv` and `thermopile.csv` and combines the sensor features at thermal-frame times using their shared timestamps.

For example, for a 1-second PIR history window at `t = 10.0 s`:

```text
PIR measurements over approximately 9–10 s  -> PIR features
Thermopile frame at t = 10 s                  -> thermal features
PIR + thermal features                        -> fused feature vector
Inbound label at t = 10 s                     -> binary training target
```

The PIR window length and model selection are configured in the offline processing code. Training, validation and test recordings remain in their separate `Datasets/` folders. Do not mix testing recordings into the model-selection process.

Run the project's offline entry point from the project directory when the required recordings have been labelled:

```bash
python main.py
```

**Key principle:** video, PIR and thermopile data all use `time_s` relative to the same `T0`; recording names identify sessions without extra timestamp-named directories.
