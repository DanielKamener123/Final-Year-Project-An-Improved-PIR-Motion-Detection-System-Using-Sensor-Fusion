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

## 2b. GUI collector (camera optional, live labels)

```bash
python -m Collection.pc_collector_gui --host <ROCK_IP> --camera 1
```

Arguments only pre-fill the form (`--no-camera` starts with the camera unticked).

1. Choose the data type, type a new recording name, check the ROCK IP, tick/untick **Record camera** (use **Test camera** to check the index).
2. **Connect + calibrate** - keep the area empty while the ROCK calibrates.
3. **Start recording**. Press **SPACE** (or the Enter/Leave button) when someone enters the inbound zone and again when it is empty. **Undo last press** removes a mistake.
4. **Stop + save**. Labels are written straight to `labels.csv` (same format as the playback tool), and can still be refined in `Tools.synchronised_playback` if a video was recorded.

If the camera is missing or drops out, recording continues sensors-only and `metadata.json` records it (`camera_used`, `camera_error`, `camera_lost_during_recording`). A silent PIR/thermopile or lost ROCK connection still stops the recording.

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

The viewer displays the recorded video alongside the PIR and thermopile signals on their shared timeline. The thermopile card replays the recording through the real `ThermopilePipeline`; press `V`/`C` (or `1`-`8`) to step through its stages: raw temperature with detections, background model, difference, Z score, filtered Z score, the recent-frame-average detector, the threshold ratio, and the foreground mask with blobs and tracks. Use **HUMAN START** when a person enters the defined Inbound monitored area, and **HUMAN END** when no person remains inside it. Use **Save labels** (or press `S`) to save:

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

The dataset loader (`Fusion/dataset.py`) finds recordings recursively by `metadata.json`. It reads each recording's `labels.csv`, `pir.csv` and `thermopile.csv` and combines the sensor features at thermal-frame times using their shared timestamps. Recordings without `labels.csv` are skipped. Evaluation recordings (validation and testing) also need `labels(outbound).csv`; a header-only label file means nobody was present.

For example, for a 1-second PIR history window at `t = 10.0 s`:

```text
PIR measurements over approximately 9–10 s  -> PIR features
Thermopile frame at t = 10 s                  -> thermal features
PIR + thermal features                        -> fused feature vector
Inbound label at t = 10 s                     -> binary training target
```

Training, validation and test recordings remain in their separate `Datasets/` folders. Validation is only used to choose the model, window and alarm thresholds; testing gives the verification result. Run everything from the project directory:

```bash
# 1. Model and window selection: every scikit-learn classifier at every window vs thresholded PIR
python offlineRun.py --compare --name Compare_22Oct

# 2. Train the first-principles Random Forest at the chosen window, tune its alarm
#    thresholds on validation and export it to Processed/realtime/fusion_rf.json
python offlineRun.py --train --window 1.5 --name Train_22Oct

# 3. Final verification of the exported model on the testing recordings
python offlineRun.py --test --name Test_22Oct
```

Each run writes `report.txt` and PNG figures to `Processed/<name>/`. Reports check the proposal requirements: Spec 1 (at least 20% accuracy gain and at least 50% fewer false alarms than the thresholded PIR) and Spec 4 (alarm within 10 s); Specs 2 and 3 (range and speed) come from the per-recording table on the testing set. Metric definitions are at the top of `Fusion/evaluation.py`.

| File | Purpose |
|------|---------|
| `Fusion/dataset.py` | Builds the fused feature rows from the recordings |
| `Fusion/randomForest.py` | First-principles Random Forest (training and inference, numpy only) |
| `Fusion/realtimeModel.py` | Three-state alarm machine and the real-time model wrapper |
| `Fusion/evaluation.py` | Pd, Pfa, accuracy, Ward event metrics and alarm-threshold tuning |
| `Fusion/models.py` | scikit-learn comparison of every classifier at every window |
| `Fusion/report.py` | Report tables and figures |

## 7. Live system on the ROCK

```bash
python main.py
```

Loads `Processed/realtime/fusion_rf.json`, calibrates both sensors (keep the zone empty) and shows the PIR-only and fused states.

## 8. Demo fallback: replay a recording as if live

```bash
python -m Tools.demo_replay --recording "Datasets/validation/HomeDadValid" --start 20
```

Plays the video while feeding the recorded PIR samples and thermal frames through the same real-time code as `main.py` (thermopile pipeline, PIR processor, first-principles Random Forest and state machine). Shows the video, a thermopile processing stage, the PIR signal, and the PIR-only and fused states with the fused confidence. Keys: `Space` pause, `R` restart, `V`/`C` stage, `Q` quit.

## 9. Other tools

- `python -m Thermopile.display` - live thermopile debug dashboard on the ROCK.
- `python -m PIR.display` - live PIR signal and features on the ROCK.
- `python -m Tools.export_review_predictions` - runs the most recently trained model (`Processed/realtime/fusion_rf.json`) on every validation and testing recording and writes `review_predictions.csv` into each, so the synchronised playback can show its confidence and states and jump between false alarms (`N`/`P`). `offlineRun.py --train` runs this automatically. Add `--recording <folder>` for a single recording.

- `python -m Tools.contrast_analysis` - measures, for every labelled recording, how warm the hottest pixel gets with a person present vs with nobody present, and writes a table plus two figures (hottest-pixel histograms and typical frames, high vs low contrast) to `Processed/Contrast_Analysis/`.

**Key principle:** video, PIR and thermopile data all use `time_s` relative to the same `T0`; recording names identify sessions without extra timestamp-named directories.
