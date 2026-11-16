from pathlib import Path
import glob

import numpy as np
import pandas as pd

from Fusion.dataset import buildRecordingDataset, getFusedFeatureNames
from Fusion.gradientBoosting import GradientBoosting
from Fusion.models import buildModels, confidenceScores
from Fusion.realtimeModel import StateMachine, INTRUDER, MAYBE_INTRUDER, NO_INTRUDER, applyPirStrongRule, PIR_STRONG_V
from PIR.processing import getHistoryFeatureNames


# Training sources, including older recordings
DEMO_DATASETS = [
    "Datasets/demo",
    "Datasets/training",
    "Datasets/validation",
    "Datasets/testing",
    "Datasets/old",          
]
# PIR feature history per thermal decision
DEMO_WINDOW_S = 2.0
# Venue recordings used for held-out evaluation
VENUE_PREFIXES = ("Uni", "Backup")   
# Names treated as empty-scene recordings
EMPTY_KEYWORDS = ("empty", "distractor", "negative")
# Ignore uncertain labels around walk boundaries
BOUNDARY_S = 1.0             
# Ignore nearby negative rows that may still contain a person
CLEAN_NEGATIVE_S = 5.0       
# Repeat venue and empty rows to give them more influence
FOCUS_WEIGHT = 3             
# Minimum fraction of walks that must trigger an alarm
MIN_DETECTION = 95.0         


# Test thresholds with slightly reduced confidence scores
ROBUST_SHIFT = 0.05
# Minimum walk detection after reducing scores
ROBUST_DETECTION = 90.0
# Allow small false alarm differences before choosing by detection
ALARM_TOLERANCE = 1.0


# Limit repeated INTRUDER alarms during one walk
MAX_SEGMENTS = 1.7
# Maximum time after entry to count a walk as detected
MAX_DELAY_S = 10.0           


# Keep frames inside short, fast passes
SHORT_WALK_S = 3.0


# Empty-scene percentile used to identify motionless frames
STILL_PERCENTILE = 99.0


# Find all recordings with a metadata file
# Use a set to avoid duplicates; sort for repeatable order
def demoRecordings(datasets=DEMO_DATASETS):
    folders = set()
    for pattern in datasets:
        for root in glob.glob(pattern):
            folders.update(path.parent for path in Path(root).rglob("metadata.json"))
    return sorted(folders)


# Flag empty recordings by name or absence of human labels
def isEmptyRecording(path, labels):
    name = Path(path).name.lower()
    return any(word in name for word in EMPTY_KEYWORDS) or not (labels["human"] == 1).any()


# Include Uni/Backup sessions, excluding old recordings
def isVenueRecording(path):
    path = Path(path)
    return path.name.startswith(VENUE_PREFIXES) and "old" not in path.parts


# Build one feature table across all recording folders
# Add flags that control training and venue evaluation
def buildDemoData(window, progress=print):
    parts = []
    for path in demoRecordings():

        # Require human labels for each recording
        labelsPath = path / "labels.csv"
        if not labelsPath.exists():
            progress(f"  skipping {path}: no labels.csv")
            continue

        # Extract aligned PIR and thermal features
        data = buildRecordingDataset(path, window)
        if data.empty:
            continue

        # Read walk intervals and identify the recording type
        labels = pd.read_csv(labelsPath)
        walks = labels[labels["human"] == 1][["start_s", "end_s"]].to_numpy(dtype=float)
        times = data["fusionTime"].to_numpy()
        empty = isEmptyRecording(path, labels)
        data["empty"] = empty
        data["venue"] = isVenueRecording(path)
        data["session"] = path.name

        # Track rows too close to walk start or end labels
        near = np.zeros(len(data), dtype=bool)
        closeToWalk = np.zeros(len(data), dtype=bool)
        for start, end in walks:

            # For short walks, exclude nearby outside frames only
            # Keep the actual walk frames for learning fast motion
            if end - start < SHORT_WALK_S:

                near |= ((times >= start - BOUNDARY_S) & (times < start)) | \
                        ((times > end) & (times <= end + BOUNDARY_S))
                continue

            # Longer walks: exclude frames at uncertain entry/exit times
            near |= (np.abs(times - start) <= BOUNDARY_S) | (np.abs(times - end) <= BOUNDARY_S)
            closeToWalk |= ((times >= start - CLEAN_NEGATIVE_S) & (times < start)) | \
                           ((times > end) & (times <= end + CLEAN_NEGATIVE_S))
        data["nearBoundary"] = near


        # Exclude outbound walks labelled as empty
        # A person can still be visible while walking away
        outbound = np.zeros(len(data), dtype=bool)
        outboundPath = path / "labels(outbound).csv"
        if outboundPath.exists():
            outboundLabels = pd.read_csv(outboundPath)
            if "human" in outboundLabels.columns:
                outboundLabels = outboundLabels[outboundLabels["human"] == 1]
            for start, end in outboundLabels[["start_s", "end_s"]].itertuples(index=False):
                outbound |= (times >= start - BOUNDARY_S) & (times <= end + BOUNDARY_S)
        data["outbound"] = outbound


        # Read optional intervals confirmed to be empty
        known = np.zeros(len(data), dtype=bool)
        emptyPath = path / "empty.csv"
        if emptyPath.exists():
            for start, end in pd.read_csv(emptyPath)[["start_s", "end_s"]].itertuples(index=False):
                known |= (times >= start) & (times <= end)
        data["knownEmpty"] = known
        data["emptyTime"] = empty | known


        # Keep reliable training rows only
        # Drop uncertain boundaries, nearby negatives and outbound negatives
        data["train"] = ~near & ~(closeToWalk & (data["human"].to_numpy() == 0) & ~data["emptyTime"].to_numpy()) \
                        & ~(outbound & (data["human"].to_numpy() == 0))

        # Collect recording data and show its row count
        parts.append(data)
        progress(f"  {path}: {len(data)} rows{' (venue)' if data['venue'].iloc[0] else ''}"
                 f"{' (empty)' if empty else ''}{' (+empty.csv)' if known.any() else ''}"
                 f"{' (+outbound)' if outbound.any() else ''}")
    return pd.concat(parts, ignore_index=True)


# Find labelled human frames without detectable movement
# Remove these from training and moving-frame detection scoring
def markStillRows(data, progress=print):

    # Use venue empty frames to estimate normal sensor noise
    # Fall back to other empty recordings if needed
    venueEmpty = (data["venue"] & data["emptyTime"]).to_numpy()
    reference = venueEmpty if venueEmpty.any() else data["emptyTime"].to_numpy()
    if not reference.any():
        progress("  WARNING: no empty time to set the still-row limits; no rows marked still")

        # Save still flags and exclude them from training
        data["still"] = False
        return data

    # Set PIR and thermal motion limits from empty-scene percentiles
    pirLimit = float(np.percentile(data.loc[reference, "pirPeakToPeak"], STILL_PERCENTILE))
    fdLimit = float(np.percentile(data.loc[reference, "thermalFdWindowMaxEnergy"], STILL_PERCENTILE))

    # Still = human present, weak PIR, no blob, little thermal change
    still = (data["human"].to_numpy() == 1) & (data["pirPeakToPeak"].to_numpy() < pirLimit) & \
            (data["thermalNumberOfBlobs"].to_numpy() == 0) & (data["thermalFdWindowMaxEnergy"].to_numpy() < fdLimit)

    # Save still flags and exclude them from training
    data["still"] = still
    data["train"] = data["train"].to_numpy() & ~still
    human = int((data["human"] == 1).sum())
    progress(f"  still rows (no PIR peak-to-peak >= {pirLimit:.2f} V, no moving blob, frame difference < "
             f"{fdLimit:.3f}): {int(still.sum())} of {human} human rows left out")
    return data


# Build feature matrix X and human labels y
# Weight venue and empty examples by repeating their rows
def trainingRows(data, rows, features):
    X = data.loc[rows, features]
    y = data.loc[rows, "human"]

    # Select rows needing additional training weight
    focus = (data.loc[rows, "venue"] | data.loc[rows, "emptyTime"]).to_numpy()
    if FOCUS_WEIGHT > 1 and focus.any():
        X = pd.concat([X] + [X[focus]] * (FOCUS_WEIGHT - 1))
        y = pd.concat([y] + [y[focus]] * (FOCUS_WEIGHT - 1))
    return X, y


# Leave-one-venue-recording-out evaluation
# Each venue recording is scored by a model trained without it
def heldOutScores(data, features, progress=print):

    # Start with missing scores for non-venue rows
    scores = np.full(len(data), np.nan)
    sessions = data.loc[data["venue"], "session"].unique()

    # Hold out one session and train on the remaining eligible rows
    for k, session in enumerate(sessions, 1):
        test = (data["session"] == session).to_numpy()
        X, y = trainingRows(data, ~test & data["train"].to_numpy(), features)

        # Use the scikit-learn model for threshold evaluation
        model = buildModels()["Gradient Boosting"].fit(X, y)
        scores[test] = confidenceScores(model, data.loc[test, features])
        progress(f"  held-out {k}/{len(sessions)}: {session}")
    return scores


# Generate candidate three-state thresholds and hold times
def candidateSettings():
    for intruderEnter in np.arange(0.30, 0.951, 0.05):
        for gap in (0.05, 0.10, 0.15):
            for hold in (0.0, 1.0, 2.0, 3.0):

                # Set enter/exit gaps to reduce rapid state switching
                enter = round(float(intruderEnter), 3)
                exitLevel = round(enter - gap, 3)
                maybeEnter = round(exitLevel - 0.10, 3)
                maybeExit = round(maybeEnter - 0.10, 3)

                # Skip thresholds below zero probability
                if maybeExit < 0:
                    continue
                yield {"maybe_enter": maybeEnter, "maybe_exit": maybeExit, "intruder_enter": enter,
                       "intruder_exit": exitLevel, "hold_s": hold}


# Replay probability scores through the live state machine
# Reset alarm state at the start of each recording
def runStates(data, scores, settings):
    states = np.zeros(len(data), dtype=int)
    times = data["fusionTime"].to_numpy()

    # Process each recording independently in timestamp order
    for _, idx in data.groupby("recording", sort=False).indices.items():
        idx = idx[np.argsort(times[idx])]
        machine = StateMachine(settings)
        for i in idx:
            machine.update(scores[i], times[i])
            states[i] = machine.state
    return states


# Measure walk detections, alarm fragmentation and empty-scene alarms
def recordingResults(data, states):
    rows = []
    times = data["fusionTime"].to_numpy()
    emptyTime = data["emptyTime"].to_numpy()
    for recording, idx in data.groupby("recording", sort=False).indices.items():
        idx = idx[np.argsort(times[idx])]
        t = times[idx]
        st = states[idx]
        labels = pd.read_csv(Path(recording) / "labels.csv")
        empty = bool(data["empty"].iloc[idx[0]])
        walks = detected = 0
        delays = []
        segments = []
        intruder = st == INTRUDER

        # Evaluate each labelled walk only for nonempty recordings
        if not empty:
            for start, end in labels[labels["human"] == 1][["start_s", "end_s"]].itertuples(index=False):

                # Ignore walks beginning before processing was ready
                if start < t[0] + 1.0 or start > t[-1]:   
                    continue
                walks += 1

                # Count an INTRUDER alarm within the allowed response window
                window = (t >= start) & (t <= min(start + MAX_DELAY_S, end + BOUNDARY_S))
                hits = np.flatnonzero(window & intruder)
                if len(hits):
                    detected += 1
                    delays.append(t[hits[0]] - start)

                # Count separate INTRUDER segments within each walk
                inside = intruder[(t >= start) & (t <= end)]
                segments.append(int(inside.any()) + int(np.sum(inside[1:] & ~inside[:-1])) if inside.size else 0)

        # Estimate time per row, capped to avoid inflating gaps
        step = np.clip(np.diff(t, prepend=t[0]), 0.0, 1.0)
        emptyRows = emptyTime[idx]

        # Score frame detection only for moving humans away from boundaries
        moving = (data["human"].to_numpy()[idx] == 1) & ~data["nearBoundary"].to_numpy()[idx]
        if "still" in data:
            moving &= ~data["still"].to_numpy()[idx]

        # Store event, frame and empty-time metrics for this recording
        rows.append({
            "recording": Path(recording).name, "empty": empty, "walks": walks, "detected": detected,
            "delays": delays, "segments": segments, "empty_s": step[emptyRows].sum(),
            "alarmed_s": step[emptyRows & (st >= MAYBE_INTRUDER)].sum(),
            "intruder_s": step[emptyRows & intruder].sum(),
            "moving_frames": int(moving.sum()), "moving_intruder": int((moving & intruder).sum()),
        })
    return pd.DataFrame(rows)


# Combine per-recording metrics into one summary
def summarise(results):
    walks = int(results["walks"].sum())
    detected = int(results["detected"].sum())
    emptySeconds = float(results["empty_s"].sum())

    # Flatten detection delays and alarm-segment counts
    delays = [d for row in results["delays"] for d in row]
    segments = [s for row in results["segments"] for s in row]
    movingFrames = int(results["moving_frames"].sum())

    # Calculate walk Pd, moving-frame Pd and empty alarm percentages
    return {
        "frame_pd": 100.0 * results["moving_intruder"].sum() / movingFrames if movingFrames else float("nan"),
        "walks": walks, "detected": detected, "pd": 100.0 * detected / walks if walks else float("nan"),
        "delay": float(np.median(delays)) if delays else float("nan"),
        "segments": float(np.mean(segments)) if segments else float("nan"),
        "empty_min": emptySeconds / 60.0,
        "alarmed": 100.0 * results["alarmed_s"].sum() / emptySeconds if emptySeconds else float("nan"),
        "intruder": 100.0 * results["intruder_s"].sum() / emptySeconds if emptySeconds else float("nan"),
    }


# Evaluate every threshold candidate on held-out venue predictions
# Prefer reliable detection with low empty-scene alarm time
def chooseSettings(venueData, scores, progress=print):
    table = []

    # Simulate weaker predictions on a different day
    lowerScores = np.clip(np.asarray(scores) - ROBUST_SHIFT, 0.0, 1.0)

    # Run original and reduced scores through each alarm setting
    for settings in candidateSettings():
        summary = summarise(recordingResults(venueData, runStates(venueData, scores, settings)))
        lower = summarise(recordingResults(venueData, runStates(venueData, lowerScores, settings)))
        table.append({**settings, **summary, "detected_lower": lower["detected"], "pd_lower": lower["pd"]})
    table = pd.DataFrame(table)

    # Check whether false alarm performance can be measured
    noEmpty = table["alarmed"].isna().all()
    if noEmpty:
        progress("  WARNING: no empty venue time (record Empty sessions or add empty.csv); "
                 "choosing on detection only")

    # Keep settings meeting both normal and reduced-score targets
    ok = table[(table["pd"] >= MIN_DETECTION) & (table["pd_lower"] >= ROBUST_DETECTION)]
    if not ok.empty:

        # Prefer settings with fewer repeated alarms per walk
        unfragmented = ok[ok["segments"] <= MAX_SEGMENTS]
        if unfragmented.empty:
            progress(f"  WARNING: no setting has <= {MAX_SEGMENTS:g} INTRUDER segments per walk; ignoring that limit")
        else:
            ok = unfragmented

    # Fallback if no settings meet the preferred constraints
    if ok.empty:
        progress(f"  WARNING: no setting detects {ROBUST_DETECTION:g}% of venue walks with scores {ROBUST_SHIFT:g} "
                 f"lower; ignoring the lower-score check")
        ok = table[table["pd"] >= MIN_DETECTION]

    # Fallback if no settings meet the preferred constraints
    if ok.empty:

        progress(f"  WARNING: no setting reaches {MIN_DETECTION:g}% of venue walks; choosing the fewest "
                 f"false alarms within 2 points of the best detection ({table['pd'].max():.1f}%)")
        ok = table[table["pd"] >= table["pd"].max() - 2.0]
    if not noEmpty:

        # Allow settings close to the minimum false alarm time
        ok = ok[ok["alarmed"] <= ok["alarmed"].min() + ALARM_TOLERANCE]

    # Choose highest moving-frame Pd, then walk Pd
    # Break remaining ties by false alarms and fragmentation
    best = ok.sort_values(["frame_pd", "pd", "alarmed", "intruder", "segments"],
                          ascending=[False, False, True, True, True], na_position="last").iloc[0]

    # Save only the thresholds and hold time needed live
    keys = ("maybe_enter", "maybe_exit", "intruder_enter", "intruder_exit", "hold_s")
    settings = {k: float(best[k]) for k in keys}
    progress(f"  chosen: INTRUDER {settings['intruder_enter']}/{settings['intruder_exit']}, "
             f"MAYBE {settings['maybe_enter']}/{settings['maybe_exit']}, hold {settings['hold_s']:g} s -> "
             f"venue walks {int(best['detected'])}/{int(best['walks'])} ({int(best['detected_lower'])} with scores "
             f"{ROBUST_SHIFT:g} lower), frame Pd {best['frame_pd']:.1f}%, empty venue alarmed {best['alarmed']:.1f}%")
    return settings, table


# Format a short threshold trade-off comparison
def tradeOffLines(name, table, chosen):
    lines = [f"{name}: fewest false alarms at each venue detection level (held-out)",
             f"{'walks':>9s} {'detect %':>8s} {'frame Pd %':>10s} {'empty alarmed %':>15s} {'INTRUDER %':>10s} {'segs/walk':>9s} "
             f"{'delay s':>7s} {f'walks if scores {ROBUST_SHIFT:g} lower':>26s}  setting"]

    # Locate the selected setting in the results table
    isChosen = np.all([np.abs(table[k] - v) < 1e-9 for k, v in chosen.items()], axis=0)

    # Show representative low-alarm settings for each detection level
    best = table.sort_values(["alarmed", "intruder"]).drop_duplicates("detected").sort_values("pd", ascending=False)
    best = best.head(15)
    if not np.all([np.abs(best[k] - v) < 1e-9 for k, v in chosen.items()], axis=0).any():
        best = pd.concat([best, table[isChosen]]).sort_values(["pd", "alarmed"], ascending=[False, True])

    # Mark the chosen setting in the report
    for _, row in best.iterrows():
        mark = "  <- chosen" if all(abs(row[k] - v) < 1e-9 for k, v in chosen.items()) else ""
        lines.append(f"{int(row['detected']):>4d}/{int(row['walks']):<4d} {row['pd']:8.1f} {row['frame_pd']:10.1f} {row['alarmed']:15.1f} "
                     f"{row['intruder']:10.1f} {row['segments']:9.2f} {row['delay']:7.1f} "
                     f"{int(row['detected_lower']):>22d}/{int(row['walks']):<3d}  "
                     f"INTRUDER {row['intruder_enter']}/{row['intruder_exit']} MAYBE {row['maybe_enter']}/"
                     f"{row['maybe_exit']} hold {row['hold_s']:g}{mark}")
    return lines


# Baseline: PIR threshold crossing gives an INTRUDER state
def pirStates(data):
    return np.where(data["pirThresholdIntervalActive"].to_numpy() == 1, INTRUDER, NO_INTRUDER)


# Build side-by-side per-recording performance text
def reportLines(title, systems):
    """systems: {name: per-recording results}. Returns report lines."""
    lines = [title, "-" * len(title)]
    header = f"{'Venue recording':36s}" + "".join(f" | {name:>44s}" for name in systems)
    lines += [header, "-" * len(header)]

    # Compress each recording summary into a short report cell
    def cell(summary):
        parts = []
        if summary["walks"]:
            parts.append(f"walks {summary['detected']}/{summary['walks']}")
            if not np.isnan(summary["frame_pd"]):
                parts.append(f"frame Pd {summary['frame_pd']:.0f}%")
            if not np.isnan(summary["segments"]):
                parts.append(f"{summary['segments']:.1f} seg")
        if summary["empty_min"]:
            parts.append(f"empty {summary['empty_min']:.0f}min alarmed {summary['alarmed']:.1f}%")
        return " ".join(parts) if parts else "-"

    # Write each venue recording followed by the venue total
    first = next(iter(systems.values()))
    for i, recording in enumerate(first["recording"]):
        lines.append(f"{recording[:36]:36s}" + "".join(
            f" | {cell(summarise(results.iloc[[i]])):>44s}" for results in systems.values()))
    lines.append("-" * len(header))
    lines.append(f"{'VENUE TOTAL':36s}" + "".join(f" | {cell(summarise(results)):>44s}" for results in systems.values()))
    return lines


# Run the full demo training and evaluation workflow
# Export a fused model and a PIR-only backup
def runDemoTraining(runName, folder, window, modelPath, backupPath, progress=print):

    # Create a new output folder and prepare all training rows
    folder.mkdir(parents=True, exist_ok=False)
    progress(f"Building features ({window:g} s window) for every demo recording...")
    data = buildDemoData(window, progress)
    data = markStillRows(data, progress)

    # Keep venue rows for held-out performance reporting
    venue = data["venue"].to_numpy()
    if not venue.any():
        raise ValueError(f"No venue recordings found (names starting with {VENUE_PREFIXES} outside Datasets/old)")
    venueData = data[venue].reset_index(drop=True)
    progress(f"  {len(data)} rows from {data['recording'].nunique()} recordings; "
             f"{data.loc[venue, 'recording'].nunique()} venue recordings "
             f"({int(venueData['emptyTime'].sum())} empty-time rows); "
             f"{int(data['train'].sum())} training rows before weighting")

    # Train fused and PIR-only versions using the same procedure
    outputs = {}
    tradeOffs = {}
    for name, features, path, tag in (("Fused system", getFusedFeatureNames(), modelPath, "fused"),
                                      ("PIR-only backup", getHistoryFeatureNames(), backupPath, "pir_backup")):
        progress(f"\n{name}: held-out scores for each venue recording...")

        # Get held-out venue scores for this feature set
        scores = heldOutScores(data, features, progress)[venue]

        # Apply the same strong-PIR override as the live system
        scores = applyPirStrongRule(scores, venueData["pirPeakToPeak"].to_numpy())

        # Export probabilities for later inspection
        pd.DataFrame({"recording": venueData["session"], "time_s": venueData["fusionTime"],
                      "human": venueData["human"], "emptyTime": venueData["emptyTime"], "score": scores}) \
            .to_csv(folder / f"demo_scores_{tag}.csv", index=False)
        progress(f"{name}: choosing alarm settings (>= {MIN_DETECTION:g}% of venue walks, >= {ROBUST_DETECTION:g}% "
                 f"with scores {ROBUST_SHIFT:g} lower, fewest false alarms, then most walks)...")

        # Tune alarm thresholds using held-out venue performance
        settings, table = chooseSettings(venueData, scores, progress)
        table.to_csv(folder / f"demo_tradeoff_{tag}.csv", index=False)
        progress(f"{name}: training the first-principles Gradient Boosting on all training rows...")

        # Train the final custom Gradient Boosting on all eligible rows
        X, y = trainingRows(data, data["train"].to_numpy(), features)
        model = GradientBoosting().fit(X.to_numpy(dtype=float), y.to_numpy(), features, progress=progress)

        # Export the trained model together with its alarm settings
        model.save(path, window, settings)
        progress(f"Saved {path}")
        outputs[name] = (settings, recordingResults(venueData, runStates(venueData, scores, settings)))
        tradeOffs[name] = tradeOffLines(name, table, settings)

    # Compare fused, PIR backup and simple PIR threshold systems
    systems = {"Thresholded PIR": recordingResults(venueData, pirStates(venueData))}
    systems.update({name: results for name, (_, results) in outputs.items()})

    # Write training assumptions and venue results to a text report
    lines = [f"DEMO TRAINING REPORT: {runName}",
             f"Window {window:g} s. Trained on every recording under: {', '.join(DEMO_DATASETS)}",
             f"Venue recordings: names starting with {', '.join(VENUE_PREFIXES)} (outside Datasets/old).",
             f"Training: rows within {BOUNDARY_S:g} s of a label start/end dropped; non-walk rows within "
             f"{CLEAN_NEGATIVE_S:g} s of a walk dropped (clean negatives); venue and empty rows x{FOCUS_WEIGHT}.",
             "Every number below is HELD-OUT: each venue recording is scored by a model that never saw it.",
             f"Walk detected = INTRUDER within {MAX_DELAY_S:g} s of entry. seg = INTRUDER segments per walk "
             "(1.0 = no fragmentation). Empty alarmed = MAYBE or INTRUDER time in empty venue time "
             "(Empty/Distractor/Negative recordings and empty.csv stretches).",
             f"Rule: settings detecting >= {MIN_DETECTION:g}% of venue walks and >= {ROBUST_DETECTION:g}% with every "
             f"score {ROBUST_SHIFT:g} lower and <= {MAX_SEGMENTS:g} INTRUDER segments per walk; of those within {ALARM_TOLERANCE:g} points of the fewest false alarms, the one with the "
             f"highest frame Pd.",
             f"Thermopile threshold: {_thresholdText()}.",
             f"Motion only: human rows with no motion on either sensor are left out of training and frame Pd; "
             f"PIR strong-signal rule: INTRUDER when PIR peak-to-peak > {PIR_STRONG_V:g} V.",
             "frame Pd = INTRUDER frames / moving human frames (not still, not within the label boundary).", ""]

    # Add selected alarm thresholds for each model
    for name, (settings, _) in outputs.items():
        lines.append(f"{name} alarm settings: INTRUDER on {settings['intruder_enter']} / off {settings['intruder_exit']}, "
                     f"MAYBE on {settings['maybe_enter']} / off {settings['maybe_exit']}, hold {settings['hold_s']:g} s")
    lines.append("")

    # Append venue metrics and the threshold trade-off tables
    lines += reportLines("VENUE RESULTS (held-out)", systems)
    for name in tradeOffs:
        lines += [""] + tradeOffs[name]
    lines += ["", "Full trade-off tables: demo_tradeoff_fused.csv / demo_tradeoff_pir_backup.csv; "
                  "held-out scores: demo_scores_*.csv"]
    text = "\n".join(lines) + "\n"
    (folder / "demo_report.txt").write_text(text, encoding="utf-8")
    return text


# Describe the active thermal foreground threshold mode
def _thresholdText():
    from Thermopile import processing
    if getattr(processing, "USE_FIXED_THRESHOLD", False):
        return f"fixed Z {processing.FIXED_Z_RAW:g} (average detector {processing.FIXED_Z_AVG:g})"
    return "temperature table"
