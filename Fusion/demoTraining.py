"""Demo training: one model for the exam venue (and backup site), fewest false alarms first.

  python offlineRun.py --demo --name Demo_Wed

1. Loads every recording under DEMO_DATASETS (no validation/testing split).
2. Training rows that are NOT used:
   - rows within BOUNDARY_S of a label start/end (live SPACE labels lag the real entry/exit)
   - "clean negatives": non-walk rows within CLEAN_NEGATIVE_S before/after a labelled walk
     in walk recordings (the walker is just outside the zone or walking back: a person is
     there but labelled "nobody", which taught the model that person-like signals are
     ambiguous and caused false alarms in empty scenes). Empty recordings and empty.csv
     stretches are always kept.
   - rows inside a labels(outbound).csv walk (a person walking away, labelled "nobody")
3. Venue and empty rows are repeated FOCUS_WEIGHT times so the exam venue and the
   "nobody there" data are not drowned out by the older sites.
4. Leave-one-recording-out on the VENUE recordings only: each venue recording is scored by a
   scikit-learn Gradient Boosting (same settings) trained on everything else.
5. Alarm settings (three-state thresholds + hold) are chosen on those held-out venue scores:
   only settings that detect at least MIN_DETECTION % of the venue walks, and still at least
   ROBUST_DETECTION % with every score ROBUST_SHIFT lower; then, among those within
   ALARM_TOLERANCE points of the fewest false alarms (alarmed time in empty venue time), the one
   that detects the most walks. Only settings with at most MAX_SEGMENTS INTRUDER segments per walk
   are considered (falls back to all settings if none qualifies).
6. The first-principles Gradient Boosting is trained on all training rows and exported with
   the chosen settings. Steps 4-6 are repeated with only the PIR features for the backup.
7. demo_report.txt (venue only), demo_tradeoff_*.csv (every setting) and demo_scores_*.csv
   (held-out scores) are written so another operating point can be chosen without retraining.

Venue recordings: recordings outside Datasets/old whose name starts with one of VENUE_PREFIXES
(exam venue "Uni...", backup site "Backup..."). Recordings whose name contains Empty,
Distractor or Negative (or that have no labelled walks) are empty recordings. A recording may
also contain empty.csv (start_s,end_s): stretches known to have nobody in view.
"""
from pathlib import Path
import glob

import numpy as np
import pandas as pd

from Fusion.dataset import buildRecordingDataset, getFusedFeatureNames
from Fusion.gradientBoosting import GradientBoosting
from Fusion.models import buildModels, confidenceScores
from Fusion.realtimeModel import StateMachine, INTRUDER, MAYBE_INTRUDER, NO_INTRUDER
from PIR.processing import getHistoryFeatureNames

# Config
DEMO_DATASETS = [
    "Datasets/demo",
    "Datasets/training",
    "Datasets/validation",
    "Datasets/testing",
    "Datasets/old",          # every older recording (training, validation and testing)
]
DEMO_WINDOW_S = 2.0
VENUE_PREFIXES = ("Uni", "Backup")   # exam venue and backup site (outside Datasets/old)
EMPTY_KEYWORDS = ("empty", "distractor", "negative")
BOUNDARY_S = 1.0             # label grace: key-press lag around each entry/exit
CLEAN_NEGATIVE_S = 5.0       # non-walk rows this close to a walk are not used for training
FOCUS_WEIGHT = 3             # venue and empty rows count this many times in training
MIN_DETECTION = 95.0         # % of venue walks that must be detected; then fewest false alarms
# Robust choice (10 Nov): a setting must still detect ROBUST_DETECTION % when every score is ROBUST_SHIFT
# lower (a new day scores a bit differently), and among settings within ALARM_TOLERANCE points of the
# fewest false alarms the one detecting the most walks is taken (the empty venue time is short, so
# smaller differences in false alarms are noise). Demo_v3 without this picked INTRUDER 0.9, which
# detected 183/191 walks but 0/191 with scores 0.1 lower.
ROBUST_SHIFT = 0.05
ROBUST_DETECTION = 90.0
ALARM_TOLERANCE = 1.0
# Fragmentation (10 Nov): only settings with at most MAX_SEGMENTS INTRUDER segments per venue walk are
# considered (on Demo_v3 this gives hold 2 s: 2.22 -> 1.63 segments, same false-alarm events, each 2 s longer)
MAX_SEGMENTS = 1.7
MAX_DELAY_S = 10.0           # a walk counts as detected if INTRUDER within 10 s of entry


# Every Recording Folder Under The Demo Datasets (folders with metadata.json)
def demoRecordings(datasets=DEMO_DATASETS):
    folders = set()
    for pattern in datasets:
        for root in glob.glob(pattern):
            folders.update(path.parent for path in Path(root).rglob("metadata.json"))
    return sorted(folders)


def isEmptyRecording(path, labels):
    name = Path(path).name.lower()
    return any(word in name for word in EMPTY_KEYWORDS) or not (labels["human"] == 1).any()


def isVenueRecording(path):
    path = Path(path)
    return path.name.startswith(VENUE_PREFIXES) and "old" not in path.parts


# Builds One Fused Row Per Thermal Frame For Every Recording, With Training/Scoring Flags
def buildDemoData(window, progress=print):
    parts = []
    for path in demoRecordings():
        labelsPath = path / "labels.csv"
        if not labelsPath.exists():
            progress(f"  skipping {path}: no labels.csv")
            continue
        data = buildRecordingDataset(path, window)
        if data.empty:
            continue
        labels = pd.read_csv(labelsPath)
        walks = labels[labels["human"] == 1][["start_s", "end_s"]].to_numpy(dtype=float)
        times = data["fusionTime"].to_numpy()
        empty = isEmptyRecording(path, labels)
        data["empty"] = empty
        data["venue"] = isVenueRecording(path)
        data["session"] = path.name

        near = np.zeros(len(data), dtype=bool)
        closeToWalk = np.zeros(len(data), dtype=bool)
        for start, end in walks:
            near |= (np.abs(times - start) <= BOUNDARY_S) | (np.abs(times - end) <= BOUNDARY_S)
            closeToWalk |= ((times >= start - CLEAN_NEGATIVE_S) & (times < start)) | \
                           ((times > end) & (times <= end + CLEAN_NEGATIVE_S))
        data["nearBoundary"] = near

        # Person walking away from the sensor (optional labels(outbound).csv): not "nobody"
        outbound = np.zeros(len(data), dtype=bool)
        outboundPath = path / "labels(outbound).csv"
        if outboundPath.exists():
            outboundLabels = pd.read_csv(outboundPath)
            if "human" in outboundLabels.columns:
                outboundLabels = outboundLabels[outboundLabels["human"] == 1]
            for start, end in outboundLabels[["start_s", "end_s"]].itertuples(index=False):
                outbound |= (times >= start - BOUNDARY_S) & (times <= end + BOUNDARY_S)
        data["outbound"] = outbound

        # Known-empty stretches inside a walk recording (optional empty.csv)
        known = np.zeros(len(data), dtype=bool)
        emptyPath = path / "empty.csv"
        if emptyPath.exists():
            for start, end in pd.read_csv(emptyPath)[["start_s", "end_s"]].itertuples(index=False):
                known |= (times >= start) & (times <= end)
        data["knownEmpty"] = known
        data["emptyTime"] = empty | known

        # Rows used for training: no label-boundary rows, no "person just outside the zone" rows
        data["train"] = ~near & ~(closeToWalk & (data["human"].to_numpy() == 0) & ~data["emptyTime"].to_numpy()) \
                        & ~(outbound & (data["human"].to_numpy() == 0))
        parts.append(data)
        progress(f"  {path}: {len(data)} rows{' (venue)' if data['venue'].iloc[0] else ''}"
                 f"{' (empty)' if empty else ''}{' (+empty.csv)' if known.any() else ''}"
                 f"{' (+outbound)' if outbound.any() else ''}")
    return pd.concat(parts, ignore_index=True)


# Training Matrix With Venue And Empty Rows Repeated FOCUS_WEIGHT Times
def trainingRows(data, rows, features):
    X = data.loc[rows, features]
    y = data.loc[rows, "human"]
    focus = (data.loc[rows, "venue"] | data.loc[rows, "emptyTime"]).to_numpy()
    if FOCUS_WEIGHT > 1 and focus.any():
        X = pd.concat([X] + [X[focus]] * (FOCUS_WEIGHT - 1))
        y = pd.concat([y] + [y[focus]] * (FOCUS_WEIGHT - 1))
    return X, y


# Held-Out Human Probabilities For Every Venue Recording (scikit-learn, same settings)
def heldOutScores(data, features, progress=print):
    scores = np.full(len(data), np.nan)
    sessions = data.loc[data["venue"], "session"].unique()
    for k, session in enumerate(sessions, 1):
        test = (data["session"] == session).to_numpy()
        X, y = trainingRows(data, ~test & data["train"].to_numpy(), features)
        model = buildModels()["Gradient Boosting"].fit(X, y)
        scores[test] = confidenceScores(model, data.loc[test, features])
        progress(f"  held-out {k}/{len(sessions)}: {session}")
    return scores


# Three-State Settings Searched (same structure as the hysteresis state machine)
def candidateSettings():
    for intruderEnter in np.arange(0.30, 0.951, 0.05):
        for gap in (0.05, 0.10, 0.15):
            for hold in (0.0, 1.0, 2.0, 3.0):
                enter = round(float(intruderEnter), 3)
                exitLevel = round(enter - gap, 3)
                maybeEnter = round(exitLevel - 0.10, 3)
                maybeExit = round(maybeEnter - 0.10, 3)
                if maybeExit < 0:
                    continue
                yield {"maybe_enter": maybeEnter, "maybe_exit": maybeExit, "intruder_enter": enter,
                       "intruder_exit": exitLevel, "hold_s": hold}


# Runs The Live State Machine Over Every Recording In Time Order
def runStates(data, scores, settings):
    states = np.zeros(len(data), dtype=int)
    times = data["fusionTime"].to_numpy()
    for _, idx in data.groupby("recording", sort=False).indices.items():
        idx = idx[np.argsort(times[idx])]
        machine = StateMachine(settings)
        for i in idx:
            machine.update(scores[i], times[i])
            states[i] = machine.state
    return states


# Walk Detection, Fragmentation And Empty-Time Alarms Per Recording
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
        if not empty:
            for start, end in labels[labels["human"] == 1][["start_s", "end_s"]].itertuples(index=False):
                if start < t[0] + 1.0 or start > t[-1]:   # walk starts before the system was ready
                    continue
                walks += 1
                window = (t >= start) & (t <= min(start + MAX_DELAY_S, end + BOUNDARY_S))
                hits = np.flatnonzero(window & intruder)
                if len(hits):
                    detected += 1
                    delays.append(t[hits[0]] - start)
                inside = intruder[(t >= start) & (t <= end)]
                segments.append(int(inside.any()) + int(np.sum(inside[1:] & ~inside[:-1])) if inside.size else 0)
        step = np.clip(np.diff(t, prepend=t[0]), 0.0, 1.0)
        emptyRows = emptyTime[idx]
        rows.append({
            "recording": Path(recording).name, "empty": empty, "walks": walks, "detected": detected,
            "delays": delays, "segments": segments, "empty_s": step[emptyRows].sum(),
            "alarmed_s": step[emptyRows & (st >= MAYBE_INTRUDER)].sum(),
            "intruder_s": step[emptyRows & intruder].sum(),
        })
    return pd.DataFrame(rows)


def summarise(results):
    walks = int(results["walks"].sum())
    detected = int(results["detected"].sum())
    emptySeconds = float(results["empty_s"].sum())
    delays = [d for row in results["delays"] for d in row]
    segments = [s for row in results["segments"] for s in row]
    return {
        "walks": walks, "detected": detected, "pd": 100.0 * detected / walks if walks else float("nan"),
        "delay": float(np.median(delays)) if delays else float("nan"),
        "segments": float(np.mean(segments)) if segments else float("nan"),
        "empty_min": emptySeconds / 60.0,
        "alarmed": 100.0 * results["alarmed_s"].sum() / emptySeconds if emptySeconds else float("nan"),
        "intruder": 100.0 * results["intruder_s"].sum() / emptySeconds if emptySeconds else float("nan"),
    }


# Every Setting Scored On The Venue; Robust To Lower Scores, Then Fewest False Alarms, Then Most Walks
def chooseSettings(venueData, scores, progress=print):
    table = []
    lowerScores = np.clip(np.asarray(scores) - ROBUST_SHIFT, 0.0, 1.0)
    for settings in candidateSettings():
        summary = summarise(recordingResults(venueData, runStates(venueData, scores, settings)))
        lower = summarise(recordingResults(venueData, runStates(venueData, lowerScores, settings)))
        table.append({**settings, **summary, "detected_lower": lower["detected"], "pd_lower": lower["pd"]})
    table = pd.DataFrame(table)
    noEmpty = table["alarmed"].isna().all()
    if noEmpty:
        progress("  WARNING: no empty venue time (record Empty sessions or add empty.csv); "
                 "choosing on detection only")
    ok = table[(table["pd"] >= MIN_DETECTION) & (table["pd_lower"] >= ROBUST_DETECTION)]
    if not ok.empty:
        unfragmented = ok[ok["segments"] <= MAX_SEGMENTS]
        if unfragmented.empty:
            progress(f"  WARNING: no setting has <= {MAX_SEGMENTS:g} INTRUDER segments per walk; ignoring that limit")
        else:
            ok = unfragmented
    if ok.empty:
        progress(f"  WARNING: no setting detects {ROBUST_DETECTION:g}% of venue walks with scores {ROBUST_SHIFT:g} "
                 f"lower; ignoring the lower-score check")
        ok = table[table["pd"] >= MIN_DETECTION]
    if ok.empty:
        # Still false alarms first: fewest false alarms within 2 points of the best detection reached
        progress(f"  WARNING: no setting reaches {MIN_DETECTION:g}% of venue walks; choosing the fewest "
                 f"false alarms within 2 points of the best detection ({table['pd'].max():.1f}%)")
        ok = table[table["pd"] >= table["pd"].max() - 2.0]
    if not noEmpty:
        # Differences in false alarms smaller than ALARM_TOLERANCE points are noise: take the most walks
        ok = ok[ok["alarmed"] <= ok["alarmed"].min() + ALARM_TOLERANCE]
    best = ok.sort_values(["pd", "alarmed", "intruder", "segments"], ascending=[False, True, True, True],
                          na_position="last").iloc[0]
    keys = ("maybe_enter", "maybe_exit", "intruder_enter", "intruder_exit", "hold_s")
    settings = {k: float(best[k]) for k in keys}
    progress(f"  chosen: INTRUDER {settings['intruder_enter']}/{settings['intruder_exit']}, "
             f"MAYBE {settings['maybe_enter']}/{settings['maybe_exit']}, hold {settings['hold_s']:g} s -> "
             f"venue walks {int(best['detected'])}/{int(best['walks'])} ({int(best['detected_lower'])} with scores "
             f"{ROBUST_SHIFT:g} lower), empty venue alarmed {best['alarmed']:.1f}%")
    return settings, table


# Trade-Off Lines: fewest false alarms reachable at each venue detection level
def tradeOffLines(name, table, chosen):
    lines = [f"{name}: fewest false alarms at each venue detection level (held-out)",
             f"{'walks':>9s} {'detect %':>8s} {'empty alarmed %':>15s} {'INTRUDER %':>10s} {'segs/walk':>9s} "
             f"{'delay s':>7s} {f'walks if scores {ROBUST_SHIFT:g} lower':>26s}  setting"]
    isChosen = np.all([np.abs(table[k] - v) < 1e-9 for k, v in chosen.items()], axis=0)
    best = table.sort_values(["alarmed", "intruder"]).drop_duplicates("detected").sort_values("pd", ascending=False)
    best = best.head(15)
    if not np.all([np.abs(best[k] - v) < 1e-9 for k, v in chosen.items()], axis=0).any():
        best = pd.concat([best, table[isChosen]]).sort_values(["pd", "alarmed"], ascending=[False, True])
    for _, row in best.iterrows():
        mark = "  <- chosen" if all(abs(row[k] - v) < 1e-9 for k, v in chosen.items()) else ""
        lines.append(f"{int(row['detected']):>4d}/{int(row['walks']):<4d} {row['pd']:8.1f} {row['alarmed']:15.1f} "
                     f"{row['intruder']:10.1f} {row['segments']:9.2f} {row['delay']:7.1f} "
                     f"{int(row['detected_lower']):>22d}/{int(row['walks']):<3d}  "
                     f"INTRUDER {row['intruder_enter']}/{row['intruder_exit']} MAYBE {row['maybe_enter']}/"
                     f"{row['maybe_exit']} hold {row['hold_s']:g}{mark}")
    return lines


def pirStates(data):
    return np.where(data["pirThresholdIntervalActive"].to_numpy() == 1, INTRUDER, NO_INTRUDER)


def reportLines(title, systems):
    """systems: {name: per-recording results}. Returns report lines."""
    lines = [title, "-" * len(title)]
    header = f"{'Venue recording':36s}" + "".join(f" | {name:>44s}" for name in systems)
    lines += [header, "-" * len(header)]

    def cell(summary):
        parts = []
        if summary["walks"]:
            parts.append(f"walks {summary['detected']}/{summary['walks']}")
            if not np.isnan(summary["segments"]):
                parts.append(f"{summary['segments']:.1f} seg")
        if summary["empty_min"]:
            parts.append(f"empty {summary['empty_min']:.0f}min alarmed {summary['alarmed']:.1f}%")
        return " ".join(parts) if parts else "-"

    first = next(iter(systems.values()))
    for i, recording in enumerate(first["recording"]):
        lines.append(f"{recording[:36]:36s}" + "".join(
            f" | {cell(summarise(results.iloc[[i]])):>44s}" for results in systems.values()))
    lines.append("-" * len(header))
    lines.append(f"{'VENUE TOTAL':36s}" + "".join(f" | {cell(summarise(results)):>44s}" for results in systems.values()))
    return lines


# Full Demo Run: settings, final first-principles models, report
def runDemoTraining(runName, folder, window, modelPath, backupPath, progress=print):
    folder.mkdir(parents=True, exist_ok=False)
    progress(f"Building features ({window:g} s window) for every demo recording...")
    data = buildDemoData(window, progress)
    venue = data["venue"].to_numpy()
    if not venue.any():
        raise ValueError(f"No venue recordings found (names starting with {VENUE_PREFIXES} outside Datasets/old)")
    venueData = data[venue].reset_index(drop=True)
    progress(f"  {len(data)} rows from {data['recording'].nunique()} recordings; "
             f"{data.loc[venue, 'recording'].nunique()} venue recordings "
             f"({int(venueData['emptyTime'].sum())} empty-time rows); "
             f"{int(data['train'].sum())} training rows before weighting")

    outputs = {}
    tradeOffs = {}
    for name, features, path, tag in (("Fused system", getFusedFeatureNames(), modelPath, "fused"),
                                      ("PIR-only backup", getHistoryFeatureNames(), backupPath, "pir_backup")):
        progress(f"\n{name}: held-out scores for each venue recording...")
        scores = heldOutScores(data, features, progress)[venue]
        pd.DataFrame({"recording": venueData["session"], "time_s": venueData["fusionTime"],
                      "human": venueData["human"], "emptyTime": venueData["emptyTime"], "score": scores}) \
            .to_csv(folder / f"demo_scores_{tag}.csv", index=False)
        progress(f"{name}: choosing alarm settings (>= {MIN_DETECTION:g}% of venue walks, >= {ROBUST_DETECTION:g}% "
                 f"with scores {ROBUST_SHIFT:g} lower, fewest false alarms, then most walks)...")
        settings, table = chooseSettings(venueData, scores, progress)
        table.to_csv(folder / f"demo_tradeoff_{tag}.csv", index=False)
        progress(f"{name}: training the first-principles Gradient Boosting on all training rows...")
        X, y = trainingRows(data, data["train"].to_numpy(), features)
        model = GradientBoosting().fit(X.to_numpy(dtype=float), y.to_numpy(), features, progress=progress)
        model.save(path, window, settings)
        progress(f"Saved {path}")
        outputs[name] = (settings, recordingResults(venueData, runStates(venueData, scores, settings)))
        tradeOffs[name] = tradeOffLines(name, table, settings)

    systems = {"Thresholded PIR": recordingResults(venueData, pirStates(venueData))}
    systems.update({name: results for name, (_, results) in outputs.items()})

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
             f"score {ROBUST_SHIFT:g} lower and <= {MAX_SEGMENTS:g} INTRUDER segments per walk; of those within {ALARM_TOLERANCE:g} points of the fewest false alarms, the one detecting the "
             f"most walks.",
             f"Thermopile threshold: {_thresholdText()}.", ""]
    for name, (settings, _) in outputs.items():
        lines.append(f"{name} alarm settings: INTRUDER on {settings['intruder_enter']} / off {settings['intruder_exit']}, "
                     f"MAYBE on {settings['maybe_enter']} / off {settings['maybe_exit']}, hold {settings['hold_s']:g} s")
    lines.append("")
    lines += reportLines("VENUE RESULTS (held-out)", systems)
    for name in tradeOffs:
        lines += [""] + tradeOffs[name]
    lines += ["", "Full trade-off tables: demo_tradeoff_fused.csv / demo_tradeoff_pir_backup.csv; "
                  "held-out scores: demo_scores_*.csv"]
    text = "\n".join(lines) + "\n"
    (folder / "demo_report.txt").write_text(text, encoding="utf-8")
    return text


def _thresholdText():
    from Thermopile import processing
    if getattr(processing, "USE_FIXED_THRESHOLD", False):
        return f"fixed Z {processing.FIXED_Z_RAW:g} (average detector {processing.FIXED_Z_AVG:g})"
    return "temperature table"
