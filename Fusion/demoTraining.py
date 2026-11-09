"""Demo training: one model trained on every recording, for the best live performance.

  python offlineRun.py --demo --name Demo_Tue

1. Loads every recording under DEMO_DATASETS (no validation/testing split).
2. Rows within BOUNDARY_S of a label start/end are dropped from training
   (live SPACE labels lag the real entry/exit).
3. Leave-one-session-out: every session is scored by a scikit-learn Gradient
   Boosting (same settings) trained on all the OTHER sessions.
4. Alarm settings (three-state thresholds + hold) are chosen on those held-out
   scores: highest walk detection first, then (within PD_TOLERANCE percentage
   points of it) the least alarmed time in the empty recordings.
5. The first-principles Gradient Boosting is trained on all rows and exported
   with the chosen settings. Steps 3-5 are repeated with only the PIR features
   for the PIR-only backup.
6. demo_report.txt lists every session and group: walks detected, time to
   INTRUDER, and alarmed time when empty, next to the thresholded PIR.

Recording names should follow Site_Day_Type_Weather (e.g. Venue_Tue_Empty_Hot33C).
Recordings whose name contains Empty, Distractor or Negative (or that have no
labelled walks) are the false-alarm recordings: nobody is labelled inside the zone.
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
    "Datasets/old/training/Uni*",
]
DEMO_WINDOW_S = 2.0
EMPTY_KEYWORDS = ("empty", "distractor", "negative")
BOUNDARY_S = 1.0             # label grace: key-press lag around each entry/exit
MAX_DELAY_S = 10.0           # a walk counts as detected if INTRUDER within 10 s of entry
PD_TOLERANCE = 1.0           # % points of detection traded for fewer false alarms
SPLIT_LONGER_THAN_S = 15 * 60  # long recordings are held out in thirds


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


# Builds One Fused Row Per Thermal Frame For Every Recording, With Session And Grace Flags
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
        data["empty"] = isEmptyRecording(path, labels)
        data["session"] = path.name
        times = data["fusionTime"].to_numpy()
        if times[-1] - times[0] > SPLIT_LONGER_THAN_S:
            third = np.minimum(((times - times[0]) / (times[-1] - times[0] + 1e-9) * 3).astype(int), 2)
            data["session"] = [f"{path.name}#{'ABC'[k]}" for k in third]
        near = np.zeros(len(data), dtype=bool)
        for start, end in labels[labels["human"] == 1][["start_s", "end_s"]].itertuples(index=False):
            near |= (np.abs(times - start) <= BOUNDARY_S) | (np.abs(times - end) <= BOUNDARY_S)
        data["nearBoundary"] = near
        parts.append(data)
        progress(f"  {path}: {len(data)} rows{' (empty)' if data['empty'].iloc[0] else ''}")
    return pd.concat(parts, ignore_index=True)


# Leave-One-Session-Out Human Probabilities (scikit-learn, same settings as the final model)
def heldOutScores(data, features, progress=print):
    scores = np.full(len(data), np.nan)
    sessions = data["session"].unique()
    for k, session in enumerate(sessions, 1):
        test = (data["session"] == session).to_numpy()
        train = ~test & ~data["nearBoundary"].to_numpy()
        model = buildModels()["Gradient Boosting"].fit(data.loc[train, features], data.loc[train, "human"])
        scores[test] = confidenceScores(model, data.loc[test, features])
        progress(f"  held-out {k}/{len(sessions)}: {session}")
    return scores


# Three-State Settings Searched (same structure as the hysteresis state machine)
def candidateSettings():
    for intruderEnter in np.arange(0.30, 0.901, 0.05):
        for gap in (0.05, 0.10):
            for hold in (0.0, 1.0, 2.0):
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


# Walk Detection And Empty-Scene Alarm Time Per Recording
def recordingResults(data, states):
    rows = []
    times = data["fusionTime"].to_numpy()
    for recording, idx in data.groupby("recording", sort=False).indices.items():
        idx = idx[np.argsort(times[idx])]
        t = times[idx]
        st = states[idx]
        labels = pd.read_csv(Path(recording) / "labels.csv")
        empty = bool(data["empty"].iloc[idx[0]])
        walks = detected = 0
        delays = []
        if not empty:
            for start, end in labels[labels["human"] == 1][["start_s", "end_s"]].itertuples(index=False):
                if start < t[0] + 1.0 or start > t[-1]:   # walk starts before the system was ready
                    continue
                walks += 1
                window = (t >= start) & (t <= min(start + MAX_DELAY_S, end + BOUNDARY_S))
                hits = np.flatnonzero(window & (st == INTRUDER))
                if len(hits):
                    detected += 1
                    delays.append(t[hits[0]] - start)
        step = np.clip(np.diff(t, prepend=t[0]), 0.0, 1.0)
        rows.append({
            "recording": Path(recording).name, "empty": empty, "walks": walks, "detected": detected,
            "delays": delays, "empty_s": step.sum() if empty else 0.0,
            "alarmed_s": step[st >= MAYBE_INTRUDER].sum() if empty else 0.0,
            "intruder_s": step[st == INTRUDER].sum() if empty else 0.0,
        })
    return pd.DataFrame(rows)


def summarise(results):
    walks = int(results["walks"].sum())
    detected = int(results["detected"].sum())
    emptySeconds = float(results["empty_s"].sum())
    delays = [d for row in results["delays"] for d in row]
    return {
        "walks": walks, "detected": detected, "pd": 100.0 * detected / walks if walks else float("nan"),
        "delay": float(np.median(delays)) if delays else float("nan"),
        "empty_min": emptySeconds / 60.0,
        "alarmed": 100.0 * results["alarmed_s"].sum() / emptySeconds if emptySeconds else float("nan"),
        "intruder": 100.0 * results["intruder_s"].sum() / emptySeconds if emptySeconds else float("nan"),
        # Mean over the empty recordings, so one long recording cannot dominate
        "alarmed_mean": _perRecordingMean(results, "alarmed_s"),
        "intruder_mean": _perRecordingMean(results, "intruder_s"),
    }


def _perRecordingMean(results, column):
    empty = results[results["empty_s"] > 0]
    if empty.empty:
        return float("nan")
    return float(np.mean(100.0 * empty[column] / empty["empty_s"]))


# Highest Detection First, Then The Least Alarmed Time In The Empty Recordings
# (mean over the empty recordings). Returns the chosen settings and every trial.
def chooseSettings(data, scores, progress=print):
    trials = []
    for settings in candidateSettings():
        summary = summarise(recordingResults(data, runStates(data, scores, settings)))
        trials.append((settings, summary))
    bestPd = max(summary["pd"] for _, summary in trials)
    close = [trial for trial in trials if trial[1]["pd"] >= bestPd - PD_TOLERANCE]
    close.sort(key=lambda trial: (np.nan_to_num(trial[1]["alarmed_mean"], nan=0.0),
                                  np.nan_to_num(trial[1]["intruder_mean"], nan=0.0),
                                  -trial[1]["pd"], trial[1]["delay"]))
    settings, summary = close[0]
    progress(f"  chosen: INTRUDER {settings['intruder_enter']}/{settings['intruder_exit']}, "
             f"MAYBE {settings['maybe_enter']}/{settings['maybe_exit']}, hold {settings['hold_s']:g} s -> "
             f"walks {summary['detected']}/{summary['walks']}, empty alarmed {summary['alarmed_mean']:.1f}% (mean)")
    return settings, trials


# Trade-Off Table: for each detection level, the setting with the least empty alarmed time
def tradeOffLines(name, trials, chosen):
    table = pd.DataFrame([{**settings, **summary} for settings, summary in trials])
    table = table.sort_values(["pd", "alarmed_mean"], ascending=[False, True]).drop_duplicates("detected")
    lines = [f"{name}: detection vs false alarms (best setting at each detection level, held-out)",
             f"{'walks':>9s} {'detect %':>8s} {'empty alarmed % (mean)':>23s} {'INTRUDER %':>10s} {'delay s':>7s}  setting"]
    for _, row in table.head(12).iterrows():
        mark = "  <- chosen" if all(row[k] == chosen[k] for k in chosen) else ""
        lines.append(f"{int(row['detected']):>4d}/{int(row['walks']):<4d} {row['pd']:8.1f} {row['alarmed_mean']:23.1f} "
                     f"{row['intruder_mean']:10.1f} {row['delay']:7.1f}  INTRUDER {row['intruder_enter']}/{row['intruder_exit']} "
                     f"MAYBE {row['maybe_enter']}/{row['maybe_exit']} hold {row['hold_s']:g}{mark}")
    return lines


def pirStates(data):
    return np.where(data["pirThresholdIntervalActive"].to_numpy() == 1, INTRUDER, NO_INTRUDER)


# Session Groups For The Report: Site_Day_Type_Weather -> site, type and weather
def groupNames(recording):
    parts = recording.split("#")[0].split("_")
    if len(parts) < 4:      # older names do not follow the convention
        return []
    return [f"site: {parts[0]}", f"type: {parts[2]}", f"weather: {parts[3]}"]


def reportLines(title, systems):
    """systems: {name: per-recording results}. Returns report lines."""
    lines = [title, "-" * len(title)]
    header = f"{'Recording / group':42s}" + "".join(f" | {name:>36s}" for name in systems)
    lines += [header, "-" * len(header)]

    def cell(summary):
        parts = []
        if summary["walks"]:
            parts.append(f"walks {summary['detected']}/{summary['walks']}")
            if not np.isnan(summary["delay"]):
                parts.append(f"{summary['delay']:.1f}s")
        if summary["empty_min"]:
            parts.append(f"empty alarmed {summary['alarmed']:.1f}%")
        return " ".join(parts) if parts else "-"

    first = next(iter(systems.values()))
    for i, recording in enumerate(first["recording"]):
        lines.append(f"{recording[:42]:42s}" + "".join(
            f" | {cell(summarise(results.iloc[[i]])):>36s}" for results in systems.values()))
    lines.append("-" * len(header))
    groups = sorted({g for recording in first["recording"] for g in groupNames(recording)})
    for group in groups:
        mask = [group in groupNames(r) for r in first["recording"]]
        lines.append(f"{group[:42]:42s}" + "".join(
            f" | {cell(summarise(results[mask])):>36s}" for results in systems.values()))
    lines.append(f"{'ALL':42s}" + "".join(f" | {cell(summarise(results)):>36s}" for results in systems.values()))
    return lines


# Full Demo Run: settings, final first-principles models, report
def runDemoTraining(runName, folder, window, modelPath, backupPath, progress=print):
    folder.mkdir(parents=True, exist_ok=False)
    progress(f"Building features ({window:g} s window) for every demo recording...")
    data = buildDemoData(window, progress)
    progress(f"  {len(data)} rows, {data['session'].nunique()} sessions, "
             f"{int(data.drop_duplicates('recording')['empty'].sum())} empty recordings")
    trainRows = ~data["nearBoundary"].to_numpy()

    outputs = {}
    tradeOffs = {}
    for name, features, path in (("Fused system", getFusedFeatureNames(), modelPath),
                                 ("PIR-only backup", getHistoryFeatureNames(), backupPath)):
        progress(f"\n{name}: leave-one-session-out scores...")
        scores = heldOutScores(data, features, progress)
        progress(f"{name}: choosing alarm settings...")
        settings, trials = chooseSettings(data, scores, progress)
        progress(f"{name}: training the first-principles Gradient Boosting on all rows...")
        model = GradientBoosting().fit(data.loc[trainRows, features].to_numpy(dtype=float),
                                       data.loc[trainRows, "human"].to_numpy(), features, progress=progress)
        model.save(path, window, settings)
        progress(f"Saved {path}")
        outputs[name] = (settings, recordingResults(data, runStates(data, scores, settings)))
        tradeOffs[name] = tradeOffLines(name, trials, settings)

    pirResults = recordingResults(data, pirStates(data))
    systems = {"Thresholded PIR": pirResults}
    systems.update({name: results for name, (_, results) in outputs.items()})

    lines = [f"DEMO TRAINING REPORT: {runName}",
             f"Window {window:g} s. Trained on every recording under: {', '.join(DEMO_DATASETS)}",
             f"Rows within {BOUNDARY_S:g} s of a label start/end dropped from training (key-press lag).",
             "Every number below is LEAVE-ONE-SESSION-OUT: each session is scored by a model that never saw it.",
             f"Walk detected = INTRUDER within {MAX_DELAY_S:g} s of entry. Empty alarmed = MAYBE or INTRUDER time "
             "in recordings named Empty/Distractor/Negative (or with no labelled walks).",
             f"Thermopile threshold: {'fixed Z' if _fixedThreshold() else 'temperature table'}.", ""]
    for name, (settings, _) in outputs.items():
        lines.append(f"{name} alarm settings: INTRUDER on {settings['intruder_enter']} / off {settings['intruder_exit']}, "
                     f"MAYBE on {settings['maybe_enter']} / off {settings['maybe_exit']}, hold {settings['hold_s']:g} s")
    lines.append("")
    lines += reportLines("RESULTS (walks detected out of total, median time to INTRUDER, empty alarmed time)", systems)
    for name in tradeOffs:
        lines += [""] + tradeOffs[name]
    text = "\n".join(lines) + "\n"
    (folder / "demo_report.txt").write_text(text, encoding="utf-8")
    pd.concat([results.assign(system=name) for name, results in systems.items()]).drop(columns="delays") \
        .to_csv(folder / "demo_sessions.csv", index=False)
    return text


def _fixedThreshold():
    from Thermopile import processing
    return bool(getattr(processing, "USE_FIXED_THRESHOLD", False))
