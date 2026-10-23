"""Evaluation of the three-state alarm output on labelled recordings.

Requirements (project proposal) checked here:
  Spec 1    the fused system must achieve at least a 20% improvement in
            detection accuracy over the PIR on its own, and at least a 50%
            lower false alarm rate (both relative, same recordings).
  Spec 4    alarm within 10 s of a person entering the zone.
  Specs 2 and 3 (field of view / range and speed) are verified per recording
  on the testing set, recorded at known distances and speeds.

Project definitions:
  Event Pd  = a labelled entry counts as detected only if the system reaches
              INTRUDER at least once within MAX_DELAY_S of the entry, while the
              person is still labelled inbound. Entries already in progress at
              the first fusion interval are excluded (true start unknown).
  Pfa       = fraction of non-human fusion intervals that are false alarms:
              MAYBE or INTRUDER while nobody is inbound or outbound, or
              INTRUDER while a person is only in the outbound zone.
  Accuracy  = alert (MAYBE or INTRUDER) vs inbound label, per fusion interval.
  Ward      = event quality after Ward, Lukowicz & Gellersen (2011), on the
              INTRUDER output: C (one-to-one), D (deleted / missed),
              F (fragmented), M (merged), FM (both), I (inserted false alarm).
              Fresh Pd: the alarm was OFF just before the entry and switches ON
              within MAX_DELAY_S while the person is present.
  Improvements over thresholded PIR are relative:
              Pd improvement = (fused - PIR) / PIR x 100
              Pfa reduction  = (PIR - fused) / PIR x 100
              Accuracy gain  = (fused - PIR) / PIR x 100

Every recording is evaluated on its own; nothing crosses recording boundaries.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from Fusion.realtimeModel import StateMachine, NO_INTRUDER, MAYBE_INTRUDER, INTRUDER

LABEL_COLUMN = "human"
MAX_DELAY_S = 10.0

# Requirements (project proposal)
TARGET_ACC_GAIN = 20.0      # Spec 1: accuracy improvement over PIR (%)
TARGET_PFA_DROP = 50.0      # Spec 1: false alarm reduction over PIR (%)

# Alarm-threshold search grid (tuned on the validation recordings)
GRID_INTRUDER_ENTER = [round(0.50 + 0.025 * k, 3) for k in range(17)]   # 0.50 .. 0.90
GRID_INTRUDER_GAP = [0.05, 0.10, 0.15, 0.20]       # intruder_enter - intruder_exit
GRID_MAYBE_BELOW = [0.01, 0.05, 0.10]              # intruder_exit - maybe_enter
GRID_MAYBE_GAP = [0.05, 0.10, 0.15]                # maybe_enter - maybe_exit
GRID_HOLD_S = [0.0, 1.0, 2.0]

WARD_KEYS = ("C", "D", "F", "M", "FM", "I")


# Reads Outbound Labels At The Evaluation Timestamps (never used for training)
def addOutboundLabels(evalData):
    evaluation = evalData.copy()
    outbound = np.zeros(len(evaluation), dtype=np.uint8)
    for recording, positions in evaluation.groupby("recording", sort=False).indices.items():
        labelPath = Path(recording) / "labels(outbound).csv"
        if not labelPath.is_file():
            raise FileNotFoundError(f"Missing outbound evaluation labels: {labelPath}")
        intervals = pd.read_csv(labelPath)
        if not {"start_s", "end_s", "human"}.issubset(intervals.columns):
            raise ValueError(f"Invalid outbound label columns in {labelPath}")
        times = evaluation.iloc[positions]["fusionTime"].to_numpy(dtype=float)
        for start, end, human in intervals[["start_s", "end_s", "human"]].itertuples(index=False, name=None):
            if int(human) == 1:
                outbound[positions] |= ((times >= float(start)) & (times < float(end)))
    evaluation["outboundHuman"] = outbound
    return evaluation


# Row Indices Of Each Recording, In Time Order
def recordingIndices(evalData):
    times = evalData["fusionTime"].to_numpy(dtype=float)
    groups = []
    for recording, idx in evalData.groupby("recording", sort=False).indices.items():
        idx = np.asarray(idx)
        idx = idx[np.argsort(times[idx], kind="stable")]
        groups.append((recording, idx))
    return groups


# Runs The Real-Time State Machine Over Every Recording (fresh machine per recording)
def runStateMachine(evalData, scores, thresholds):
    scores = np.asarray(scores, dtype=float)
    times = evalData["fusionTime"].to_numpy(dtype=float)
    states = np.zeros(len(evalData), dtype=int)
    for _, idx in recordingIndices(evalData):
        machine = StateMachine(thresholds)
        for i in idx:
            machine.update(scores[i], times[i])
            states[i] = machine.state
    return states


# Thresholded PIR Baseline: INTRUDER if ANY PIR sample since the previous
# thermal frame crossed the threshold (built in Fusion/dataset.py)
def pirBaselineStates(evalData):
    active = pd.to_numeric(evalData["pirThresholdIntervalActive"], errors="raise").to_numpy()
    if not np.isin(active, [0, 1]).all():
        raise ValueError("Nonbinary PIR interval flags in evaluation data")
    return np.where(active == 1, INTRUDER, NO_INTRUDER)


# Splits A 0/1 Sequence Into Runs Of 1s: [(start, end), ...]
def runs(values):
    values = np.asarray(values, dtype=bool)
    if not len(values):
        return []
    start = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    end = np.flatnonzero(values & ~np.r_[values[1:], False])
    return list(zip(start.tolist(), end.tolist()))


# Ward-Style Event Quality For One Recording
def wardCounts(times, truth, predicted):
    counts = {key: 0 for key in WARD_KEYS + ("Eligible", "Fresh", "Preexisting")}
    delays = []
    truthRuns = runs(truth)
    predictedRuns = runs(predicted)
    truthToPredicted = [set() for _ in truthRuns]
    predictedToTruth = [set() for _ in predictedRuns]
    for gi, (a, b) in enumerate(truthRuns):
        for pi, (c, d) in enumerate(predictedRuns):
            if max(a, c) <= min(b, d):
                truthToPredicted[gi].add(pi)
                predictedToTruth[pi].add(gi)
    counts["I"] = sum(not matches for matches in predictedToTruth)
    for gi, (a, b) in enumerate(truthRuns):
        links = truthToPredicted[gi]
        if not links:
            counts["D"] += 1
        else:
            fragmented = len(links) > 1
            merged = any(len(predictedToTruth[p]) > 1 for p in links)
            counts["FM" if fragmented and merged else "F" if fragmented else "M" if merged else "C"] += 1
        # Fresh alert: OFF just before entry, OFF -> ON within the delay limit
        if a == 0:
            continue
        counts["Eligible"] += 1
        if predicted[a - 1]:
            counts["Preexisting"] += 1
            continue
        for i in range(a, b + 1):
            if times[i] > times[a] + MAX_DELAY_S:
                break
            if predicted[i] and not predicted[i - 1]:
                counts["Fresh"] += 1
                delays.append(float(times[i] - times[a]))
                break
    return counts, delays


# Per-Recording Arrays, Prepared Once (threshold tuning re-uses them many times)
class Recordings:
    def __init__(self, evalData):
        if "outboundHuman" not in evalData.columns:
            raise ValueError("Outbound evaluation labels were not loaded (use addOutboundLabels)")
        times = evalData["fusionTime"].to_numpy(dtype=float)
        human = evalData[LABEL_COLUMN].to_numpy(dtype=int)
        outbound = evalData["outboundHuman"].to_numpy(dtype=int)
        self.items = []
        for recording, idx in recordingIndices(evalData):
            t = times[idx]
            if len(t) > 1 and np.any(np.diff(t) <= 0):
                raise ValueError(f"Non-increasing times in recording {recording}")
            h = human[idx]
            starts = np.flatnonzero((h == 1) & (np.r_[0, h[:-1]] == 0))
            ends = np.flatnonzero((h == 1) & (np.r_[h[1:], 0] == 0))
            entries = [(s, e) for s, e in zip(starts, ends) if s != 0]   # left-censored excluded
            self.items.append(dict(recording=recording, idx=idx, times=t, human=h,
                                   outbound=outbound[idx], entries=entries))

    # Core Numbers Used For Tuning And Reporting (states aligned to the evaluation rows)
    def core(self, states):
        states = np.asarray(states, dtype=int)
        events = detected = negatives = false = episodes = correct = total = 0
        humanFrames = humanAlerts = 0
        delays = []
        for r in self.items:
            st = states[r["idx"]]
            for s, e in r["entries"]:
                events += 1
                deadline = min(r["times"][s] + MAX_DELAY_S, r["times"][e])
                for k in range(s, e + 1):
                    if r["times"][k] > deadline:
                        break
                    if st[k] == INTRUDER:
                        detected += 1
                        delays.append(float(r["times"][k] - r["times"][s]))
                        break
            negative = r["human"] == 0
            falseAlarm = negative & ((st == INTRUDER) | ((st == MAYBE_INTRUDER) & (r["outbound"] == 0)))
            alert = st >= MAYBE_INTRUDER
            correct += int(np.count_nonzero(alert == (r["human"] == 1)))
            total += len(st)
            negatives += int(negative.sum())
            false += int(falseAlarm.sum())
            episodes += int(np.count_nonzero(falseAlarm & ~np.r_[False, falseAlarm[:-1]]))
            humanFrames += int(np.count_nonzero(~negative))
            humanAlerts += int(np.count_nonzero(alert & ~negative))
        return {
            "Event Pd (%)": 100.0 * detected / events if events else np.nan,
            "Pfa (%)": 100.0 * false / negatives if negatives else np.nan,
            "Accuracy (%)": 100.0 * correct / total if total else np.nan,
            "Frame Pd (%)": 100.0 * humanAlerts / humanFrames if humanFrames else np.nan,
            "Mean Delay (s)": float(np.mean(delays)) if delays else np.nan,
            "Max Delay (s)": float(np.max(delays)) if delays else np.nan,
            "Events": events,
            "Detected Events": detected,
            "False Alarm Episodes": episodes,
        }

    # Ward Metrics On The INTRUDER Output
    def ward(self, states):
        states = np.asarray(states, dtype=int)
        totals = {key: 0 for key in WARD_KEYS + ("Eligible", "Fresh", "Preexisting")}
        delays = []
        for r in self.items:
            counts, d = wardCounts(r["times"], r["human"] == 1, states[r["idx"]] == INTRUDER)
            for key in totals:
                totals[key] += counts[key]
            delays += d
        result = {f"Ward {key}": totals[key] for key in WARD_KEYS}
        result["Ward Fresh Pd (%)"] = 100.0 * totals["Fresh"] / totals["Eligible"] if totals["Eligible"] else np.nan
        result["Ward Fresh Delay (s)"] = float(np.mean(delays)) if delays else np.nan
        result["Ward Preexisting"] = totals["Preexisting"]
        return result


# Every Reported Metric For One Set Of States
def evaluateStates(evalData, states, recordings=None):
    recordings = recordings or Recordings(evalData)
    states = np.asarray(states, dtype=int)
    metrics = recordings.core(states)
    metrics.update(recordings.ward(states))
    metrics["Maybe Intervals"] = int(np.count_nonzero(states == MAYBE_INTRUDER))
    metrics["Intruder Intervals"] = int(np.count_nonzero(states == INTRUDER))
    return metrics


def improvement(new, old, reduction=False):
    if old is None or not np.isfinite(old) or old == 0 or not np.isfinite(new):
        return np.nan
    return 100.0 * ((old - new) if reduction else (new - old)) / old


# Spec 4: every labelled entry alarmed (INTRUDER) within MAX_DELAY_S
def meetsResponseTime(metrics):
    return bool(metrics["Events"] > 0 and metrics["Detected Events"] == metrics["Events"])


# Improvements Over The Thresholded PIR And The Pass/Fail Of Each Requirement
REQUIREMENT_CHECKS = ("Meets Accuracy Gain", "Meets Pfa Reduction", "Meets Response Time")

def compareToPir(fused, pir):
    result = {
        "Pd Improvement (%)": improvement(fused["Event Pd (%)"], pir["Event Pd (%)"]),
        "Pfa Reduction (%)": improvement(fused["Pfa (%)"], pir["Pfa (%)"], reduction=True),
        "Accuracy Gain (%)": improvement(fused["Accuracy (%)"], pir["Accuracy (%)"]),
    }
    result["Meets Accuracy Gain"] = bool(result["Accuracy Gain (%)"] >= TARGET_ACC_GAIN)     # Spec 1
    result["Meets Pfa Reduction"] = bool(result["Pfa Reduction (%)"] >= TARGET_PFA_DROP)     # Spec 1
    result["Meets Response Time"] = meetsResponseTime(fused)                                # Spec 4
    result["Requirements Met"] = sum(result[key] for key in REQUIREMENT_CHECKS)
    return result


def candidateThresholds():
    for ie in GRID_INTRUDER_ENTER:
        for gap in GRID_INTRUDER_GAP:
            ix = round(ie - gap, 3)
            for below in GRID_MAYBE_BELOW:
                me = round(ix - below, 3)
                for mgap in GRID_MAYBE_GAP:
                    mx = round(me - mgap, 3)
                    if not (0 <= mx < me < ix < ie <= 1):
                        continue
                    for hold in GRID_HOLD_S:
                        yield {"maybe_enter": me, "maybe_exit": mx,
                               "intruder_enter": ie, "intruder_exit": ix, "hold_s": hold}


# Tune The Alarm Thresholds On The Validation Recordings
# Choice rule (proposal requirements, relative to the thresholded PIR on the same data):
#   1. most Spec 1 parts met (accuracy gain >= 20%, false alarm reduction >= 50%)
#   2. then the most entries alarmed within MAX_DELAY_S (Spec 4)
#   3. then the highest accuracy, then the lowest false alarm rate
def tuneThresholds(evalData, scores):
    recordings = Recordings(evalData)
    pir = recordings.core(pirBaselineStates(evalData))
    best = None
    for thresholds in candidateThresholds():
        states = runStateMachine(evalData, scores, thresholds)
        m = recordings.core(states)
        spec1 = (int(improvement(m["Accuracy (%)"], pir["Accuracy (%)"]) >= TARGET_ACC_GAIN)
                 + int(improvement(m["Pfa (%)"], pir["Pfa (%)"], reduction=True) >= TARGET_PFA_DROP))
        key = (-spec1, -m["Detected Events"], -m["Accuracy (%)"], m["Pfa (%)"])
        if best is None or key < best[0]:
            best = (key, thresholds)
    key, thresholds = best
    return {"thresholds": thresholds, "meets_spec1": key[0] == -2}
