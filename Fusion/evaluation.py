from pathlib import Path

import numpy as np
import pandas as pd

from Fusion.realtimeModel import StateMachine, NO_INTRUDER, MAYBE_INTRUDER, INTRUDER

# Evaluation label and maximum allowed detection delay
LABEL_COLUMN = "human"
MAX_DELAY_S = 10.0

# Required improvement compared with the PIR baseline
TARGET_ACC_GAIN = 20.0
TARGET_PFA_DROP = 50.0

# Values tested when choosing the three state alarm thresholds
GRID_INTRUDER_ENTER = [round(0.50 + 0.025 * k, 3) for k in range(17)]
GRID_INTRUDER_GAP = [0.05, 0.10, 0.15, 0.20]
GRID_MAYBE_BELOW = [0.01, 0.05, 0.10]
GRID_MAYBE_GAP = [0.05, 0.10, 0.15]
GRID_HOLD_S = [0.0, 1.0, 2.0]

# Ward event types: correct, missed, fragmented, merged, both and inserted
WARD_KEYS = ("C", "D", "F", "M", "FM", "I")

# Load outbound human intervals for evaluation only
def addOutboundLabels(evalData):
    # Keep the original evaluation data unchanged
    evaluation = evalData.copy()
    outbound = np.zeros(len(evaluation), dtype=np.uint8)
    for recording, positions in evaluation.groupby("recording", sort=False).indices.items():
        # Each recording needs its own outbound labels
        labelPath = Path(recording) / "labels(outbound).csv"
        if not labelPath.is_file():
            raise FileNotFoundError(f"Missing outbound evaluation labels: {labelPath}")
        intervals = pd.read_csv(labelPath)
        if not {"start_s", "end_s", "human"}.issubset(intervals.columns):
            raise ValueError(f"Invalid outbound label columns in {labelPath}")
        times = evaluation.iloc[positions]["fusionTime"].to_numpy(dtype=float)
        # Mark outbound human frames using their timestamps
        for start, end, human in intervals[["start_s", "end_s", "human"]].itertuples(index=False, name=None):
            if int(human) == 1:
                outbound[positions] |= ((times >= float(start)) & (times < float(end)))
    evaluation["outboundHuman"] = outbound
    return evaluation

# Group each recording and sort its frames by time
def recordingIndices(evalData):
    times = evalData["fusionTime"].to_numpy(dtype=float)
    groups = []
    for recording, idx in evalData.groupby("recording", sort=False).indices.items():
        idx = np.asarray(idx)
        idx = idx[np.argsort(times[idx], kind="stable")]
        groups.append((recording, idx))
    return groups

# Run the live alarm logic separately for each recording
def runStateMachine(evalData, scores, thresholds):
    scores = np.asarray(scores, dtype=float)
    times = evalData["fusionTime"].to_numpy(dtype=float)
    states = np.zeros(len(evalData), dtype=int)
    for _, idx in recordingIndices(evalData):
        # Reset the alarm state at the start of each recording
        machine = StateMachine(thresholds)
        for i in idx:
            machine.update(scores[i], times[i])
            states[i] = machine.state
    return states

# PIR baseline: alarm if any sample crossed the PIR threshold
def pirBaselineStates(evalData):
    active = pd.to_numeric(evalData["pirThresholdIntervalActive"], errors="raise").to_numpy()
    # Reject invalid baseline values
    if not np.isin(active, [0, 1]).all():
        raise ValueError("Nonbinary PIR interval flags in evaluation data")
    return np.where(active == 1, INTRUDER, NO_INTRUDER)

# Find the start and end of each continuous active interval
def runs(values):
    values = np.asarray(values, dtype=bool)
    if not len(values):
        return []
    start = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    end = np.flatnonzero(values & ~np.r_[values[1:], False])
    return list(zip(start.tolist(), end.tolist()))

# Compare actual human events against INTRUDER alarm events
def wardCounts(times, truth, predicted):
    counts = {key: 0 for key in WARD_KEYS + ("Eligible", "Fresh", "Preexisting")}
    delays = []
    # Extract the human events and predicted alarm intervals
    truthRuns = runs(truth)
    predictedRuns = runs(predicted)
    truthToPredicted = [set() for _ in truthRuns]
    predictedToTruth = [set() for _ in predictedRuns]
    # Match events and alarms when their intervals overlap
    for gi, (a, b) in enumerate(truthRuns):
        for pi, (c, d) in enumerate(predictedRuns):
            if max(a, c) <= min(b, d):
                truthToPredicted[gi].add(pi)
                predictedToTruth[pi].add(gi)
    # Unmatched alarm intervals count as insertions
    counts["I"] = sum(not matches for matches in predictedToTruth)
    # Identify missed, fragmented, merged and correct events
    for gi, (a, b) in enumerate(truthRuns):
        # Classify each human event by its alarm matches
        links = truthToPredicted[gi]
        if not links:
            counts["D"] += 1
        else:
            fragmented = len(links) > 1
            merged = any(len(predictedToTruth[p]) > 1 for p in links)
            counts["FM" if fragmented and merged else "F" if fragmented else "M" if merged else "C"] += 1
        # Skip fresh alert checks for events already active at recording start
        if a == 0:
            continue
        # Only a new alarm within ten seconds counts as a fresh detection
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
    # Measure time outside and inside the matched alarm intervals
    durations = intervalDurations(times)
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    counts["Overfill"] = 0.0
    counts["Underfill"] = 0.0
    # Overfill: extra alarm time outside the human event
    for pi, (c, d) in enumerate(predictedRuns):
        if predictedToTruth[pi]:
            counts["Overfill"] += float(durations[c:d + 1][~truth[c:d + 1]].sum())
    # Underfill: missing alarm time during a detected human event
    for gi, (a, b) in enumerate(truthRuns):
        if truthToPredicted[gi]:
            counts["Underfill"] += float(durations[a:b + 1][~predicted[a:b + 1]].sum())
    return counts, delays

# Estimate the duration represented by each fusion frame
def intervalDurations(times):
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return np.zeros(len(times))
    gaps = np.diff(times)
    # Use typical frame spacing for large gaps and the final frame
    typical = float(np.median(gaps))
    gaps = np.where(gaps > 1.0, typical, gaps)
    return np.r_[gaps, typical]

# Prepare recording data once for repeated evaluation and threshold tuning
class Recordings:
    # Store ordered timestamps, labels and valid human entries
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
            # Locate each entry and exit in the human labels
            starts = np.flatnonzero((h == 1) & (np.r_[0, h[:-1]] == 0))
            ends = np.flatnonzero((h == 1) & (np.r_[h[1:], 0] == 0))
            # Ignore entries already active at the first frame
            entries = [(s, e) for s, e in zip(starts, ends) if s != 0]
            self.items.append(dict(recording=recording, idx=idx, times=t, human=h,
                                   outbound=outbound[idx], entries=entries))

    # Calculate detection, false alarm, accuracy and response time metrics
    def core(self, states):
        states = np.asarray(states, dtype=int)
        events = detected = negatives = false = episodes = correct = total = 0
        humanFrames = humanAlerts = 0
        delays = []
        for r in self.items:
            st = states[r["idx"]]
            # Check for an INTRUDER alarm within ten seconds of each entry
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
            # Maybe and Intruder count as positive for classification accuracy
            negative = r["human"] == 0
            # Outbound Maybe is excluded from false alarms, but outbound Intruder is not
            falseAlarm = negative & ((st == INTRUDER) | ((st == MAYBE_INTRUDER) & (r["outbound"] == 0)))
            alert = st >= MAYBE_INTRUDER
            correct += int(np.count_nonzero(alert == (r["human"] == 1)))
            total += len(st)
            negatives += int(negative.sum())
            false += int(falseAlarm.sum())
            # Count the beginning of each separate false alarm episode
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

    # Calculate Ward event quality using only the INTRUDER state
    def ward(self, states):
        states = np.asarray(states, dtype=int)
        totals = {key: 0 for key in WARD_KEYS + ("Eligible", "Fresh", "Preexisting", "Overfill", "Underfill")}
        delays = []
        for r in self.items:
            counts, d = wardCounts(r["times"], r["human"] == 1, states[r["idx"]] == INTRUDER)
            for key in totals:
                totals[key] += counts[key]
            delays += d
        result = {f"Ward {key}": totals[key] for key in WARD_KEYS}
        # Calculate average extra and missing alarm time per detected event
        matched = totals["C"] + totals["F"] + totals["M"] + totals["FM"]
        result["Overfill (s)"] = float(totals["Overfill"])
        result["Underfill (s)"] = float(totals["Underfill"])
        result["Mean Overfill (s)"] = float(totals["Overfill"]) / matched if matched else np.nan
        result["Mean Underfill (s)"] = float(totals["Underfill"]) / matched if matched else np.nan
        result["Ward Fresh Pd (%)"] = 100.0 * totals["Fresh"] / totals["Eligible"] if totals["Eligible"] else np.nan
        result["Ward Fresh Delay (s)"] = float(np.mean(delays)) if delays else np.nan
        result["Ward Preexisting"] = totals["Preexisting"]
        return result

# Combine standard metrics, Ward metrics and state counts
def evaluateStates(evalData, states, recordings=None):
    recordings = recordings or Recordings(evalData)
    states = np.asarray(states, dtype=int)
    metrics = recordings.core(states)
    metrics.update(recordings.ward(states))
    metrics["Maybe Intervals"] = int(np.count_nonzero(states == MAYBE_INTRUDER))
    metrics["Intruder Intervals"] = int(np.count_nonzero(states == INTRUDER))
    return metrics

# Calculate percentage gain or percentage reduction against a baseline
def improvement(new, old, reduction=False):
    if old is None or not np.isfinite(old) or old == 0 or not np.isfinite(new):
        return np.nan
    return 100.0 * ((old - new) if reduction else (new - old)) / old

# Check the maximum delay of detected entries against the ten second limit
def meetsResponseTime(metrics):
    maxDelay = metrics.get("Max Delay (s)", np.nan)
    return bool(metrics["Detected Events"] > 0 and np.isfinite(maxDelay) and maxDelay <= MAX_DELAY_S)

# Requirements included in the overall pass count
REQUIREMENT_CHECKS = ("Meets Accuracy Gain", "Meets Pfa Reduction", "Meets Response Time")

# Compare fused results to PIR and check project requirements
def compareToPir(fused, pir):
    result = {
        "Pd Improvement (%)": improvement(fused["Event Pd (%)"], pir["Event Pd (%)"]),
        "Pfa Reduction (%)": improvement(fused["Pfa (%)"], pir["Pfa (%)"], reduction=True),
        "Accuracy Gain (%)": improvement(fused["Accuracy (%)"], pir["Accuracy (%)"]),
    }
    # Check the accuracy gain, false alarm reduction and response time
    result["Meets Accuracy Gain"] = bool(result["Accuracy Gain (%)"] >= TARGET_ACC_GAIN)
    result["Meets Pfa Reduction"] = bool(result["Pfa Reduction (%)"] >= TARGET_PFA_DROP)
    result["Meets Response Time"] = meetsResponseTime(fused)
    result["Requirements Met"] = sum(result[key] for key in REQUIREMENT_CHECKS)
    return result

# Generate valid alarm thresholds and hold times from the search grid
def candidateThresholds():
    for ie in GRID_INTRUDER_ENTER:
        for gap in GRID_INTRUDER_GAP:
            ix = round(ie - gap, 3)
            for below in GRID_MAYBE_BELOW:
                me = round(ix - below, 3)
                for mgap in GRID_MAYBE_GAP:
                    mx = round(me - mgap, 3)
                    # Keep exit thresholds below enter thresholds in the correct order
                    if not (0 <= mx < me < ix < ie <= 1):
                        continue
                    for hold in GRID_HOLD_S:
                        yield {"maybe_enter": me, "maybe_exit": mx,
                               "intruder_enter": ie, "intruder_exit": ix, "hold_s": hold}

# Find the best state machine settings using validation recordings
def tuneThresholds(evalData, scores):
    recordings = Recordings(evalData)
    # Evaluate the PIR baseline on the same recordings
    pir = recordings.core(pirBaselineStates(evalData))
    best = None
    for thresholds in candidateThresholds():
        states = runStateMachine(evalData, scores, thresholds)
        m = recordings.core(states)
        # First priority: satisfy both accuracy and false alarm requirements
        spec1 = (int(improvement(m["Accuracy (%)"], pir["Accuracy (%)"]) >= TARGET_ACC_GAIN)
                 + int(improvement(m["Pfa (%)"], pir["Pfa (%)"], reduction=True) >= TARGET_PFA_DROP))
        # Then prioritise detected events, accuracy and fewer false alarms
        key = (-spec1, -m["Detected Events"], -m["Accuracy (%)"], m["Pfa (%)"])
        if best is None or key < best[0]:
            best = (key, thresholds)
    key, thresholds = best
    # Return the chosen settings and whether both requirements passed
    return {"thresholds": thresholds, "meets_spec1": key[0] == -2}
