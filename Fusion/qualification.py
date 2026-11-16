from pathlib import Path
from datetime import datetime
import json
import math
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Fusion.evaluation import (MAX_DELAY_S, TARGET_ACC_GAIN, TARGET_PFA_DROP, Recordings,
                               compareToPir, evaluateStates)
from Fusion.realtimeModel import INTRUDER
from Fusion import report

SEEN_SITE = "Driveway"                 
HALF_FOV_DEG = 55.0                  

PATH_LENGTH_M = {1: 1.84, 4: 5.96, 7: 10.88, 10: 12.88}

SPEED_CLASSES = ("Very slow", "Slow", "Medium", "Fast")

SPEED_CLASS_MS = {"Very slow": 0.2, "Slow": 0.5, "Medium": 1.4, "Fast": 5.0}

PROTOCOL_ORDER = ("Fast", "Medium", "Slow", "Very slow")
WALKS_PER_CLASS = 3
REQUIRED_RANGE_M = (1.0, 10.0)       
REQUIRED_SPEED_MS = (0.2, 5.0)       

PIR = "Thresholded PIR"
FUSED = "Fused system"
MULTI_TYPES = {"sidebyside": "Side by side", "apart": r"Separated by $\sim$3~s"}

# Identify the site, test type and conditions from folder names
def parseRecordingName(name):
    info = {"name": name, "site": None, "kind": None, "distance": None, "edge": None,
            "people": None, "condition": None}

    # Separate the folder name into its test fields
    parts = name.split("_")
    if len(parts) < 3:
        return info
    site, test, condition = parts[0], parts[1], parts[2].lower()
    if condition not in ("day", "night"):
        return info
    info["site"] = site
    info["condition"] = condition.capitalize()
    distance = re.fullmatch(r"(\d+(?:\.\d+)?)m", test, flags=re.IGNORECASE)
    edge = re.fullmatch(r"edge([+-])55", test, flags=re.IGNORECASE)
    multi = re.fullmatch(r"2people-(sidebyside|apart)", test, flags=re.IGNORECASE)

    # Identify distance walks from the recording name
    if distance:
        info["kind"] = "distance"
        info["distance"] = float(distance.group(1))
    elif edge:
        info["kind"] = "edge"
        info["edge"] = f"{edge.group(1)}55"
    elif test.lower() == "empty":
        info["kind"] = "empty"
    elif multi:
        info["kind"] = "multi"
        info["people"] = multi.group(1).lower()
    return info

# Use a measured walking path or calculate it from the field of view
def pathLength(distance):
    measured = PATH_LENGTH_M.get(int(round(distance))) if float(distance).is_integer() else None

    # Prefer the actual measured walking path
    if measured:
        return float(measured), True
    return 2.0 * distance * math.tan(math.radians(HALF_FOV_DEG)), False

# Describe the recording for the results table
def contentText(info, entries):
    kind = info["kind"]
    if kind == "distance":
        return f"{entries} walks at {info['distance']:g}~m"
    if kind == "edge":
        return f"{entries} walks along the ${info['edge'][0]}55^\\circ$ edge"
    if kind == "empty":
        return "Empty scene"
    if kind == "multi":
        return f"{entries} two-person entries ({MULTI_TYPES[info['people']].lower()})"
    return f"{entries} entries"

# Find the first intrusion alarm within the allowed response time
def firstIntruder(times, states, s, e):
    deadline = min(times[s] + MAX_DELAY_S, times[e])
    for k in range(s, e + 1):
        if times[k] > deadline:
            break
        if states[k] == INTRUDER:
            return float(times[k] - times[s])
    return None

# Find the first alarm while the person is inside the zone
def firstIntruderInZone(times, states, s, e):
    for k in range(s, e + 1):
        if states[k] == INTRUDER:
            return float(times[k] - times[s])
    return None

# Collect detection results and response times for every walk
def walkTable(data, warnings):
    rows = []
    fused = data["fusedState"].to_numpy()
    pir = data["pirState"].to_numpy()

    # Process recordings separately in time order
    for item in Recordings(data).items:
        name = Path(item["recording"]).name
        info = parseRecordingName(name)
        labels = pd.read_csv(Path(item["recording"]) / "labels.csv")
        labels = labels[labels["human"] == 1].sort_values("start_s")
        intervals = labels[["start_s", "end_s"]].to_numpy(dtype=float)
        if len(intervals) != len(item["entries"]):
            warnings.append(f"{name}: {len(intervals)} labelled entries but {len(item['entries'])} evaluated "
                            "(an entry already in progress at the start, joined labels, or excluded sections).")
        times = item["times"]
        fusedStates = fused[item["idx"]]
        pirStates = pir[item["idx"]]

        # Evaluate each labelled entry individually
        for number, (s, e) in enumerate(item["entries"], start=1):
            match = [(a, b) for a, b in intervals if a <= times[s] < b]
            duration = float(match[0][1] - match[0][0]) if match else float(times[e] - times[s])
            speed = None       
            fusedDelay = firstIntruder(times, fusedStates, s, e)
            pirDelay = firstIntruder(times, pirStates, s, e)
            fusedResponse = firstIntruderInZone(times, fusedStates, s, e)
            pirResponse = firstIntruderInZone(times, pirStates, s, e)
            rows.append({"recording": name, "site": info["site"], "kind": info["kind"],
                         "condition": info["condition"], "distance": info["distance"], "edge": info["edge"],
                         "people": info["people"], "walk": number, "start_s": float(times[s]),
                         "duration_s": duration, "speed_ms": speed,
                         "fusedDetected": fusedDelay is not None, "fusedDelay": fusedDelay,
                         "pirDetected": pirDelay is not None, "pirDelay": pirDelay,
                         "fusedSeen": fusedResponse is not None, "fusedResponse": fusedResponse,
                         "pirSeen": pirResponse is not None, "pirResponse": pirResponse})
    walks = pd.DataFrame(rows, columns=["recording", "site", "kind", "condition", "distance", "edge", "people",
                                        "walk", "start_s", "duration_s", "speed_ms", "fusedDetected",
                                        "fusedDelay", "pirDetected", "pirDelay", "fusedSeen", "fusedResponse",
                                        "pirSeen", "pirResponse"])
    for column in ("duration_s", "speed_ms", "fusedDelay", "pirDelay", "fusedResponse", "pirResponse", "distance"):
        walks[column] = pd.to_numeric(walks[column], errors="coerce")
    walks["fusedDetected"] = walks["fusedDetected"].astype(bool)
    walks["pirDetected"] = walks["pirDetected"].astype(bool)
    walks["fusedSeen"] = walks["fusedSeen"].astype(bool)
    walks["pirSeen"] = walks["pirSeen"].astype(bool)
    walks["speedClass"] = None
    recordingPaths = {Path(r).name: Path(r) for r in data["recording"].unique()}
    for name, group in walks[walks["kind"] == "distance"].groupby("recording"):
        
        # Read optional speed class labels for each walk
        classFile = recordingPaths[name] / "speed_classes.csv"
        if classFile.is_file():
            classes = pd.read_csv(classFile)["speed_class"].astype(str).str.strip().tolist()
            unknown = sorted(set(classes) - set(SPEED_CLASSES))
            if unknown:
                raise ValueError(f"{classFile}: unknown speed classes {unknown}; use {SPEED_CLASSES}")
            if len(classes) != len(group):
                raise ValueError(f"{classFile}: {len(classes)} classes for {len(group)} evaluated walks")
            walks.loc[group.sort_values("walk").index, "speedClass"] = classes
            counts = pd.Series(classes).value_counts()
            for speedClass in SPEED_CLASSES:
                if counts.get(speedClass, 0) != WALKS_PER_CLASS:
                    warnings.append(f"{name}: {counts.get(speedClass, 0)} {speedClass.lower()} walks "
                                    f"instead of {WALKS_PER_CLASS}.")
            continue
        
        # Fall back to the specified walk order when labels are absent
        expected = WALKS_PER_CLASS * len(SPEED_CLASSES)
        if len(group) != expected:
            raise ValueError(f"{name}: {len(group)} walks instead of {expected}. Add "
                             f"{recordingPaths[name] / 'speed_classes.csv'} (column speed_class, one row per "
                             f"labelled walk in time order, values {SPEED_CLASSES}).")
        ordered = group.sort_values("walk").index.to_numpy()
        for speedClass, members in zip(PROTOCOL_ORDER, np.split(ordered, len(PROTOCOL_ORDER))):
            walks.loc[members, "speedClass"] = speedClass
    
    isDistance = walks["kind"] == "distance"

    # Assign the target speed for each class
    walks.loc[isDistance, "speed_ms"] = walks.loc[isDistance, "speedClass"].map(SPEED_CLASS_MS)
    walks["speed_ms"] = pd.to_numeric(walks["speed_ms"], errors="coerce")
    return walks

# Keep only recordings from the requested group
def subset(data, names):
    return data[data["recording"].map(lambda r: Path(r).name in names)].reset_index(drop=True)

# Find recordings matching the selected site and test conditions
def selectNames(infos, site=None, unseen=False, condition=None, kinds=None):
    names = []
    for name, info in infos.items():
        if info["kind"] is None:
            continue
        if site is not None and info["site"] != site:
            continue
        if unseen and info["site"] == SEEN_SITE:
            continue
        if condition is not None and info["condition"] != condition:
            continue
        if kinds is not None and info["kind"] not in kinds:
            continue
        names.append(name)
    return names

# Evaluate both systems and compare them against Requirement 1
def groupMetrics(data):
    if data.empty:
        return None
    recordings = Recordings(data)
    metrics = {PIR: evaluateStates(data, data["pirState"].to_numpy(), recordings),
               FUSED: evaluateStates(data, data["fusedState"].to_numpy(), recordings)}

    # Calculate improvement relative to the PIR baseline
    comparison = compareToPir(metrics[FUSED], metrics[PIR])
    durations = intervalSeconds(data)
    human = data["human"].to_numpy() == 1
    extra = {"Entries": int(metrics[FUSED]["Events"]), "Negative Intervals": int((~human).sum()),
             "Human Seconds": float(durations[human].sum()), "Negative Seconds": float(durations[~human].sum())}
    return {"metrics": metrics, "comparison": comparison, **extra}

# Calculate the duration covered by each fusion decision
def intervalSeconds(data):
    from Fusion.evaluation import intervalDurations
    seconds = np.zeros(len(data))

    # Process recordings separately in time order
    for item in Recordings(data).items:
        seconds[item["idx"]] = intervalDurations(item["times"])
    return seconds

# Format missing and numeric values for the report
def num(value, decimals=1):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{float(value):.{decimals}f}"

# Convert a requirement result into PASS or FAIL
def passText(ok):
    return "PASS" if ok else "FAIL"

# Escape special characters for LaTeX tables
def texEscape(text):
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

# Build a formatted LaTeX table from supplied data
def texTable(columns, header, rows, caption, label, tabularx=False):
    lines = [r"\begin{table}[H]", r"\centering", r"\fontsize{10}{12}\selectfont"]
    lines.append(rf"\begin{{tabularx}}{{\linewidth}}{{{columns}}}" if tabularx else rf"\begin{{tabular}}{{{columns}}}")
    lines += [r"\hline", " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\", r"\hline"]
    for row in rows:
        if row == "hline":
            lines.append(r"\hline")
        else:
            lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    if rows and rows[-1] != "hline":
        lines.append(r"\hline")
    lines.append(r"\end{tabularx}" if tabularx else r"\end{tabular}")
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"

# Summarise detection, accuracy and false alarms for Requirement 1
def requirementOneTables(group, label, caption, title):
    lines = [f"\n{title}"]
    if group is None:
        lines.append("  No recordings in this group.")
        return lines, None
    m, c = group["metrics"], group["comparison"]
    lines += [f"  Entries: {group['Entries']}, non-human intervals: {group['Negative Intervals']}",
              f"  Human-labelled time: {group['Human Seconds']:.0f} s ({group['Human Seconds'] / 60:.1f} min), "
              f"non-human time: {group['Negative Seconds']:.0f} s ({group['Negative Seconds'] / 60:.1f} min)",
              f"  {'System':16s} | {'Pd %':>6} | {'Pfa %':>6} | {'Acc %':>6} | {'Acc gain %':>10} | {'Pfa red %':>9}"]
    for name in (PIR, FUSED):
        gain = num(c["Accuracy Gain (%)"]) if name == FUSED else "-"
        drop = num(c["Pfa Reduction (%)"]) if name == FUSED else "-"
        lines.append(f"  {name:16s} | {num(m[name]['Event Pd (%)']):>6} | {num(m[name]['Pfa (%)']):>6} | "
                     f"{num(m[name]['Accuracy (%)']):>6} | {gain:>10} | {drop:>9}")
    lines.append(f"  Requirement 1: accuracy gain {passText(c['Meets Accuracy Gain'])}, "
                 f"false alarm reduction {passText(c['Meets Pfa Reduction'])}")
    rows = [
        [PIR, num(m[PIR]["Event Pd (%)"]), num(m[PIR]["Pfa (%)"]), num(m[PIR]["Accuracy (%)"]), "--", "--"],
        [FUSED, num(m[FUSED]["Event Pd (%)"]), num(m[FUSED]["Pfa (%)"]), num(m[FUSED]["Accuracy (%)"]),
         num(c["Accuracy Gain (%)"]), num(c["Pfa Reduction (%)"])],
        "hline",
        [r"\textbf{Target}", "--", "--", "--", rf"$\geq {TARGET_ACC_GAIN:g}$", rf"$\geq {TARGET_PFA_DROP:g}$"],
        "hline",
        [r"\textbf{Pass/Fail}", "--", "--", "--", passText(c["Meets Accuracy Gain"]),
         passText(c["Meets Pfa Reduction"])],
    ]
    tex = texTable("|l|c|c|c|c|c|", ["System", r"Pd (\%)", r"Pfa (\%)", r"Accuracy (\%)",
                                     r"Accuracy gain (\%)", r"Pfa reduction (\%)"], rows,
                   caption.format(n=group["Entries"], negatives=group["Negative Intervals"]), label)
    return lines, tex

# Evaluate range, walking speed and field of view
def qt2Tables(walks, edgeWalks, warnings):
    lines = ["\nQT2: RANGE, FIELD OF VIEW AND SPEED (Requirements 2 and 3)",
             "  Cells: walks detected by fused system / thresholded PIR (out of n)."]
    tex = {}
    if walks.empty:
        lines.append("  No distance recordings.")
    else:
        distances = sorted(walks["distance"].unique())
        rows, textRows = [], []

        # Count successful detections at each distance
        for d in distances:
            atDistance = walks[walks["distance"] == d]
            cells, textCells = [], []
            for speedClass in SPEED_CLASSES + ("Total",):
                cell = atDistance if speedClass == "Total" else atDistance[atDistance["speedClass"] == speedClass]
                cells.append(f"{int(cell['fusedDetected'].sum())}/{int(cell['pirDetected'].sum())}")
                textCells.append(f"{int(cell['fusedDetected'].sum())}/{int(cell['pirDetected'].sum())} "
                                 f"of {len(cell)}")
            rows.append([f"{d:g}"] + cells)
            textRows.append(f"  {d:5g} m | " + " | ".join(f"{c:>12}" for c in textCells))
        totals, speeds, textTotals, textSpeeds = [], [], [], []
        for speedClass in SPEED_CLASSES + ("Total",):
            cell = walks if speedClass == "Total" else walks[walks["speedClass"] == speedClass]
            totals.append(f"{int(cell['fusedDetected'].sum())}/{int(cell['pirDetected'].sum())}")
            textTotals.append(f"{int(cell['fusedDetected'].sum())}/{int(cell['pirDetected'].sum())} of {len(cell)}")
            span = f"{num(cell['speed_ms'].min(), 2)}--{num(cell['speed_ms'].max(), 2)}" if len(cell) else "--"
            speeds.append(span)
            textSpeeds.append(span.replace("--", "-"))
        lines.append("  Distance | " + " | ".join(f"{c:>12}" for c in SPEED_CLASSES + ("Total",)))
        lines += textRows
        lines.append("  Total    | " + " | ".join(f"{c:>12}" for c in textTotals))
        lines.append("  Speed m/s| " + " | ".join(f"{c:>12}" for c in textSpeeds))
        rows += ["hline", [r"\textbf{Total}"] + totals, "hline", [r"\textbf{Target speed (m/s)}"] + speeds]
        counts = walks.groupby(["distance", "speedClass"]).size()
        cellText = (f"Each cell contains {int(counts.iloc[0])} walks." if counts.nunique() == 1
                    else "The number of walks per cell is given in report.txt.")
        tex["qt2_range_speed"] = texTable(
            "|c|c|c|c|c|c|", ["Distance (m)"] + list(SPEED_CLASSES) + ["Total"], rows,
            "Entries detected per distance and speed class, given as fused system / thresholded PIR. " + cellText,
            "tab:qt2_range_speed")
        lines.append("  Speeds are the target speed of each class: " +
                     ", ".join(f"{c} {SPEED_CLASS_MS[c]:g} m/s" for c in SPEED_CLASSES))

    lines.append("  Edge walks:")
    edgeRows = []

    # Check both boundaries of the sensor field of view
    for edge in ("+55", "-55"):
        e = edgeWalks[edgeWalks["edge"] == edge] if not edgeWalks.empty else edgeWalks
        lines.append(f"    {edge} deg: {len(e)} walks, fused {int(e['fusedDetected'].sum()) if len(e) else 0}, "
                     f"PIR {int(e['pirDetected'].sum()) if len(e) else 0}")
        edgeRows.append([rf"${edge}^\circ$", len(e), int(e["fusedDetected"].sum()) if len(e) else "--",
                         int(e["pirDetected"].sum()) if len(e) else "--"])
    tex["qt2_edges"] = texTable("|c|c|c|c|", ["Edge", "Walks", "Fused detected", "PIR detected"], edgeRows,
                                r"Detections for walks along the edges of the $110^\circ$ field of view.",
                                "tab:qt2_edges")

    # Require every distance walk to be detected
    allDetected = bool(len(walks)) and bool(walks["fusedDetected"].all())
    if len(walks):
        dmin, dmax = walks["distance"].min(), walks["distance"].max()
        rangeOk = allDetected and dmin <= REQUIRED_RANGE_M[0] and dmax >= REQUIRED_RANGE_M[1]
        rangeText = f"{int(walks['fusedDetected'].sum())}/{len(walks)} walks detected, {dmin:g}--{dmax:g}~m"
        detectedSpeeds = walks.loc[walks["fusedDetected"], "speed_ms"]
        smin, smax = walks["speed_ms"].min(), walks["speed_ms"].max()
        speedOk = allDetected and smin <= REQUIRED_SPEED_MS[0] and smax >= REQUIRED_SPEED_MS[1]
        speedText = (f"{num(detectedSpeeds.min(), 2)}--{num(detectedSpeeds.max(), 2)}~m/s"
                     if len(detectedSpeeds) else "--")
    else:
        rangeOk = speedOk = False
        rangeText = speedText = "--"
    edgesTested = set(edgeWalks["edge"]) if not edgeWalks.empty else set()
    fovOk = edgesTested == {"+55", "-55"} and bool(edgeWalks["fusedDetected"].all())
    fovText = (f"{int(edgeWalks['fusedDetected'].sum())}/{len(edgeWalks)} edge walks detected"
               if not edgeWalks.empty else "--")
    lines += [f"  Range 1-10 m: {rangeText.replace('~', ' ').replace('--', '-')} -> {passText(rangeOk)}",
              f"  Field of view 110 deg: {fovText} -> {passText(fovOk)}",
              f"  Speed 0.2-5 m/s (detected walks): {speedText.replace('~', ' ').replace('--', '-')} -> "
              f"{passText(speedOk)}",
              "  Pass rules: range = every distance walk detected and 1 m and 10 m tested; field of view = every",
              "  walk along both edges detected; speed = every distance walk detected and walks tested at",
              "  <= 0.2 m/s and >= 5 m/s."]
    tex["qt2_passfail"] = texTable(
        "|l|c|c|", ["Requirement", "Measured", "Pass/Fail"],
        [["Detection from 1~m to 10~m", rangeText, passText(rangeOk)],
         [r"Horizontal field of view $\geq 110^\circ$", fovText, passText(fovOk)],
         ["Detection from 0.2~m/s to 5~m/s", speedText, passText(speedOk)]],
        "Requirement 2 and 3 outcomes.", "tab:qt2_passfail")
    return lines, tex, {"range": rangeOk, "fov": fovOk, "speed": speedOk}

# Summarise response times and identify late alarms
def qt3Tables(qt1Walks, qt2Walks):
    """Response time of EVERY entry: first INTRUDER while the person is in the zone (no 10 s
    limit), so an entry alarmed late is reported here with its real response time."""
    lines = ["\nQT3: RESPONSE TIME (Requirement 4)"]

    # Separate alarmed entries from entries with no alarm
    seen = qt1Walks[qt1Walks["fusedSeen"]]
    response = seen["fusedResponse"].to_numpy(dtype=float)

    # Count responses within the ten second limit
    within = int(np.sum(response <= MAX_DELAY_S))
    late = seen[seen["fusedResponse"] > MAX_DELAY_S]
    never = len(qt1Walks) - len(seen)
    meanDelay = float(np.mean(response)) if len(response) else np.nan
    medianDelay = float(np.median(response)) if len(response) else np.nan
    p95Delay = float(np.percentile(response, 95)) if len(response) else np.nan
    maxDelay = float(np.max(response)) if len(response) else np.nan
    delayOk = bool(len(qt1Walks)) and within == len(qt1Walks)
    lines += [f"  Entries alarmed (INTRUDER) within {MAX_DELAY_S:g} s: {within}/{len(qt1Walks)}",
              f"  Entries alarmed after {MAX_DELAY_S:g} s: {len(late)}"
              + "".join(f"  [{r['recording']} walk {int(r['walk'])}: {r['fusedResponse']:.2f} s]" for _, r in late.iterrows()),
              f"  Entries never alarmed: {never}",
              f"  Response time: mean {num(meanDelay, 2)} s, median {num(medianDelay, 2)} s, "
              f"95th percentile {num(p95Delay, 2)} s, maximum {num(maxDelay, 2)} s",
              "  Processing time per frame is not measured by this script; worst-case response time =",
              f"  maximum response ({num(maxDelay, 2)} s) + maximum processing time."]
    rows = [[f"Entries alarmed within {MAX_DELAY_S:g}~s", f"{within}/{len(qt1Walks)}"],
            [f"Entries alarmed after {MAX_DELAY_S:g}~s", f"{len(late)}"],
            ["Mean response time (s)", num(meanDelay, 2)],
            ["Median response time (s)", num(medianDelay, 2)],
            ["95th percentile response time (s)", num(p95Delay, 2)],
            ["Maximum response time (s)", num(maxDelay, 2)],
            "hline",
            ["Mean processing time per frame (ms)", "XX"],
            ["Maximum processing time per frame (ms)", "XX"],
            "hline",
            [r"\textbf{Specification}", r"$\leq 10$~s"]]
    tex = {"qt3_response": texTable(
        "|l|c|", ["Measurement", "Value"], rows,
        "Response time of the fused system on the testing recordings: time from the labelled entry to "
        "the first Intruder state while the person is in the detection zone.", "tab:qt3_response")}

    speedRows = []
    lines.append(f"  {'Speed class':10s} | {'<=10 s':>7} | {'Mean s':>6} | {'Max s':>6}")
    for speedClass in SPEED_CLASSES:
        cell = qt2Walks[qt2Walks["speedClass"] == speedClass] if len(qt2Walks) else qt2Walks
        hit = cell[cell["fusedSeen"]] if len(cell) else cell
        inTime = int((hit["fusedResponse"] <= MAX_DELAY_S).sum()) if len(hit) else 0
        values = [f"{inTime}/{len(cell)}", num(hit["fusedResponse"].mean() if len(hit) else np.nan, 2),
                  num(hit["fusedResponse"].max() if len(hit) else np.nan, 2)]
        speedRows.append([speedClass] + values)
        lines.append(f"  {speedClass:10s} | {values[0]:>7} | {values[1]:>6} | {values[2]:>6}")
    tex["qt3_speed_delay"] = texTable("|l|c|c|c|", ["Speed class", f"Alarmed within {MAX_DELAY_S:g}~s",
                                                    "Mean response (s)", "Maximum response (s)"], speedRows,
                                      "Response time of the fused system per speed class.",
                                      "tab:qt3_speed_delay")
    return lines, tex, delayOk

# Compare day, night and multiple person tests
def qt4Tables(groups, multiWalks, qt1Group):
    lines = ["\nQT4: FIELD CONDITIONS"]
    rows = []
    lines.append(f"  {'Condition':20s} | {'Entries':>7} | {'Fused Pd':>8} | {'PIR Pd':>6} | {'Fused Pfa':>9} | {'PIR Pfa':>7}")
    for title, key, walksRow in [("Day, walks", "dayWalks", True), ("Day, empty scene", "dayEmpty", False),
                                 ("Night, walks", "nightWalks", True), ("Night, empty scene", "nightEmpty", False)]:
        g = groups[key]
        if g is None:
            values = ["--"] * 5
        else:
            m = g["metrics"]
            values = [g["Entries"] if walksRow else "--",
                      num(m[FUSED]["Event Pd (%)"]) if walksRow else "--",
                      num(m[PIR]["Event Pd (%)"]) if walksRow else "--",
                      num(m[FUSED]["Pfa (%)"]), num(m[PIR]["Pfa (%)"])]
        rows.append([title] + values)
        lines.append(f"  {title:20s} | " + " | ".join(f"{str(v):>7}" for v in values))
    tex = {"qt4_daynight": texTable("|l|c|c|c|c|c|", ["Condition", "Entries", r"Fused Pd (\%)", r"PIR Pd (\%)",
                                                      r"Fused Pfa (\%)", r"PIR Pfa (\%)"], rows,
                                    "Detection and false alarm performance under day and night conditions.",
                                    "tab:qt4_daynight")}

    multiRows = []
    lines.append("  Two people:")
    for key, title in MULTI_TYPES.items():
        cell = multiWalks[multiWalks["people"] == key] if len(multiWalks) else multiWalks
        values = [len(cell), int(cell["fusedDetected"].sum()) if len(cell) else "--",
                  int(cell["pirDetected"].sum()) if len(cell) else "--"]
        multiRows.append([title] + values)
        lines.append(f"    {key:10s}: {values[0]} entries, fused {values[1]}, PIR {values[2]}")
    total = [len(multiWalks), int(multiWalks["fusedDetected"].sum()) if len(multiWalks) else "--",
             int(multiWalks["pirDetected"].sum()) if len(multiWalks) else "--"]
    multiRows += ["hline", [r"\textbf{Total}"] + total]
    tex["qt4_multiperson"] = texTable("|l|c|c|c|", ["Entry type", "Entries", "Fused detected", "PIR detected"],
                                      multiRows, "Detections with two people in the detection zone.",
                                      "tab:qt4_multiperson")

    # Check performance for one environmental condition
    def conditionResult(walkGroup, emptyGroup, data):
        if data is None:
            return "--", "--"
        c = data["comparison"]
        m = data["metrics"][FUSED]
        ok = c["Meets Accuracy Gain"] and c["Meets Pfa Reduction"]
        return (f"Pd {num(m['Event Pd (%)'])}\\%, Pfa {num(m['Pfa (%)'])}\\%, accuracy gain "
                f"{num(c['Accuracy Gain (%)'])}\\%, Pfa reduction {num(c['Pfa Reduction (%)'])}\\%"), passText(ok)

    # Check the daytime requirement result
    day = conditionResult(groups["dayWalks"], groups["dayEmpty"], qt1Group)
    night = conditionResult(groups["nightWalks"], groups["nightEmpty"], groups["night"])
    multiOk = bool(len(multiWalks)) and bool(multiWalks["fusedDetected"].all())
    multi = ((f"{int(multiWalks['fusedDetected'].sum())}/{len(multiWalks)} entries detected", passText(multiOk))
             if len(multiWalks) else ("--", "--"))
    rows = [["Sunlight and wind-driven vegetation (day)", *day], ["Night operation", *night],
            ["More than one person in the zone", *multi]]
    for title, outcome, verdict in rows:
        lines.append(f"  {title}: {outcome.replace(chr(92), '')} -> {verdict}")
    lines += ["  Pass rules: day and night = Requirement 1 met on that data (day = the QT1 data);",
              "  two people = every two-person entry alarmed within 10 s."]
    tex["qt4_passfail"] = texTable("|l|c|c|", ["Field condition", "Outcome", "Pass/Fail"],
                                   [list(r) for r in rows], "Field condition outcomes.", "tab:qt4_passfail")
    tex["qt4_passfail"] = tex["qt4_passfail"].replace("{|l|c|c|}", "{|p{4.2cm}|p{8cm}|c|}")
    return lines, tex

# Plot the distribution of intrusion alarm delays
def figDelayHistogram(walks, out, filename="qt3_delay_histogram.png"):
    
    # Collect response delays including late detections
    delays = walks.loc[walks["fusedSeen"], "fusedResponse"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(report.FIG_WIDTH * 0.85, 2.8))
    upper = max(MAX_DELAY_S, float(np.max(delays)) if len(delays) else 0.0) + 1.0
    bins = np.arange(0, upper + 0.5, 0.5)
    ax.hist(delays, bins=bins, color=report.SYSTEM_COLOURS[0], edgecolor="white", linewidth=1)
    ax.axvline(MAX_DELAY_S, color=report.TEXT, linestyle="--", linewidth=1.2)
    ax.text(MAX_DELAY_S, ax.get_ylim()[1], f" {MAX_DELAY_S:g} s requirement", ha="right", va="top",
            color=report.TEXT_2, fontsize=8, rotation=90)
    ax.set_xlim(0, upper)
    ax.set_xlabel("Response time (s)")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("Number of entries")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / filename)
    plt.close(fig)

# Plot detection results across distance and speed
def figWalks(walks, out, filename="qt2_walks.png"):
    fig, axes = plt.subplots(1, 2, figsize=(report.FIG_WIDTH, 3.0), sharey=True)
    for ax, column, seenColumn, name, colour in [(axes[0], "fusedDetected", "fusedSeen", FUSED, report.SYSTEM_COLOURS[0]),
                                                 (axes[1], "pirDetected", "pirSeen", PIR, report.PIR_COLOUR)]:
        hit = walks[walks[column]]
        lateHit = walks[walks[seenColumn] & ~walks[column]]
        miss = walks[~walks[seenColumn]]
        
        spread = walks.groupby(["distance", "speedClass"]).cumcount() - 1
        x = walks["speed_ms"] * (1.0 + 0.12 * spread)
        ax.scatter(x[hit.index], hit["distance"], s=28, color=colour, edgecolor="white", linewidth=0.8)
        ax.scatter(x[lateHit.index], lateHit["distance"], s=30, facecolor="white", edgecolor=colour, linewidth=1.4)
        ax.scatter(x[miss.index], miss["distance"], s=36, marker="x", color="#e34948", linewidth=1.6)
        for limit in REQUIRED_SPEED_MS:
            ax.axvline(limit, color=report.TEXT_2, linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0, max(11, float(walks["distance"].max()) + 1))
        ax.set_xlabel("Target walking speed (m/s)", fontsize=9)
        
        ax.set_title(name, loc="left", fontsize=9, fontweight="normal")
    axes[0].set_ylabel("Distance from sensor (m)", fontsize=9)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=report.TEXT_2, label="Detected within 10 s"),
                            Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                                   markeredgecolor=report.TEXT_2, label="Detected after 10 s"),
                            Line2D([], [], marker="x", linestyle="", color="#e34948", label="Not detected")],
                   fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / filename)
    plt.close(fig)

# Summarise the duration and labels of one recording
def recordingSummary(folder, split):
    folder = Path(folder)
    with open(folder / "metadata.json", "r", encoding="utf-8") as metadataFile:
        metadata = json.load(metadataFile)
    times = pd.read_csv(folder / "thermopile.csv", usecols=["time_s"])["time_s"].to_numpy(dtype=float)
    start, end = (float(times.min()), float(times.max())) if len(times) else (0.0, 0.0)
    human = 0.0
    entries = 0
    labelsPath = folder / "labels.csv"
    if labelsPath.is_file():
        labels = pd.read_csv(labelsPath)
        labels = labels[labels["human"] == 1]
        entries = len(labels)
        for a, b in labels[["start_s", "end_s"]].itertuples(index=False, name=None):
            human += max(0.0, min(float(b), end) - max(float(a), start))
    stamp = metadata.get("common_start_pc_time")
    date = datetime.fromtimestamp(float(stamp)).strftime("%Y-%m-%d") if stamp else "--"
    info = parseRecordingName(folder.name)
    return {"Recording": folder.name, "Split": split, "Duration (s)": end - start, "Human (s)": human,
            "Negative (s)": (end - start) - human, "Entries": entries, "Date": date,
            "Condition": info["condition"] or "--",
            "Content": contentText(info, entries) if info["kind"] else f"{entries} entries",
            "Labelled": labelsPath.is_file()}

# Summarise recordings and label coverage for each dataset split
def datasetSummary(splits):
    from Fusion.dataset import recordingFolders
    rows, problems = [], []
    for split, path in splits.items():
        for folder in recordingFolders(path):
            try:
                rows.append(recordingSummary(folder, split))
            except (OSError, ValueError, KeyError) as error:
                problems.append(f"{folder}: {error}")
    table = pd.DataFrame(rows)
    lines = ["\nDATASET SUMMARY (all recordings, labelled time from labels.csv)",
             *[f"  Skipped {problem}" for problem in problems],
             f"  {'Split':10s} | {'Recordings':>10} | {'Total min':>9} | {'Human min':>9} | {'Negative min':>12} | Entries"]
    if table.empty:
        return lines + ["  No recordings found."], table, ""
    for split in list(splits) + ["All"]:
        part = table if split == "All" else table[table["Split"] == split]
        lines.append(f"  {split:10s} | {len(part):10d} | {part['Duration (s)'].sum() / 60:9.1f} | "
                     f"{part['Human (s)'].sum() / 60:9.1f} | {part['Negative (s)'].sum() / 60:12.1f} | "
                     f"{int(part['Entries'].sum())}")
    unlabelled = table.loc[~table["Labelled"], "Recording"].tolist()
    if unlabelled:
        lines.append(f"  Not labelled yet (no labels.csv): {', '.join(unlabelled)}")
    lines += ["", f"  {'Recording':32s} | {'Split':10s} | {'Date':10s} | {'Cond.':5s} | {'Min':>5} | "
                  f"{'Human min':>9} | Content"]
    for _, r in table.iterrows():
        lines.append(f"  {r['Recording']:32s} | {r['Split']:10s} | {r['Date']:10s} | {r['Condition']:5s} | "
                     f"{r['Duration (s)'] / 60:5.1f} | {r['Human (s)'] / 60:9.1f} | {r['Content'].replace('~', ' ')}")
    texRows = [[texEscape(r["Recording"]), r["Content"], r["Condition"], r["Date"], r["Split"]]
               for _, r in table.iterrows()]
    tex = texTable("|l|X|l|l|l|", ["Recording", "Content", "Day/night", "Date", "Purpose"], texRows,
                   "Recordings used for training, validation and testing.", "tab:data_split", tabularx=True)
    return lines, table, tex

# Run all qualification tests and save the final results
def writeQualificationReport(runName, evalData, scores, model, outDir, description, splits):
    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "tables").mkdir()
    warnings = []

    data = evalData.copy()
    data["fusedScore"] = np.asarray(scores, dtype=float)
    from Fusion.evaluation import pirBaselineStates, runStateMachine
    data["pirState"] = pirBaselineStates(data)
    data["fusedState"] = runStateMachine(data, data["fusedScore"].to_numpy(), model.thresholds)

    names = sorted({Path(r).name for r in data["recording"]})
    infos = {name: parseRecordingName(name) for name in names}
    ignored = [name for name, info in infos.items() if info["kind"] is None]
    if ignored:
        warnings.append("Folder names not in the <Site>_<Test>_<Day|Night> format (left out of every test): "
                        + ", ".join(ignored))
    if not any(info["site"] == SEEN_SITE for info in infos.values()):
        warnings.append(f"No recordings from the seen site '{SEEN_SITE}' (change SEEN_SITE in Fusion/qualification.py).")

    # Define the recording groups used for each qualification test
    groupNames = {
        "qt1": selectNames(infos, site=SEEN_SITE, condition="Day", kinds=("distance", "edge", "empty")),
        "dayWalks": selectNames(infos, site=SEEN_SITE, condition="Day", kinds=("distance", "edge")),
        "dayEmpty": selectNames(infos, site=SEEN_SITE, condition="Day", kinds=("empty",)),
        "night": selectNames(infos, site=SEEN_SITE, condition="Night", kinds=("distance", "edge", "empty")),
        "nightWalks": selectNames(infos, site=SEEN_SITE, condition="Night", kinds=("distance", "edge")),
        "nightEmpty": selectNames(infos, site=SEEN_SITE, condition="Night", kinds=("empty",)),
        "multi": selectNames(infos, site=SEEN_SITE, kinds=("multi",)),
        "unseen": selectNames(infos, unseen=True, condition="Day", kinds=("distance", "edge", "empty")),
    }

    # Calculate metrics for each test group
    groups = {key: groupMetrics(subset(data, value)) if value else None for key, value in groupNames.items()}

    named = subset(data, [n for n in names if n not in ignored])
    walks = walkTable(named, warnings) if not named.empty else walkTable(data.iloc[0:0], warnings)
    isSeen = walks["site"] == SEEN_SITE
    qt1Walks = walks[isSeen & (walks["condition"] == "Day") & walks["kind"].isin(["distance", "edge"])]
    qt2Walks = walks[isSeen & (walks["condition"] == "Day") & (walks["kind"] == "distance")]
    edgeWalks = walks[isSeen & (walks["condition"] == "Day") & (walks["kind"] == "edge")]
    multiWalks = walks[isSeen & (walks["kind"] == "multi")]

    lines = [f"QUALIFICATION TESTS: {runName}", description,
             f"Seen site: {SEEN_SITE}. Model window {model.window:g} s, alarm thresholds {model.thresholds}.",
             *report.DEFINITIONS, "", "RECORDINGS PER TEST"]
    for key, value in groupNames.items():
        lines.append(f"  {key:10s}: {', '.join(value) if value else '-'}")

    tex = {}

    # Generate Requirement 1 results
    qt1Lines, tex["qt1_results"] = requirementOneTables(
        groups["qt1"], "tab:qt1_results",
        "Fused system compared to the thresholded PIR on the testing recordings ($n = {n}$ entries, "
        "{negatives} non-human intervals).", "QT1: FUSED SYSTEM VS PIR ON ITS OWN (Requirement 1)")
    lines += qt1Lines
    
    inZone = lambda w: w.assign(fusedDetected=w["fusedSeen"], pirDetected=w["pirSeen"])

    # Generate range, speed and edge results
    qt2Lines, qt2Tex, _ = qt2Tables(inZone(qt2Walks), inZone(edgeWalks), warnings)
    qt2Lines.insert(2, f"  (detected = INTRUDER while the person is in the zone; within 10 s: fused "
                       f"{int(qt2Walks['fusedDetected'].sum())}/{len(qt2Walks)}, PIR {int(qt2Walks['pirDetected'].sum())}/{len(qt2Walks)})")
    lines += qt2Lines
    tex.update(qt2Tex)

    # Generate response time results
    qt3Lines, qt3Tex, _ = qt3Tables(qt1Walks, qt2Walks)
    lines += qt3Lines
    tex.update(qt3Tex)

    # Generate environmental and multiple person results
    qt4Lines, qt4Tex = qt4Tables(groups, multiWalks, groups["qt1"])
    lines += qt4Lines
    tex.update(qt4Tex)
    unseenLines, tex["unseen_results"] = requirementOneTables(
        groups["unseen"], "tab:unseen_results",
        "Fused system compared to the thresholded PIR at a site not used for training ($n = {n}$ entries, "
        "{negatives} non-human intervals).", "UNSEEN SITE (same test as QT1, not a proposal requirement)")
    lines += unseenLines

    lines += ["\nEVERY WALK",
              f"  {'Recording':30s} | {'#':>2} | {'Time s':>7} | {'Dur s':>6} | {'Speed':>6} | {'Class':9s} | "
              f"{'Fused delay':>11} | {'PIR delay':>9}"]
    for _, w in walks.iterrows():
        lines.append(f"  {w['recording']:30s} | {int(w['walk']):2d} | {w['start_s']:7.1f} | {w['duration_s']:6.1f} | "
                     f"{num(w['speed_ms'], 2):>6} | {str(w['speedClass'] or '-'):9s} | "
                     f"{num(w['fusedDelay'], 2) if w['fusedDetected'] else 'missed':>11} | "
                     f"{num(w['pirDelay'], 2) if w['pirDetected'] else 'missed':>9}")

    # Export individual walk results for inspection
    walks.to_csv(out / "walks.csv", index=False)

    # Summarise the training, validation and testing datasets
    summaryLines, _, splitTex = datasetSummary(splits)
    lines += summaryLines
    if splitTex:
        tex["data_split"] = splitTex

    systems = {PIR: {"states": data["pirState"].to_numpy()},
               FUSED: {"scores": data["fusedScore"].to_numpy(), "thresholds": model.thresholds,
                       "states": data["fusedState"].to_numpy()}}
    colours = report._colours(systems)
    for key, prefix in (("qt1", "qt1"), ("unseen", "unseen")):
        if groups[key] is not None:
            report.figRequirements(groups[key]["metrics"], colours, prefix, out, f"{prefix}_requirements.png")
            report.figImprovements(groups[key]["metrics"], colours, prefix, out, f"{prefix}_improvements.png")
    if len(qt2Walks):
        figWalks(qt2Walks, out)
    if len(qt1Walks):
        figDelayHistogram(qt1Walks, out)

    # Prepare the individual recording timelines
    timelineFolder = out / "timelines"
    timelineFolder.mkdir()
    report.figTimelines(data, systems, colours, "testing", timelineFolder)
    for name in groupNames["night"] + groupNames["multi"]:
        source = timelineFolder / f"timeline_{name}.png"
        if source.is_file():
            shutil.copy(source, out / f"qt4_timeline_{name}.png")

    if groupNames["qt1"]:
        qt1Data = subset(data, groupNames["qt1"])
        qt1Systems = {PIR: {"states": qt1Data["pirState"].to_numpy()},
                      FUSED: {"scores": qt1Data["fusedScore"].to_numpy(), "thresholds": model.thresholds,
                              "states": qt1Data["fusedState"].to_numpy()}}
        report.writeEvaluationReport(f"{runName} QT1 details", qt1Data, qt1Systems, out / "details",
                                     "QT1 recordings only (seen site, day).", "QT1",
                                     featureNames=model.features, importance=model.importance, timelines=False)

    order = ["qt1_results", "qt2_range_speed", "qt2_edges", "qt2_passfail", "qt3_response", "qt3_speed_delay",
             "qt4_daynight", "qt4_multiperson", "qt4_passfail", "unseen_results", "data_split"]

    # Collect the LaTeX tables into one file
    combined = ["% Generated by offlineRun.py --test. Numbers only; check them against report.txt.\n"]
    for key in order:
        if tex.get(key):
            (out / "tables" / f"{key}.tex").write_text(tex[key], encoding="utf-8")
            combined.append(f"% ---- {key} ----\n{tex[key]}")
    (out / "tables.tex").write_text("\n".join(combined), encoding="utf-8")

    if warnings:
        lines += ["\nWARNINGS"] + [f"  - {w}" for w in warnings]
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    return text
