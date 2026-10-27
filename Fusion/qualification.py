"""Qualification tests (report Section 4) from the testing recordings.

offlineRun.py --test runs the exported model once over Datasets/testing; this
module splits that single run into the report's qualification tests:

  QT1  Requirement 1      seen site, day: distance walks, edge walks and empty scenes
  QT2  Requirements 2, 3  seen site, day: distance walks (range, speed), edge walks (field of view)
  QT3  Requirement 4      detection delay of the QT1 entries (per speed class from QT2)
  QT4  Field conditions   day vs night, two people in the zone
  Unseen site             same as QT1 for any site other than SEEN_SITE (not a proposal requirement)

Recording folder names carry the test information:
    <Site>_<Test>_<Day|Night>[_anything]
    Test = <d>m                  walks across the zone at d metres (3 walks per speed class)
           Edge+55 / Edge-55     walks along a field of view edge
           Empty                 nobody enters
           2People-SideBySide    two people entering together
           2People-Apart         two people entering a few seconds apart
    e.g. Garden_4m_Day, Garden_Edge+55_Day, Garden_Empty_Night, Garden_2People-Apart_Day

Walk speed = walking line length / labelled time inside the zone. The walking
line length is the measured value in PATH_LENGTH_M, or 2 d tan(55 deg) if it
was not measured. Within each distance recording the walks are sorted by
speed: the slowest quarter is "Very slow", then "Slow", "Medium" and "Fast".

Outputs (all in the run folder):
    report.txt              every number, the pass rules and any warnings
    tables/*.tex            one LaTeX table per report table (same labels as Section 4)
    tables.tex              all of them in one file
    qt*.png, unseen_*.png   report figures
    timelines/              one timeline per testing recording
    details/                full evaluation (Ward, false alarms, operating curve) of the QT1 data
"""
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


# Config
SEEN_SITE = "Garden"                 # site the model was trained at (first word of the folder name)
HALF_FOV_DEG = 55.0                  # half of the 110 deg detection zone
# Measured walking line length (m) at each distance; None = use 2 d tan(55 deg)
PATH_LENGTH_M = {1: None, 4: None, 6: None, 8: None, 10: None}
SPEED_CLASSES = ("Very slow", "Slow", "Medium", "Fast")
WALKS_PER_CLASS = 3
REQUIRED_RANGE_M = (1.0, 10.0)       # Requirement 2
REQUIRED_SPEED_MS = (0.2, 5.0)       # Requirement 3

PIR = "Thresholded PIR"
FUSED = "Fused system"
MULTI_TYPES = {"sidebyside": "Side by side", "apart": r"Separated by $\sim$3~s"}


# ---------------------------------------------------------------------------
# Recording names
# ---------------------------------------------------------------------------

# Reads The Test Information From A Recording Folder Name
def parseRecordingName(name):
    info = {"name": name, "site": None, "kind": None, "distance": None, "edge": None,
            "people": None, "condition": None}
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


def pathLength(distance):
    measured = PATH_LENGTH_M.get(int(round(distance))) if float(distance).is_integer() else None
    if measured:
        return float(measured), True
    return 2.0 * distance * math.tan(math.radians(HALF_FOV_DEG)), False


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


# ---------------------------------------------------------------------------
# Per-walk results
# ---------------------------------------------------------------------------

def firstIntruder(times, states, s, e):
    deadline = min(times[s] + MAX_DELAY_S, times[e])
    for k in range(s, e + 1):
        if times[k] > deadline:
            break
        if states[k] == INTRUDER:
            return float(times[k] - times[s])
    return None


# One Row Per Labelled Entry (same detection rule as the Pd in Fusion/evaluation.py)
def walkTable(data, warnings):
    rows = []
    fused = data["fusedState"].to_numpy()
    pir = data["pirState"].to_numpy()
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
        for number, (s, e) in enumerate(item["entries"], start=1):
            match = [(a, b) for a, b in intervals if a <= times[s] < b]
            duration = float(match[0][1] - match[0][0]) if match else float(times[e] - times[s])
            speed = None
            if info["kind"] == "distance" and duration > 0:
                speed = pathLength(info["distance"])[0] / duration
            fusedDelay = firstIntruder(times, fusedStates, s, e)
            pirDelay = firstIntruder(times, pirStates, s, e)
            rows.append({"recording": name, "site": info["site"], "kind": info["kind"],
                         "condition": info["condition"], "distance": info["distance"], "edge": info["edge"],
                         "people": info["people"], "walk": number, "start_s": float(times[s]),
                         "duration_s": duration, "speed_ms": speed,
                         "fusedDetected": fusedDelay is not None, "fusedDelay": fusedDelay,
                         "pirDetected": pirDelay is not None, "pirDelay": pirDelay})
    walks = pd.DataFrame(rows, columns=["recording", "site", "kind", "condition", "distance", "edge", "people",
                                        "walk", "start_s", "duration_s", "speed_ms", "fusedDetected",
                                        "fusedDelay", "pirDetected", "pirDelay"])
    for column in ("duration_s", "speed_ms", "fusedDelay", "pirDelay", "distance"):
        walks[column] = pd.to_numeric(walks[column], errors="coerce")
    walks["fusedDetected"] = walks["fusedDetected"].astype(bool)
    walks["pirDetected"] = walks["pirDetected"].astype(bool)
    walks["speedClass"] = None
    for name, group in walks[walks["kind"] == "distance"].groupby("recording"):
        expected = WALKS_PER_CLASS * len(SPEED_CLASSES)
        if len(group) != expected:
            warnings.append(f"{name}: {len(group)} walks instead of {expected}; speed classes split as evenly as possible.")
        order = group.sort_values("speed_ms").index.to_numpy()
        for speedClass, members in zip(SPEED_CLASSES, np.array_split(order, len(SPEED_CLASSES))):
            walks.loc[members, "speedClass"] = speedClass
    return walks


# ---------------------------------------------------------------------------
# Groups of recordings
# ---------------------------------------------------------------------------

def subset(data, names):
    return data[data["recording"].map(lambda r: Path(r).name in names)].reset_index(drop=True)


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


# Pd / Pfa / accuracy of both systems on one group, plus the Requirement 1 comparison
def groupMetrics(data):
    if data.empty:
        return None
    recordings = Recordings(data)
    metrics = {PIR: evaluateStates(data, data["pirState"].to_numpy(), recordings),
               FUSED: evaluateStates(data, data["fusedState"].to_numpy(), recordings)}
    comparison = compareToPir(metrics[FUSED], metrics[PIR])
    durations = intervalSeconds(data)
    human = data["human"].to_numpy() == 1
    extra = {"Entries": int(metrics[FUSED]["Events"]), "Negative Intervals": int((~human).sum()),
             "Human Seconds": float(durations[human].sum()), "Negative Seconds": float(durations[~human].sum())}
    return {"metrics": metrics, "comparison": comparison, **extra}


def intervalSeconds(data):
    from Fusion.evaluation import intervalDurations
    seconds = np.zeros(len(data))
    for item in Recordings(data).items:
        seconds[item["idx"]] = intervalDurations(item["times"])
    return seconds


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def num(value, decimals=1):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{float(value):.{decimals}f}"


def passText(ok):
    return "PASS" if ok else "FAIL"


def texEscape(text):
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


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


# ---------------------------------------------------------------------------
# Tables (text for report.txt, LaTeX for the report)
# ---------------------------------------------------------------------------

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


def qt2Tables(walks, edgeWalks, warnings):
    lines = ["\nQT2: RANGE, FIELD OF VIEW AND SPEED (Requirements 2 and 3)",
             "  Cells: walks detected by fused system / thresholded PIR (out of n)."]
    tex = {}
    if walks.empty:
        lines.append("  No distance recordings.")
    else:
        distances = sorted(walks["distance"].unique())
        rows, textRows = [], []
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
        rows += ["hline", [r"\textbf{Total}"] + totals, "hline", [r"\textbf{Measured speed (m/s)}"] + speeds]
        counts = walks.groupby(["distance", "speedClass"]).size()
        cellText = (f"Each cell contains {int(counts.iloc[0])} walks." if counts.nunique() == 1
                    else "The number of walks per cell is given in report.txt.")
        tex["qt2_range_speed"] = texTable(
            "|c|c|c|c|c|c|", ["Distance (m)"] + list(SPEED_CLASSES) + ["Total"], rows,
            "Entries detected per distance and speed class, given as fused system / thresholded PIR. " + cellText,
            "tab:qt2_range_speed")
        measured = [f"  Walking line length used: " + ", ".join(
            f"{d:g} m -> {pathLength(d)[0]:.2f} m ({'measured' if pathLength(d)[1] else '2 d tan 55'})"
            for d in distances)]
        lines += measured
        if not all(pathLength(d)[1] for d in distances):
            warnings.append("PATH_LENGTH_M is not filled in for every distance; speeds use 2 d tan(55 deg).")

    # Edge walks
    lines.append("  Edge walks:")
    edgeRows = []
    for edge in ("+55", "-55"):
        e = edgeWalks[edgeWalks["edge"] == edge] if not edgeWalks.empty else edgeWalks
        lines.append(f"    {edge} deg: {len(e)} walks, fused {int(e['fusedDetected'].sum()) if len(e) else 0}, "
                     f"PIR {int(e['pirDetected'].sum()) if len(e) else 0}")
        edgeRows.append([rf"${edge}^\circ$", len(e), int(e["fusedDetected"].sum()) if len(e) else "--",
                         int(e["pirDetected"].sum()) if len(e) else "--"])
    tex["qt2_edges"] = texTable("|c|c|c|c|", ["Edge", "Walks", "Fused detected", "PIR detected"], edgeRows,
                                r"Detections for walks along the edges of the $110^\circ$ field of view.",
                                "tab:qt2_edges")

    # Pass / fail
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


def qt3Tables(qt1Walks, qt2Walks):
    lines = ["\nQT3: RESPONSE TIME (Requirement 4)"]
    detected = qt1Walks[qt1Walks["fusedDetected"]]
    meanDelay = detected["fusedDelay"].mean() if len(detected) else np.nan
    maxDelay = detected["fusedDelay"].max() if len(detected) else np.nan
    alarmed = f"{len(detected)}/{len(qt1Walks)}"
    delayOk = bool(len(qt1Walks)) and len(detected) == len(qt1Walks) and maxDelay <= MAX_DELAY_S
    lines += [f"  Entries alarmed (INTRUDER within {MAX_DELAY_S:g} s): {alarmed}",
              f"  Mean detection delay: {num(meanDelay, 2)} s, maximum: {num(maxDelay, 2)} s",
              "  Processing time per frame is not measured by this script; worst-case response time =",
              f"  maximum delay ({num(maxDelay, 2)} s) + maximum processing time.",
              f"  Detection delay alone within {MAX_DELAY_S:g} s for every entry: {passText(delayOk)}"]
    rows = [["Entries alarmed (INTRUDER)", alarmed],
            ["Mean detection delay (s)", num(meanDelay, 2)],
            ["Maximum detection delay (s)", num(maxDelay, 2)],
            "hline",
            ["Mean processing time per frame (ms)", "XX"],
            ["Maximum processing time per frame (ms)", "XX"],
            "hline",
            [r"\textbf{Worst-case response time (s)}", r"\textbf{XX}"],
            [r"\textbf{Requirement}", r"$\leq 10$~s"],
            [r"\textbf{Pass/Fail}", f"XX % detection delay alone: {passText(delayOk)}"]]
    tex = {"qt3_response": texTable(
        "|l|c|", ["Measurement", "Value"], rows,
        "Response time of the fused system on the testing recordings. Worst-case response time is the "
        "maximum detection delay plus the maximum processing time.", "tab:qt3_response")}
    # The Pass/Fail cell holds a LaTeX comment, so end that line properly
    tex["qt3_response"] = tex["qt3_response"].replace(
        f"XX % detection delay alone: {passText(delayOk)} \\\\",
        f"XX \\\\ % detection delay alone: {passText(delayOk)}")

    speedRows = []
    lines.append(f"  {'Speed class':10s} | {'Alarmed':>7} | {'Mean s':>6} | {'Max s':>6}")
    for speedClass in SPEED_CLASSES:
        cell = qt2Walks[qt2Walks["speedClass"] == speedClass] if len(qt2Walks) else qt2Walks
        hit = cell[cell["fusedDetected"]] if len(cell) else cell
        values = [f"{len(hit)}/{len(cell)}", num(hit["fusedDelay"].mean() if len(hit) else np.nan, 2),
                  num(hit["fusedDelay"].max() if len(hit) else np.nan, 2)]
        speedRows.append([speedClass] + values)
        lines.append(f"  {speedClass:10s} | {values[0]:>7} | {values[1]:>6} | {values[2]:>6}")
    tex["qt3_speed_delay"] = texTable("|l|c|c|c|", ["Speed class", "Entries alarmed", "Mean delay (s)",
                                                    "Maximum delay (s)"], speedRows,
                                      "Detection delay of the fused system per speed class.",
                                      "tab:qt3_speed_delay")
    return lines, tex, delayOk


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

    # Pass / fail: day and night = Requirement 1 met on that data; two people = every entry alarmed
    def conditionResult(walkGroup, emptyGroup, data):
        if data is None:
            return "--", "--"
        c = data["comparison"]
        m = data["metrics"][FUSED]
        ok = c["Meets Accuracy Gain"] and c["Meets Pfa Reduction"]
        return (f"Pd {num(m['Event Pd (%)'])}\\%, Pfa {num(m['Pfa (%)'])}\\%, accuracy gain "
                f"{num(c['Accuracy Gain (%)'])}\\%, Pfa reduction {num(c['Pfa Reduction (%)'])}\\%"), passText(ok)

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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figDelayHistogram(walks, out, filename="qt3_delay_histogram.png"):
    delays = walks.loc[walks["fusedDetected"], "fusedDelay"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(report.FIG_WIDTH * 0.85, 2.8))
    bins = np.arange(0, MAX_DELAY_S + 0.5, 0.5)
    ax.hist(delays, bins=bins, color=report.SYSTEM_COLOURS[0], edgecolor="white", linewidth=1)
    ax.axvline(MAX_DELAY_S, color=report.TEXT, linestyle="--", linewidth=1.2)
    ax.text(MAX_DELAY_S, ax.get_ylim()[1], f" {MAX_DELAY_S:g} s requirement", ha="right", va="top",
            color=report.TEXT_2, fontsize=8, rotation=90)
    ax.set_xlim(0, MAX_DELAY_S + 0.5)
    ax.set_xlabel("Detection delay (s)")
    ax.set_ylabel("Number of entries")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / filename)
    plt.close(fig)


# Every Distance Walk: speed vs distance, filled = detected, cross = missed
def figWalks(walks, out, filename="qt2_walks.png"):
    fig, axes = plt.subplots(1, 2, figsize=(report.FIG_WIDTH, 3.0), sharey=True)
    for ax, column, name, colour in [(axes[0], "fusedDetected", FUSED, report.SYSTEM_COLOURS[0]),
                                     (axes[1], "pirDetected", PIR, report.PIR_COLOUR)]:
        hit = walks[walks[column]]
        miss = walks[~walks[column]]
        ax.scatter(hit["speed_ms"], hit["distance"], s=28, color=colour, edgecolor="white", linewidth=0.8)
        ax.scatter(miss["speed_ms"], miss["distance"], s=36, marker="x", color="#e34948", linewidth=1.6)
        for limit in REQUIRED_SPEED_MS:
            ax.axvline(limit, color=report.TEXT_2, linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0, max(11, float(walks["distance"].max()) + 1))
        ax.set_xlabel("Measured walking speed (m/s)", fontsize=9)
        # Panel label (which system), kept even though figure titles are off
        ax.set_title(name, loc="left", fontsize=9, fontweight="normal")
    axes[0].set_ylabel("Distance from sensor (m)", fontsize=9)
    axes[1].legend(handles=[Line2D([], [], marker="o", linestyle="", color=report.TEXT_2, label="Detected"),
                            Line2D([], [], marker="x", linestyle="", color="#e34948", label="Missed")],
                   fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset summary (Section 3 data paragraph and the data split table)
# ---------------------------------------------------------------------------

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


def datasetSummary(splits):
    """splits: {"Training": Path, "Validation": Path, "Testing": Path}"""
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


# ---------------------------------------------------------------------------
# Main entry: offlineRun.py --test
# ---------------------------------------------------------------------------

def writeQualificationReport(runName, evalData, scores, model, outDir, description, splits):
    """evalData: testing data with outbound labels; scores: fused confidence per row."""
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
    qt1Lines, tex["qt1_results"] = requirementOneTables(
        groups["qt1"], "tab:qt1_results",
        "Fused system compared to the thresholded PIR on the testing recordings ($n = {n}$ entries, "
        "{negatives} non-human intervals).", "QT1: FUSED SYSTEM VS PIR ON ITS OWN (Requirement 1)")
    lines += qt1Lines
    qt2Lines, qt2Tex, _ = qt2Tables(qt2Walks, edgeWalks, warnings)
    lines += qt2Lines
    tex.update(qt2Tex)
    qt3Lines, qt3Tex, _ = qt3Tables(qt1Walks, qt2Walks)
    lines += qt3Lines
    tex.update(qt3Tex)
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
    walks.to_csv(out / "walks.csv", index=False)

    summaryLines, _, splitTex = datasetSummary(splits)
    lines += summaryLines
    if splitTex:
        tex["data_split"] = splitTex

    # Figures
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
    timelineFolder = out / "timelines"
    timelineFolder.mkdir()
    report.figTimelines(data, systems, colours, "testing", timelineFolder)
    for name in groupNames["night"] + groupNames["multi"]:
        source = timelineFolder / f"timeline_{name}.png"
        if source.is_file():
            shutil.copy(source, out / f"qt4_timeline_{name}.png")

    # Full evaluation (Ward, false alarms, operating curve) of the QT1 data
    if groupNames["qt1"]:
        qt1Data = subset(data, groupNames["qt1"])
        qt1Systems = {PIR: {"states": qt1Data["pirState"].to_numpy()},
                      FUSED: {"scores": qt1Data["fusedScore"].to_numpy(), "thresholds": model.thresholds,
                              "states": qt1Data["fusedState"].to_numpy()}}
        report.writeEvaluationReport(f"{runName} QT1 details", qt1Data, qt1Systems, out / "details",
                                     "QT1 recordings only (seen site, day).", "QT1",
                                     featureNames=model.features, importance=model.importance, timelines=False)

    # LaTeX tables
    order = ["qt1_results", "qt2_range_speed", "qt2_edges", "qt2_passfail", "qt3_response", "qt3_speed_delay",
             "qt4_daynight", "qt4_multiperson", "qt4_passfail", "unseen_results", "data_split"]
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
