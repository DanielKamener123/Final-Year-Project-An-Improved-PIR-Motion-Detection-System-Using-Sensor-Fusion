"""Contrast analysis: when can the thermopile tell a person apart from the background?

Run from the project root:
    python -m Tools.contrast_analysis

Uses every labelled recording (training and validation). Each thermal frame is run
through the real ThermopilePipeline, and the residual (frame minus the pipeline's
own background, corrected for the scene shift) is measured. Per frame:

    peak residual = how much warmer the hottest pixel is than its background (C)

Frames are split into PERSON frames (someone labelled inbound) and EMPTY frames
(nobody inbound, and nobody outbound when labels(outbound).csv exists). Per recording:

    clutter ceiling  = median peak residual of EMPTY frames: how warm the hottest
                       pixel gets with nobody there (sensor noise + sun, wind, plants)
    person peak      = median peak residual of PERSON frames
    SCR              = person peak / clutter ceiling (signal-to-clutter ratio)
    separability     = AUC of the peak residual, PERSON vs EMPTY frames
                       (1 = always separable, 0.5 = a coin flip)

When SCR is close to 1 the person is no warmer than the empty background's own
hottest pixel, and separability falls to chance: no thermal processing can find the
person because the information is not in the frame.

Figures (highest- and lowest-SCR recordings):
    peak_histograms.png  hottest-pixel distributions, person vs nobody present
    example_frames.png   typical frames, person vs nobody present

Writes report.txt and the figures to Processed/Contrast_Analysis/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import package from project root even when this script is run by filepath.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Fusion.dataset import loadExcludedIntervals, loadLabels, recordingFolders
from Fusion.report import PIR_COLOUR, SYSTEM_COLOURS
from Thermopile.processing import ThermopilePipeline

DATASETS = [Path("Datasets/training"), Path("Datasets/validation")]
OUTPUT = Path("Processed/Contrast_Analysis")
MIN_FRAMES = 10            # each class needs at least this many frames
DEAD_PIXEL_C = (-50, 150)  # outside this range a pixel is treated as dead
PERSON_COLOUR = SYSTEM_COLOURS[1]
EMPTY_COLOUR = PIR_COLOUR


# Area Under The ROC Curve (first principles): chance that a random PERSON frame
# has a higher value than a random EMPTY frame (ties count half)
def auc(person, empty):
    values = np.r_[person, empty]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sortedValues = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sortedValues[j + 1] == sortedValues[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = len(person), len(empty)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# Interval Labels At Each Time
def intervalFlags(intervals, times):
    flags = np.zeros(len(times), dtype=bool)
    if intervals is None or intervals.empty:
        return flags
    for start, end, human in intervals[["start_s", "end_s", "human"]].itertuples(index=False, name=None):
        if int(human) == 1:
            flags |= (times >= float(start)) & (times < float(end))
    return flags


# Peak Residual Of Every Frame Of One Recording, Plus Its Frame Classes
def measureRecording(recording):
    labels = loadLabels(recording)
    if labels is None:
        return None
    with open(recording / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    thermal = pd.read_csv(recording / "thermopile.csv").sort_values("time_s")
    frames = thermal[[f"pixel_{i}" for i in range(768)]].to_numpy(dtype=np.float64).reshape(-1, 24, 32)
    times = thermal["time_s"].to_numpy(dtype=np.float64)
    dead = ((frames < DEAD_PIXEL_C[0]) | (frames > DEAD_PIXEL_C[1])).any(axis=0)

    pipeline = ThermopilePipeline()
    pipeline.load_calibration(meta["thermal_background"], meta["thermal_sigma"])
    peaks = np.zeros(len(frames))
    residuals = np.zeros_like(frames)
    valid = np.ones(len(frames), dtype=bool)
    for k, (frame, t) in enumerate(zip(frames, times)):
        background = pipeline.bg_model.bg.copy()
        pipeline.process(frame, t)
        valid[k] = pipeline.last_frame_valid          # glitch frames are left out
        residual = frame - (background + pipeline.last_shift)
        residual[dead] = 0.0
        residuals[k] = residual
        peaks[k] = residual.max()

    for start, end in loadExcludedIntervals(recording):
        valid &= ~((times >= start) & (times <= end))
    person = intervalFlags(labels, times)
    outboundPath = recording / "labels(outbound).csv"
    outbound = intervalFlags(pd.read_csv(outboundPath), times) if outboundPath.is_file() else np.zeros(len(times), bool)
    personFrames = valid & person
    emptyFrames = valid & ~person & ~outbound
    if personFrames.sum() < MIN_FRAMES or emptyFrames.sum() < MIN_FRAMES:
        return None

    return {
        "name": f"{recording.name} ({recording.parent.name[:5]})",
        "frames": frames, "residuals": residuals, "peaks": peaks,
        "person": personFrames, "empty": emptyFrames,
        "backgroundC": float(np.median(np.asarray(meta["thermal_background"]))),
        "outboundLabels": outboundPath.is_file(),
    }


def summarise(m):
    personPeak = float(np.median(m["peaks"][m["person"]]))
    clutter = float(np.median(m["peaks"][m["empty"]]))
    return {
        "Recording": m["name"],
        "Background (C)": m["backgroundC"],
        "Clutter ceiling (C)": clutter,
        "Person peak (C)": personPeak,
        "SCR": personPeak / clutter,
        "Separability (AUC)": auc(m["peaks"][m["person"]], m["peaks"][m["empty"]]),
        "Person frames": int(m["person"].sum()),
        "Empty frames": int(m["empty"].sum()),
        "Outbound labels": "yes" if m["outboundLabels"] else "no",
    }


def figHistograms(measured, table, out):
    order = table.sort_values("SCR")
    picks = [order.iloc[-1]["Recording"], order.iloc[0]["Recording"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, name, title in zip(axes, picks, ("High contrast", "Low contrast")):
        m = next(r for r in measured if r["name"] == name)
        row = table[table["Recording"] == name].iloc[0]
        bins = np.linspace(0, max(np.percentile(m["peaks"][m["person"] | m["empty"]], 99), 1.0), 40)
        ax.hist(m["peaks"][m["empty"]], bins=bins, density=True, alpha=0.65, color=EMPTY_COLOUR, label="Nobody present")
        ax.hist(m["peaks"][m["person"]], bins=bins, density=True, alpha=0.65, color=PERSON_COLOUR, label="Person present")
        ax.set_title(f"{title}: {name}\nSCR {row['SCR']:.2f}, separability {row['Separability (AUC)']:.2f}",
                     loc="left", fontsize=9)
        ax.set_xlabel("Hottest pixel above background (C)")
        ax.set_ylabel("Fraction of frames")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "peak_histograms.png")
    plt.close(fig)


def figExampleFrames(measured, table, out):
    order = table.sort_values("SCR")
    picks = [order.iloc[-1]["Recording"], order.iloc[0]["Recording"]]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    for r, name in enumerate(picks):
        m = next(x for x in measured if x["name"] == name)
        for c, cls in enumerate(("empty", "person")):
            idx = np.flatnonzero(m[cls])
            # A typical frame of this class: peak residual closest to the class median
            k = idx[np.argmin(np.abs(m["peaks"][idx] - np.median(m["peaks"][idx])))]
            ax = axes[r, c]
            image = ax.imshow(m["residuals"][k], cmap="inferno", vmin=0, vmax=4)
            ax.set_xticks([])
            ax.set_yticks([])
            label = "Nobody present" if cls == "empty" else "Person present"
            ax.set_title(f"{name}: {label}\nhottest pixel +{m['peaks'][k]:.1f} C", loc="left", fontsize=8)
    fig.colorbar(image, ax=axes, label="Above background (C)", shrink=0.8)
    fig.suptitle("Typical frames (temperature above the pipeline's background)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.savefig(out / "example_frames.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Thermal contrast vs separability on the labelled recordings")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    measured = []
    for dataset in DATASETS:
        for recording in recordingFolders(dataset):
            print(f"Measuring {recording}...")
            m = measureRecording(recording)
            if m is not None:
                measured.append(m)
    if not measured:
        raise ValueError("No labelled recordings with both person and empty frames")
    table = pd.DataFrame([summarise(m) for m in measured]).sort_values("SCR").reset_index(drop=True)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    figHistograms(measured, table, out)
    figExampleFrames(measured, table, out)

    ranks = table[["SCR", "Separability (AUC)"]].rank()
    rankCorrelation = float(np.corrcoef(ranks["SCR"], ranks["Separability (AUC)"])[0, 1])
    lines = ["THERMAL CONTRAST ANALYSIS", __doc__.split("Figures (")[0].strip(), "",
             f"Rank correlation between SCR and separability: {rankCorrelation:.2f}", ""]
    header = (f"{'Recording':28s} | {'Bg C':>5} | {'Clutter':>7} | "
              f"{'Person':>6} | {'SCR':>5} | {'AUC':>5} | Person/Empty frames | Outbound")
    lines += [header, "-" * len(header)]
    for _, r in table.iterrows():
        lines.append(f"{r['Recording']:28s} | {r['Background (C)']:5.1f} | "
                     f"{r['Clutter ceiling (C)']:7.2f} | {r['Person peak (C)']:6.2f} | "
                     f"{r['SCR']:5.2f} | {r['Separability (AUC)']:5.2f} | {int(r['Person frames']):6d}/{int(r['Empty frames']):<6d}"
                     f"       | {r['Outbound labels']}")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Saved report and figures to {out}")


if __name__ == "__main__":
    main()
