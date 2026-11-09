"""Requirement 1 supporting analysis (report Section 4, QT1 statistical analysis).

Runs the exported model (Processed/realtime/fusion_gb.json) with its frozen alarm
thresholds over the testing recordings, exactly as offlineRun.py --test does, and adds
the evidence that the Requirement 1 result is not a chance or a tuning effect:

  1. Accuracy split into its two parts (human intervals / non-human intervals)
  2. Every recording on its own, with a sign test over the recordings
  3. Block bootstrap 95% confidence intervals of the accuracy gain and the
     false alarm reduction (whole blocks of time are resampled, because
     neighbouring fusion intervals are not independent)
  4. PIR threshold sweep: the best any PIR threshold can do on the same data
  5. Fused threshold sensitivity: all four alarm thresholds shifted together
  6. False alarm episodes per hour in the empty scenes

Nothing is tuned here; the testing recordings only replay the frozen model.

    python -m Tools.requirement1_analysis --name Requirement1_Analysis

Outputs (Processed/<name>/): report.txt, req1_operating_curves.png,
req1_per_recording.png, req1_statistics.tex
"""
from pathlib import Path
import argparse
import math
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Fusion.dataset import buildDataset
from Fusion.evaluation import (TARGET_ACC_GAIN, TARGET_PFA_DROP, Recordings, addOutboundLabels,
                               improvement, pirBaselineStates, runStateMachine)
from Fusion.gradientBoosting import GradientBoosting
from Fusion.realtimeModel import INTRUDER, MAYBE_INTRUDER, NO_INTRUDER
from Fusion.qualification import SEEN_SITE, parseRecordingName
from PIR.processing import threshold as PIR_THRESHOLD
from Fusion import report


# Config
testingDataset = Path("Datasets/testing")
modelPath = Path("Processed/realtime/fusion_gb.json")
outputPath = Path("Processed")
BOOTSTRAP_SAMPLES = 2000
BLOCK_SECONDS = (10.0, 20.0, 40.0)      # block lengths tried; the result should not depend on it
PIR_SWEEP_V = np.round(np.arange(0.05, 3.001, 0.05), 3)
THRESHOLD_SHIFTS = (-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.08)
RANDOM_SEED = 0

PIR = "Thresholded PIR"
FUSED = "Fused system"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def loadTesting(model):
    data = addOutboundLabels(buildDataset(testingDataset, windowSeconds=model.window))
    data["score"] = model.predictProba(data[model.features].to_numpy(dtype=float))
    data["pirState"] = pirBaselineStates(data)
    data["fusedState"] = runStateMachine(data, data["score"].to_numpy(), model.thresholds)
    data["name"] = data["recording"].map(lambda r: Path(r).name)
    return data


def subset(data, names):
    return data[data["name"].isin(names)].reset_index(drop=True)


def qt1Names(data):
    names = []
    for name in sorted(data["name"].unique()):
        info = parseRecordingName(name)
        if info["site"] == SEEN_SITE and info["condition"] == "Day" and info["kind"] in ("distance", "edge", "empty"):
            names.append(name)
    return names


def nightNames(data):
    return [n for n in sorted(data["name"].unique())
            if parseRecordingName(n)["site"] == SEEN_SITE and parseRecordingName(n)["condition"] == "Night"
            and parseRecordingName(n)["kind"] in ("distance", "edge", "empty")]


# Pd / Pfa / Accuracy Of Both Systems And The Requirement 1 Improvements
def compare(data, pirStates=None, fusedStates=None):
    recordings = Recordings(data)
    pir = recordings.core(data["pirState"].to_numpy() if pirStates is None else pirStates)
    fused = recordings.core(data["fusedState"].to_numpy() if fusedStates is None else fusedStates)
    return {PIR: pir, FUSED: fused,
            "gain": improvement(fused["Accuracy (%)"], pir["Accuracy (%)"]),
            "reduction": improvement(fused["Pfa (%)"], pir["Pfa (%)"], reduction=True)}


# ---------------------------------------------------------------------------
# 1. Accuracy in two parts
# ---------------------------------------------------------------------------

# Accuracy = p_h * (alerted human intervals) + (1 - p_h) * (quiet non-human intervals)
def accuracyParts(data):
    human = data["human"].to_numpy() == 1
    rows = {}
    for name, column in ((PIR, "pirState"), (FUSED, "fusedState")):
        alert = data[column].to_numpy() >= MAYBE_INTRUDER
        rows[name] = {"human share": human.mean(), "human alerted": alert[human].mean(),
                      "non-human quiet": (~alert[~human]).mean(), "accuracy": (alert == human).mean()}
    return rows


# ---------------------------------------------------------------------------
# 2. Every recording on its own + sign test
# ---------------------------------------------------------------------------

def signTestP(wins, losses):
    """One-sided exact sign test (ties left out): P(at least `wins` of n by chance)."""
    n = wins + losses
    if n == 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def perRecording(data, names):
    rows = []
    for name in names:
        c = compare(subset(data, [name]))
        rows.append({"recording": name,
                     "pirAcc": c[PIR]["Accuracy (%)"], "fusedAcc": c[FUSED]["Accuracy (%)"],
                     "pirPfa": c[PIR]["Pfa (%)"], "fusedPfa": c[FUSED]["Pfa (%)"],
                     "gain": c["gain"], "reduction": c["reduction"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Block bootstrap
# ---------------------------------------------------------------------------

def timeBlocks(data, seconds):
    blocks = []
    for _, group in data.groupby("name", sort=False):
        times = group["fusionTime"].to_numpy(dtype=float)
        number = ((times - times.min()) // seconds).astype(int)
        for k in np.unique(number):
            blocks.append(group.index[number == k].to_numpy())
    return blocks


# Interval-Level Accuracy And Pfa (same definitions as Fusion/evaluation.py)
def intervalMetrics(human, outbound, states):
    alert = states >= MAYBE_INTRUDER
    negative = ~human
    falseAlarm = negative & ((states == INTRUDER) | ((states == MAYBE_INTRUDER) & ~outbound))
    return (alert == human).mean(), falseAlarm.sum() / max(negative.sum(), 1)


def blockBootstrap(data, seconds, rng):
    blocks = timeBlocks(data, seconds)
    human = data["human"].to_numpy() == 1
    outbound = data["outboundHuman"].to_numpy() == 1
    pirStates = data["pirState"].to_numpy()
    fusedStates = data["fusedState"].to_numpy()
    gains, reductions = [], []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        pirAcc, pirPfa = intervalMetrics(human[idx], outbound[idx], pirStates[idx])
        fusedAcc, fusedPfa = intervalMetrics(human[idx], outbound[idx], fusedStates[idx])
        gains.append(100.0 * (fusedAcc - pirAcc) / pirAcc)
        reductions.append(100.0 * (pirPfa - fusedPfa) / pirPfa if pirPfa > 0 else np.nan)
    return len(blocks), np.asarray(gains), np.asarray(reductions)


# ---------------------------------------------------------------------------
# 4. PIR threshold sweep / 5. fused threshold sensitivity
# ---------------------------------------------------------------------------

def pirSweep(data):
    recordings = Recordings(data)
    peak = data["pirThresholdIntervalPeak"].to_numpy(dtype=float)
    rows = []
    for volts in PIR_SWEEP_V:
        m = recordings.core(np.where(peak >= volts, INTRUDER, NO_INTRUDER))
        rows.append({"threshold": float(volts), "pd": m["Event Pd (%)"], "pfa": m["Pfa (%)"],
                     "acc": m["Accuracy (%)"]})
    return pd.DataFrame(rows)


def fusedSweep(data):
    recordings = Recordings(data)
    scores = data["score"].to_numpy()
    rows = []
    for t in np.linspace(0.02, 0.98, 49):
        m = recordings.core(np.where(scores >= t, INTRUDER, NO_INTRUDER))
        rows.append({"threshold": float(t), "pd": m["Event Pd (%)"], "pfa": m["Pfa (%)"]})
    return pd.DataFrame(rows)


def thresholdSensitivity(data, thresholds):
    rows = []
    for shift in THRESHOLD_SHIFTS:
        shifted = {k: (min(0.99, v + shift) if k != "hold_s" else v) for k, v in thresholds.items()}
        states = runStateMachine(data, data["score"].to_numpy(), shifted)
        c = compare(data, fusedStates=states)
        rows.append({"shift": shift, "pd": c[FUSED]["Event Pd (%)"], "pfa": c[FUSED]["Pfa (%)"],
                     "gain": c["gain"], "reduction": c["reduction"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Empty scenes: false alarm episodes per hour
# ---------------------------------------------------------------------------

def episodes(flags):
    flags = np.asarray(flags, dtype=bool)
    return int(np.count_nonzero(flags & ~np.r_[False, flags[:-1]]))


def emptyScenes(data):
    rows = []
    for name in sorted(data["name"].unique()):
        if parseRecordingName(name)["kind"] != "empty":
            continue
        group = subset(data, [name])
        minutes = (group["fusionTime"].max() - group["fusionTime"].min()) / 60.0
        pirEpisodes = episodes(group["pirState"].to_numpy() == INTRUDER)
        fusedEpisodes = episodes(group["fusedState"].to_numpy() == INTRUDER)
        rows.append({"recording": name, "minutes": minutes, "pirEpisodes": pirEpisodes,
                     "fusedEpisodes": fusedEpisodes, "pirPerHour": pirEpisodes / minutes * 60.0,
                     "fusedPerHour": fusedEpisodes / minutes * 60.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figOperatingCurves(pirCurve, fusedCurve, result, out):
    colour = report.SYSTEM_COLOURS[0]
    fig, ax = plt.subplots(figsize=(report.FIG_WIDTH * 0.8, 3.6))
    ax.plot(fusedCurve["pfa"], fusedCurve["pd"], color=colour, linewidth=2, label="Fused system (threshold swept)")
    ax.plot(pirCurve["pfa"], pirCurve["pd"], color=report.PIR_COLOUR, linewidth=2,
            label="Thresholded PIR (threshold swept)")
    ax.plot(result[FUSED]["Pfa (%)"], result[FUSED]["Event Pd (%)"], "o", markersize=8, color=colour,
            markeredgecolor="white", markeredgewidth=2, label="Fused system (frozen thresholds)")
    ax.plot(result[PIR]["Pfa (%)"], result[PIR]["Event Pd (%)"], "s", markersize=8, color=report.PIR_COLOUR,
            markeredgecolor="white", markeredgewidth=2, label=f"Thresholded PIR ({PIR_THRESHOLD:g} V)")
    for volts in (0.5, 1.0):
        row = pirCurve.iloc[(pirCurve["threshold"] - volts).abs().argmin()]
        ax.annotate(f"{volts:g} V", (row["pfa"], row["pd"]), textcoords="offset points", xytext=(6, -4),
                    fontsize=7, color=report.TEXT_2)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 102)
    ax.set_xlabel("False alarm probability (%)")
    ax.set_ylabel("Event probability of detection (%)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "req1_operating_curves.png")
    plt.close(fig)


def figPerRecording(table, out):
    labels = [n.replace(f"{SEEN_SITE}_", "").replace("_", " ") for n in table["recording"]]
    y = np.arange(len(table))[::-1]
    colour = report.SYSTEM_COLOURS[0]
    fig, axes = plt.subplots(1, 2, figsize=(report.FIG_WIDTH, 0.32 * len(table) + 1.0), sharey=True)
    for ax, pirColumn, fusedColumn, xlabel in ((axes[0], "pirAcc", "fusedAcc", "Accuracy (%)"),
                                                (axes[1], "pirPfa", "fusedPfa", "False alarm probability (%)")):
        for yi, a, b in zip(y, table[pirColumn], table[fusedColumn]):
            ax.plot([a, b], [yi, yi], color=report.GRID, linewidth=3, zorder=1)
        ax.scatter(table[pirColumn], y, color=report.PIR_COLOUR, s=30, zorder=2, label=PIR)
        ax.scatter(table[fusedColumn], y, color=colour, s=30, zorder=3, label=FUSED)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].set_xlim(40, 100)
    axes[1].set_xlim(0, 65)
    axes[1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "req1_per_recording.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(runName):
    out = outputPath / runName
    out.mkdir(parents=True, exist_ok=False)
    model = GradientBoosting.load(modelPath)
    data = loadTesting(model)
    qt1 = subset(data, qt1Names(data))
    night = subset(data, nightNames(data))
    rng = np.random.default_rng(RANDOM_SEED)
    lines = [f"REQUIREMENT 1 ANALYSIS: {runName}",
             f"Frozen model {modelPath} ({model.window:g} s window), thresholds {model.thresholds}.",
             f"QT1 recordings (seen site, day): {', '.join(qt1Names(data))}",
             f"Night recordings: {', '.join(nightNames(data))}", ""]

    result = compare(qt1)
    nightResult = compare(night)
    lines.append("HEADLINE (QT1)")
    for name in (PIR, FUSED):
        m = result[name]
        lines.append(f"  {name:16s} Pd {m['Event Pd (%)']:5.1f}%  Pfa {m['Pfa (%)']:5.1f}%  Accuracy {m['Accuracy (%)']:5.1f}%")
    pirErr, fusedErr = 100 - result[PIR]["Accuracy (%)"], 100 - result[FUSED]["Accuracy (%)"]
    lines += [f"  Relative accuracy gain {result['gain']:.1f}% (target {TARGET_ACC_GAIN:g}%), "
              f"relative false alarm reduction {result['reduction']:.1f}% (target {TARGET_PFA_DROP:g}%)",
              f"  Absolute accuracy change {result[FUSED]['Accuracy (%)'] - result[PIR]['Accuracy (%)']:.1f} points; "
              f"error rate {pirErr:.1f}% -> {fusedErr:.1f}% ({100 * (pirErr - fusedErr) / pirErr:.1f}% fewer errors)",
              f"  Night: gain {nightResult['gain']:.1f}%, reduction {nightResult['reduction']:.1f}%", ""]

    lines.append("1. ACCURACY IN TWO PARTS (QT1): accuracy = share human x human alerted + (1 - share human) x non-human quiet")
    for name, part in accuracyParts(qt1).items():
        lines.append(f"  {name:16s} share human {part['human share']:.3f}  human alerted {part['human alerted']:.3f}  "
                     f"non-human quiet {part['non-human quiet']:.3f}  accuracy {part['accuracy']:.3f}")
    lines.append("")

    table = perRecording(data, qt1Names(data) + nightNames(data))
    accWins = int((table["fusedAcc"] > table["pirAcc"]).sum())
    accLosses = int((table["fusedAcc"] < table["pirAcc"]).sum())
    pfaWins = int((table["fusedPfa"] < table["pirPfa"]).sum())
    pfaLosses = int((table["fusedPfa"] > table["pirPfa"]).sum())
    lines.append("2. EVERY RECORDING ON ITS OWN")
    for _, r in table.iterrows():
        lines.append(f"  {r['recording']:24s} accuracy {r['pirAcc']:5.1f} -> {r['fusedAcc']:5.1f} ({r['gain']:+6.1f}%)   "
                     f"Pfa {r['pirPfa']:5.1f} -> {r['fusedPfa']:5.1f} ({r['reduction']:5.1f}% lower)")
    lines += [f"  Accuracy higher in {accWins}, lower in {accLosses}, equal in {len(table) - accWins - accLosses} "
              f"recordings: sign test p = {signTestP(accWins, accLosses):.4f}",
              f"  Pfa lower in {pfaWins}, higher in {pfaLosses}, equal in {len(table) - pfaWins - pfaLosses} "
              f"recordings: sign test p = {signTestP(pfaWins, pfaLosses):.4f}", ""]

    lines.append(f"3. BLOCK BOOTSTRAP (QT1, {BOOTSTRAP_SAMPLES} resamples of whole time blocks)")
    bootstrapRows = []
    for seconds in BLOCK_SECONDS:
        count, gains, reductions = blockBootstrap(qt1, seconds, rng)
        g = np.nanpercentile(gains, [2.5, 97.5])
        r = np.nanpercentile(reductions, [2.5, 97.5])
        pGain, pRed = np.mean(gains >= TARGET_ACC_GAIN), np.nanmean(reductions >= TARGET_PFA_DROP)
        bootstrapRows.append((seconds, count, g, r, pGain, pRed))
        lines.append(f"  {seconds:4.0f} s blocks ({count:3d}): accuracy gain 95% CI {g[0]:.1f}-{g[1]:.1f}% "
                     f"(share >= {TARGET_ACC_GAIN:g}%: {pGain:.3f}); false alarm reduction 95% CI "
                     f"{r[0]:.1f}-{r[1]:.1f}% (share >= {TARGET_PFA_DROP:g}%: {pRed:.3f})")
    lines.append("")

    pirCurve = pirSweep(qt1)
    best = pirCurve.loc[pirCurve["acc"].idxmax()]
    match = pirCurve[pirCurve["pfa"] <= result[FUSED]["Pfa (%)"]].head(1)
    lines.append("4. PIR THRESHOLD SWEEP (QT1): the best any single PIR threshold can do on the same data")
    for volts in (0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0):
        row = pirCurve.iloc[(pirCurve["threshold"] - volts).abs().argmin()]
        lines.append(f"  {row['threshold']:4.2f} V: Pd {row['pd']:5.1f}%  Pfa {row['pfa']:5.1f}%  accuracy {row['acc']:5.1f}%")
    lines.append(f"  Highest PIR accuracy: {best['acc']:.1f}% at {best['threshold']:.2f} V (Pd {best['pd']:.1f}%, "
                 f"Pfa {best['pfa']:.1f}%); fused accuracy is {100 * (result[FUSED]['Accuracy (%)'] - best['acc']) / best['acc']:.1f}% higher.")
    if len(match):
        m = match.iloc[0]
        lines.append(f"  To match the fused Pfa ({result[FUSED]['Pfa (%)']:.1f}%) the PIR needs {m['threshold']:.2f} V, "
                     f"where its Pd falls to {m['pd']:.1f}%.")
    lines.append("")

    sensitivity = thresholdSensitivity(qt1, model.thresholds)
    lines.append("5. FUSED THRESHOLD SENSITIVITY (QT1): all four alarm thresholds shifted together (not used for tuning)")
    for _, r in sensitivity.iterrows():
        lines.append(f"  shift {r['shift']:+.3f}: Pd {r['pd']:5.1f}%  Pfa {r['pfa']:4.1f}%  gain {r['gain']:5.1f}%  "
                     f"reduction {r['reduction']:5.1f}%")
    lines.append("")

    empty = emptyScenes(data)
    lines.append("6. EMPTY SCENES: INTRUDER alarm episodes")
    for _, r in empty.iterrows():
        lines.append(f"  {r['recording']:22s} {r['minutes']:.1f} min: PIR {r['pirEpisodes']} ({r['pirPerHour']:.0f}/h), "
                     f"fused {r['fusedEpisodes']} ({r['fusedPerHour']:.0f}/h)")

    figOperatingCurves(pirCurve, fusedSweep(qt1), result, out)
    figPerRecording(table, out)

    # LaTeX table of the statistical checks
    texRows = [rf"Accuracy higher than PIR & {accWins} of {len(table)} recordings (sign test $p = {signTestP(accWins, accLosses):.3f}$) \\",
               rf"False alarms lower than PIR & {pfaWins} of {len(table)} recordings (sign test $p = {signTestP(pfaWins, pfaLosses):.3f}$) \\"]
    for seconds, count, g, r, _, _ in bootstrapRows:
        texRows.append(rf"95\% CI, {seconds:.0f}~s blocks & accuracy gain {g[0]:.1f}--{g[1]:.1f}\%, "
                       rf"false alarm reduction {r[0]:.1f}--{r[1]:.1f}\% \\")
    texRows.append(rf"Best PIR threshold ({best['threshold']:.2f}~V) & accuracy {best['acc']:.1f}\% "
                   rf"(fused {result[FUSED]['Accuracy (%)']:.1f}\%) \\")
    tex = "\n".join([r"\begin{table}[H]", r"\centering", r"\fontsize{10}{12}\selectfont",
                     r"\begin{tabular}{|l|l|}", r"\hline", r"\textbf{Check} & \textbf{Result} \\", r"\hline",
                     *texRows, r"\hline", r"\end{tabular}",
                     r"\caption{Statistical checks of the Requirement 1 result on the testing recordings.}",
                     r"\label{tab:qt1_statistics}", r"\end{table}"]) + "\n"
    (out / "req1_statistics.tex").write_text(tex, encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Requirement 1 supporting analysis")
    parser.add_argument("--name", default="Requirement1_Analysis", help="Output folder in Processed/")
    main(parser.parse_args().name)
