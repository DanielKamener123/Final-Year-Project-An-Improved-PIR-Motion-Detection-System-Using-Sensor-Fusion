"""PIR threshold sweep against the frozen fused system (report Section 4, QT1).

The fused system is NOT changed here: the exported model, its frozen alarm
thresholds and its PIR features (which use the fixed 0.2 V activity threshold)
stay exactly as tested. Only the threshold of the PIR-only baseline is swept,
to show how a less (or more) sensitive PIR compares with the same fused output
on the same QT1 testing recordings.

For every PIR threshold:
  Event Pd, fresh Pd, Pfa and accuracy (Fusion/evaluation.py definitions)
  Ward event counts on the INTRUDER output (missed, fragmented, inserted)
  Fused accuracy gain and false alarm reduction relative to that PIR

    python -m Tools.pir_threshold_sweep --name PIR_Threshold_Sweep

Outputs (Processed/<name>/): report.txt, sweep.csv, pir_sweep_metrics.png,
pir_sweep_ward.png, pir_sweep_requirement1.png, pir_sweep_table.tex
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Fusion.evaluation import TARGET_ACC_GAIN, TARGET_PFA_DROP, Recordings, improvement
from Fusion.gradientBoosting import GradientBoosting
from Fusion.realtimeModel import INTRUDER, NO_INTRUDER
from PIR.processing import threshold as PIR_THRESHOLD
from Fusion import report
from Tools.requirement1_analysis import loadTesting, qt1Names, subset


# Config
modelPath = Path("Processed/realtime/fusion_gb.json")
outputPath = Path("Processed")
SWEEP_V = np.round(np.r_[np.arange(0.05, 0.5, 0.025), np.arange(0.5, 2.001, 0.05)], 3)
TABLE_V = (0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0)

PIR_NAME = "Thresholded PIR"
FUSED_NAME = "Fused system"
PIR_COLOUR = report.PIR_COLOUR
FUSED_COLOUR = report.SYSTEM_COLOURS[0]


# Every Metric For One Set Of States
def allMetrics(recordings, states):
    metrics = recordings.core(states)
    metrics.update(recordings.ward(states))
    return metrics


# Row Of The Sweep Table For One PIR Threshold
def sweepRow(volts, pir, fused):
    return {"threshold": float(volts),
            "pd": pir["Event Pd (%)"], "freshPd": pir["Ward Fresh Pd (%)"],
            "pfa": pir["Pfa (%)"], "acc": pir["Accuracy (%)"],
            "missed": pir["Ward D"], "fragmented": pir["Ward F"] + pir["Ward FM"],
            "inserted": pir["Ward I"], "overfill": pir["Mean Overfill (s)"],
            "underfill": pir["Mean Underfill (s)"],
            "accGain": improvement(fused["Accuracy (%)"], pir["Accuracy (%)"]),
            "pfaReduction": improvement(fused["Pfa (%)"], pir["Pfa (%)"], reduction=True)}


# Sweeps The PIR-Only Threshold; The Fused States Stay Fixed
def runSweep(data, recordings, fused):
    peak = data["pirThresholdIntervalPeak"].to_numpy(dtype=float)
    # Thresholds above the largest PIR peak in the data can never trigger (output limit)
    volts = SWEEP_V[SWEEP_V <= peak.max()]
    rows = [sweepRow(v, allMetrics(recordings, np.where(peak >= v, INTRUDER, NO_INTRUDER)), fused)
            for v in volts]
    return pd.DataFrame(rows)


# Nearest Swept Threshold To A Voltage
def nearest(sweep, volts):
    return sweep.iloc[(sweep["threshold"] - volts).abs().argmin()]


# Marks The Report Baseline (0.2 V) And The Matched-Pfa Threshold On An Axis
def markThresholds(ax, matched):
    ax.axvline(PIR_THRESHOLD, color=report.TEXT_2, linestyle=":", linewidth=1,
               label=f"PIR baseline ({PIR_THRESHOLD:g} V)")
    if matched is not None:
        ax.axvline(matched, color=report.TEXT_2, linestyle="-.", linewidth=1,
                   label=f"PIR Pfa matched to fused ({matched:g} V)")


# One Shared Legend Under The Panels (keeps the plots clear of labels)
def sharedLegend(fig, axes, columns):
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=columns, fontsize=7, bbox_to_anchor=(0.5, 0.0))


# Figure 1: Pd, Fresh Pd, Pfa And Accuracy Of The PIR Against Its Threshold
def figMetrics(sweep, fused, matched, out):
    fig, axes = plt.subplots(1, 3, figsize=(report.FIG_WIDTH * 1.25, 3.0), sharex=True)
    panels = ((axes[0], (("pd", "Event Pd (%)", "-"), ("freshPd", "Ward Fresh Pd (%)", "--")),
               "Probability of detection (%)"),
              (axes[1], (("pfa", "Pfa (%)", "-"),), "False alarm probability (%)"),
              (axes[2], (("acc", "Accuracy (%)", "-"),), "Accuracy (%)"))
    for ax, series, ylabel in panels:
        for column, key, style in series:
            tag = " (fresh)" if column == "freshPd" else ""
            ax.plot(sweep["threshold"], sweep[column], color=PIR_COLOUR, linestyle=style, linewidth=2,
                    label=f"{PIR_NAME}{tag}")
            ax.axhline(fused[key], color=FUSED_COLOUR, linestyle=style, linewidth=2,
                       label=f"{FUSED_NAME}{tag}")
        markThresholds(ax, matched)
        ax.set_xlabel("PIR threshold (V)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 102)
    sharedLegend(fig, axes, 3)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(out / "pir_sweep_metrics.png")
    plt.close(fig)


# Figure 2: Ward Event Counts Of The PIR Against Its Threshold
def figWard(sweep, fused, matched, out):
    fig, axes = plt.subplots(1, 3, figsize=(report.FIG_WIDTH * 1.25, 3.0), sharex=True)
    panels = ((axes[0], "inserted", "Ward I", "Inserted false alarms"),
              (axes[1], "missed", "Ward D", "Missed entries"),
              (axes[2], "fragmented", None, "Fragmented entries"))
    for ax, column, key, ylabel in panels:
        fusedValue = fused["Ward F"] + fused["Ward FM"] if key is None else fused[key]
        ax.plot(sweep["threshold"], sweep[column], color=PIR_COLOUR, linewidth=2, label=PIR_NAME)
        ax.axhline(fusedValue, color=FUSED_COLOUR, linewidth=2, label=FUSED_NAME)
        markThresholds(ax, matched)
        ax.set_xlabel("PIR threshold (V)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    sharedLegend(fig, axes, 4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "pir_sweep_ward.png")
    plt.close(fig)


# Figure 3: Requirement 1 Improvements Measured Against Each PIR Threshold
def figRequirement(sweep, matched, out):
    fig, axes = plt.subplots(1, 2, figsize=(report.FIG_WIDTH, 3.0), sharex=True)
    for ax, column, target, ylabel in ((axes[0], "accGain", TARGET_ACC_GAIN, "Fused accuracy gain (%)"),
                                       (axes[1], "pfaReduction", TARGET_PFA_DROP, "Fused false alarm reduction (%)")):
        ax.plot(sweep["threshold"], sweep[column], color=FUSED_COLOUR, linewidth=2,
                label="Fused system relative to the PIR at that threshold")
        ax.axhline(target, color=report.TEXT_2, linestyle="--", linewidth=1,
                   label="Requirement 1 target (20% gain, 50% reduction)")
        markThresholds(ax, matched)
        ax.set_xlabel("PIR threshold (V)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
    sharedLegend(fig, axes, 2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out / "pir_sweep_requirement1.png")
    plt.close(fig)


# LaTeX Table At A Few Thresholds Plus The Fused System
def texTable(sweep, fused, best, matched):
    rows = []
    marks = {PIR_THRESHOLD: " (baseline)", float(best["threshold"]): " (best accuracy)"}
    if matched is not None:
        marks[matched] = " (Pfa matched)"
    volts = sorted(set(TABLE_V) | set(marks))
    for v in volts:
        r = nearest(sweep, v)
        rows.append(rf"PIR {r['threshold']:.2f}~V{marks.get(v, '')} & {r['pd']:.1f} & {r['freshPd']:.1f} & "
                    rf"{r['pfa']:.1f} & {r['acc']:.1f} & {int(r['missed'])} & {int(r['inserted'])} \\")
    rows.append(rf"\hline" "\n" rf"Fused system & {fused['Event Pd (%)']:.1f} & {fused['Ward Fresh Pd (%)']:.1f} & "
                rf"{fused['Pfa (%)']:.1f} & {fused['Accuracy (%)']:.1f} & {int(fused['Ward D'])} & "
                rf"{int(fused['Ward I'])} \\")
    return "\n".join([r"\begin{table}[H]", r"\centering", r"\fontsize{10}{12}\selectfont",
                      r"\begin{tabular}{|l|c|c|c|c|c|c|}", r"\hline",
                      r"\textbf{System} & \textbf{Pd (\%)} & \textbf{Fresh Pd (\%)} & \textbf{Pfa (\%)} & "
                      r"\textbf{Accuracy (\%)} & \textbf{D} & \textbf{I} \\", r"\hline",
                      *rows, r"\hline", r"\end{tabular}",
                      r"\caption{Thresholded PIR at different thresholds compared with the unchanged fused system "
                      r"on the testing recordings. D: missed entries, I: inserted false alarms.}",
                      r"\label{tab:qt1_pir_sweep}", r"\end{table}"]) + "\n"


def main(runName):
    out = outputPath / runName
    out.mkdir(parents=True, exist_ok=False)
    model = GradientBoosting.load(modelPath)
    data = loadTesting(model)
    names = qt1Names(data)
    qt1 = subset(data, names)
    recordings = Recordings(qt1)
    fused = allMetrics(recordings, qt1["fusedState"].to_numpy())
    sweep = runSweep(qt1, recordings, fused)
    sweep.to_csv(out / "sweep.csv", index=False)

    best = sweep.loc[sweep["acc"].idxmax()]
    under = sweep[sweep["pfa"] <= fused["Pfa (%)"]]
    matched = float(under.iloc[0]["threshold"]) if len(under) else None
    both = sweep[(sweep["accGain"] >= TARGET_ACC_GAIN) & (sweep["pfaReduction"] >= TARGET_PFA_DROP)]

    lines = [f"PIR THRESHOLD SWEEP: {runName}",
             f"Frozen model {modelPath} ({model.window:g} s window), thresholds {model.thresholds}.",
             f"QT1 recordings: {', '.join(names)}",
             f"Only the PIR-only baseline threshold changes; the fused system and its PIR features "
             f"(activity threshold {PIR_THRESHOLD:g} V) are unchanged.",
             f"Largest PIR interval peak in the data: {qt1['pirThresholdIntervalPeak'].max():.3f} V "
             f"(sweep stops there).", "",
             f"Fused system: Pd {fused['Event Pd (%)']:.1f}%  fresh Pd {fused['Ward Fresh Pd (%)']:.1f}%  "
             f"Pfa {fused['Pfa (%)']:.1f}%  accuracy {fused['Accuracy (%)']:.1f}%  "
             f"D {fused['Ward D']}  F {fused['Ward F'] + fused['Ward FM']}  I {fused['Ward I']}", "",
             "Threshold |  Pd % | Fresh % | Pfa % | Acc % |  D |  F |   I | Acc gain % | Pfa red %"]
    for _, r in sweep.iterrows():
        lines.append(f"  {r['threshold']:5.3f} V | {r['pd']:5.1f} | {r['freshPd']:7.1f} | {r['pfa']:5.1f} | "
                     f"{r['acc']:5.1f} | {int(r['missed']):2d} | {int(r['fragmented']):2d} | {int(r['inserted']):3d} | "
                     f"{r['accGain']:10.1f} | {r['pfaReduction']:9.1f}")
    lines += ["",
              f"Best PIR accuracy: {best['acc']:.1f}% at {best['threshold']:.3f} V "
              f"(fused gain over it {best['accGain']:.1f}%, Pfa reduction {best['pfaReduction']:.1f}%)."]
    if matched is not None:
        m = nearest(sweep, matched)
        lines.append(f"Pfa matched to fused ({fused['Pfa (%)']:.1f}%): PIR at {matched:.3f} V has Pd {m['pd']:.1f}%, "
                     f"fresh Pd {m['freshPd']:.1f}%, accuracy {m['acc']:.1f}%, {int(m['missed'])} missed entries.")
    else:
        lines.append("No swept PIR threshold reaches the fused Pfa.")
    if len(both):
        lines.append(f"Both Requirement 1 parts hold against the PIR for thresholds "
                     f"{both['threshold'].min():.3f}-{both['threshold'].max():.3f} V "
                     f"({len(both)} of {len(sweep)} swept values).")
    else:
        lines.append("No PIR threshold where both Requirement 1 parts hold.")

    figMetrics(sweep, fused, matched, out)
    figWard(sweep, fused, matched, out)
    figRequirement(sweep, matched, out)
    (out / "pir_sweep_table.tex").write_text(texTable(sweep, fused, best, matched), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIR threshold sweep against the frozen fused system")
    parser.add_argument("--name", default="PIR_Threshold_Sweep", help="Output folder in Processed/")
    main(parser.parse_args().name)
