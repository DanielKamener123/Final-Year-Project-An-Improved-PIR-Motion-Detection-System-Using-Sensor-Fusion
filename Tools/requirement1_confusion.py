"""Requirement 1 confusion matrices (report Section 4, QT1).

Replays the exported model (Processed/realtime/fusion_gb.json) with its frozen alarm
thresholds over the QT1 testing recordings (seen site, day: distance, edge and empty
recordings), exactly as Tools/requirement1_analysis.py does, and counts every 0.5 s
decision interval of both systems against the ground truth labels.

Rows (ground truth, from labels.csv and labels(outbound).csv):
    Inbound          a person is inside the detection zone
    Outbound only    nobody inbound, a person only in the outbound zone
    No person        nobody inbound or outbound
Columns (system output):
    Thresholded PIR  No alarm / Alarm
    Fused system     No intruder / Maybe / Intruder

The metrics are read from these matrices with the definitions in Fusion/evaluation.py:
    Accuracy        alert (Maybe or Intruder) agrees with the inbound label
    Pfa             non-human intervals that are false alarms: Maybe or Intruder with
                    no person, or Intruder with a person only outbound
    Recall          inbound intervals alerted          (interval detection rate)
    Specificity     non-human intervals left quiet
    Precision       alerted intervals that were inbound
    F1              harmonic mean of precision and recall
    Event Pd        entries reaching Intruder within 10 s (detected / entries)
Accuracy and Pfa are checked against Fusion/evaluation.py so the numbers match report.txt.

Subjects: the person walking is read from an optional fourth part of the recording
name (Driveway_4m_Day_P2). Recordings without it are DEFAULT_SUBJECT. Empty recordings
have no subject and only count in the pooled results.

    python -m Tools.requirement1_confusion --name Requirement1_Confusion

Outputs (Processed/<name>/): report.txt, req1_confusion_matrices.png,
req1_confusion_counts.csv, req1_confusion_metrics.tex, req1_subjects.tex
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

from Fusion.evaluation import Recordings, improvement, MAX_DELAY_S
from Fusion.gradientBoosting import GradientBoosting
from Fusion.realtimeModel import INTRUDER, MAYBE_INTRUDER, NO_INTRUDER
from Fusion.qualification import parseRecordingName
from Fusion import report
from Tools.requirement1_analysis import PIR, FUSED, loadTesting, modelPath, outputPath, qt1Names, subset

DEFAULT_SUBJECT = "P1"           # person in recordings without a fourth name part
ROWS = ("Inbound", "Outbound only", "No person")
PIR_COLUMNS = ("No alarm", "Alarm")
FUSED_COLUMNS = ("No intruder", "Maybe", "Intruder")


def subjectOf(name):
    if parseRecordingName(name)["kind"] == "empty":
        return None
    parts = name.split("_")
    return parts[3] if len(parts) > 3 and parts[3] else DEFAULT_SUBJECT


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def truthRows(data):
    human = data["human"].to_numpy() == 1
    outbound = data["outboundHuman"].to_numpy() == 1
    return np.where(human, 0, np.where(outbound, 1, 2))


def confusion(data, column):
    rows = truthRows(data)
    states = data[column].to_numpy()
    if column == "pirState":
        cols = np.where(states == INTRUDER, 1, 0)
        matrix = np.zeros((3, 2), dtype=int)
    else:
        cols = np.select([states == NO_INTRUDER, states == MAYBE_INTRUDER, states == INTRUDER], [0, 1, 2], -1)
        if (cols < 0).any():
            raise ValueError("Unknown fused state in evaluation data")
        matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (rows, cols), 1)
    return matrix


# Metrics From A Matrix (same definitions as Fusion/evaluation.py)
def metricsFromMatrix(matrix, fused):
    inbound, outbound, nobody = matrix
    if fused:
        inboundAlert = inbound[1] + inbound[2]
        nonHumanAlert = outbound[1] + outbound[2] + nobody[1] + nobody[2]
        falseAlarms = outbound[2] + nobody[1] + nobody[2]
    else:
        inboundAlert = inbound[1]
        nonHumanAlert = outbound[1] + nobody[1]
        falseAlarms = outbound[1] + nobody[1]
    nInbound, nNonHuman = inbound.sum(), outbound.sum() + nobody.sum()
    total = nInbound + nNonHuman
    recall = inboundAlert / nInbound if nInbound else np.nan
    specificity = (nNonHuman - nonHumanAlert) / nNonHuman if nNonHuman else np.nan
    precision = inboundAlert / (inboundAlert + nonHumanAlert) if inboundAlert + nonHumanAlert else np.nan
    return {
        "Accuracy (%)": 100.0 * (inboundAlert + nNonHuman - nonHumanAlert) / total,
        "Pfa (%)": 100.0 * falseAlarms / nNonHuman if nNonHuman else np.nan,
        "Recall (%)": 100.0 * recall,
        "Specificity (%)": 100.0 * specificity,
        "Precision (%)": 100.0 * precision,
        "F1 (%)": 100.0 * 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
        "counts": {"inboundAlert": int(inboundAlert), "inbound": int(nInbound), "falseAlarms": int(falseAlarms),
                   "nonHuman": int(nNonHuman), "correct": int(inboundAlert + nNonHuman - nonHumanAlert),
                   "total": int(total)},
    }


def evaluate(data):
    recordings = Recordings(data)
    out = {}
    for name, column, fused in ((PIR, "pirState", False), (FUSED, "fusedState", True)):
        matrix = confusion(data, column)
        m = metricsFromMatrix(matrix, fused)
        core = recordings.core(data[column].to_numpy())
        for key in ("Accuracy (%)", "Pfa (%)"):
            if not np.isclose(m[key], core[key]):
                raise AssertionError(f"{name} {key}: matrix {m[key]:.3f} vs evaluation {core[key]:.3f}")
        m["matrix"] = matrix
        m["Event Pd (%)"] = core["Event Pd (%)"]
        m["events"] = (core["Detected Events"], core["Events"])
        m["episodes"] = core["False Alarm Episodes"]
        out[name] = m
    out["gain"] = improvement(out[FUSED]["Accuracy (%)"], out[PIR]["Accuracy (%)"])
    out["reduction"] = improvement(out[FUSED]["Pfa (%)"], out[PIR]["Pfa (%)"], reduction=True)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def drawMatrix(ax, matrix, columns, cmapName):
    rowTotals = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, rowTotals, out=np.zeros(matrix.shape), where=rowTotals > 0) * 100.0
    ax.imshow(share, cmap=cmapName, vmin=0, vmax=100, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]}\n({share[i, j]:.1f}%)", ha="center", va="center", fontsize=8.5,
                    color="white" if share[i, j] > 60 else report.TEXT)
    ax.set_xticks(range(len(columns)), columns, fontsize=8.5)
    ax.set_yticks(range(len(ROWS)), [f"{r}\n(n = {t})" for r, t in zip(ROWS, rowTotals.ravel())], fontsize=8.5)
    ax.set_xlabel("System output", fontsize=9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figConfusion(result, out):
    fig, axes = plt.subplots(1, 2, figsize=(report.FIG_WIDTH, 2.9), gridspec_kw={"width_ratios": [2, 3]})
    drawMatrix(axes[0], result[PIR]["matrix"], PIR_COLUMNS, "Greys")
    drawMatrix(axes[1], result[FUSED]["matrix"], FUSED_COLUMNS, "Blues")
    axes[0].set_ylabel("Ground truth", fontsize=9)
    axes[1].set_yticklabels([])
    axes[0].set_title("(a) Thresholded PIR", fontsize=9, fontweight="normal")
    axes[1].set_title("(b) Fused system", fontsize=9, fontweight="normal")
    fig.tight_layout()
    fig.savefig(out / "req1_confusion_matrices.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def metricsTex(result):
    def cell(m, key, num, den):
        return rf"{m[key]:.1f}\% ({m['counts'][num]}/{m['counts'][den]})" if num else rf"{m[key]:.1f}\%"
    p, f = result[PIR], result[FUSED]
    rows = [
        ("Accuracy", cell(p, "Accuracy (%)", "correct", "total"), cell(f, "Accuracy (%)", "correct", "total")),
        ("False alarm probability", cell(p, "Pfa (%)", "falseAlarms", "nonHuman"),
         cell(f, "Pfa (%)", "falseAlarms", "nonHuman")),
        ("Recall (inbound intervals alerted)", cell(p, "Recall (%)", "inboundAlert", "inbound"),
         cell(f, "Recall (%)", "inboundAlert", "inbound")),
        ("Specificity", cell(p, "Specificity (%)", None, None), cell(f, "Specificity (%)", None, None)),
        ("Precision", cell(p, "Precision (%)", None, None), cell(f, "Precision (%)", None, None)),
        ("F1 score", cell(p, "F1 (%)", None, None), cell(f, "F1 (%)", None, None)),
        ("Event probability of detection", rf"{p['Event Pd (%)']:.1f}\% ({p['events'][0]}/{p['events'][1]})",
         rf"{f['Event Pd (%)']:.1f}\% ({f['events'][0]}/{f['events'][1]})"),
    ]
    body = [rf"{a} & {b} & {c} \\" for a, b, c in rows]
    return "\n".join([r"\begin{table}[H]", r"\centering", r"\fontsize{10}{12}\selectfont",
                      r"\begin{tabular}{|l|c|c|}", r"\hline",
                      r"\textbf{Metric} & \textbf{Thresholded PIR} & \textbf{Fused system} \\", r"\hline",
                      *body, r"\hline", r"\end{tabular}",
                      r"\caption{Metrics read from the confusion matrices in Figure~\ref{fig:req1_confusion}. "
                      r"Counts are 0.5~s decision intervals, except the event probability of detection, "
                      rf"which counts entries alarmed within {MAX_DELAY_S:g}~s.}}",
                      r"\label{tab:req1_confusion_metrics}", r"\end{table}"]) + "\n"


def subjectTable(data, names):
    rows = []
    for subject in sorted({s for s in map(subjectOf, names) if s}):
        r = evaluate(subset(data, [n for n in names if subjectOf(n) == subject]))
        rows.append({"subject": subject, "pirAcc": r[PIR]["Accuracy (%)"], "fusedAcc": r[FUSED]["Accuracy (%)"],
                     "pirPfa": r[PIR]["Pfa (%)"], "fusedPfa": r[FUSED]["Pfa (%)"],
                     "pirPd": r[PIR]["events"], "fusedPd": r[FUSED]["events"],
                     "gain": r["gain"], "reduction": r["reduction"]})
    return pd.DataFrame(rows)


def subjectsTex(table):
    body = [rf"{r.subject} & {r.pirAcc:.1f} & {r.fusedAcc:.1f} & {r.gain:.1f} & {r.pirPfa:.1f} & {r.fusedPfa:.1f} & "
            rf"{r.reduction:.1f} & {r.fusedPd[0]}/{r.fusedPd[1]} \\" for r in table.itertuples()]
    if len(table) > 1:
        body += [r"\hline", rf"Mean $\pm$ SD & {table.pirAcc.mean():.1f} $\pm$ {table.pirAcc.std():.1f} & "
                 rf"{table.fusedAcc.mean():.1f} $\pm$ {table.fusedAcc.std():.1f} & "
                 rf"{table.gain.mean():.1f} $\pm$ {table.gain.std():.1f} & "
                 rf"{table.pirPfa.mean():.1f} $\pm$ {table.pirPfa.std():.1f} & "
                 rf"{table.fusedPfa.mean():.1f} $\pm$ {table.fusedPfa.std():.1f} & "
                 rf"{table.reduction.mean():.1f} $\pm$ {table.reduction.std():.1f} & -- \\"]
    return "\n".join([r"\begin{table}[H]", r"\centering", r"\fontsize{9}{11}\selectfont",
                      r"\begin{tabular}{|l|c|c|c|c|c|c|c|}", r"\hline",
                      r"\textbf{Person} & \multicolumn{3}{c|}{\textbf{Accuracy (\%)}} & "
                      r"\multicolumn{3}{c|}{\textbf{False alarm probability (\%)}} & \textbf{Fused} \\",
                      r" & PIR & Fused & Gain & PIR & Fused & Reduction & \textbf{event $P_d$} \\", r"\hline",
                      *body, r"\hline", r"\end{tabular}",
                      r"\caption{Requirement 1 results for each person who walked the testing protocol. "
                      r"Gain and reduction are relative to the thresholded PIR.}",
                      r"\label{tab:req1_subjects}", r"\end{table}"]) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(runName):
    out = outputPath / runName
    out.mkdir(parents=True, exist_ok=False)
    model = GradientBoosting.load(modelPath)
    data = loadTesting(model)
    names = qt1Names(data)
    qt1 = subset(data, names)
    result = evaluate(qt1)

    lines = [f"REQUIREMENT 1 CONFUSION MATRICES: {runName}",
             f"Frozen model {modelPath} ({model.window:g} s window), thresholds {model.thresholds}.",
             f"QT1 recordings: {', '.join(names)}", ""]
    counts = []
    for name, columns in ((PIR, PIR_COLUMNS), (FUSED, FUSED_COLUMNS)):
        m = result[name]
        lines.append(f"{name} (rows: ground truth, columns: output)")
        lines.append("  " + " " * 16 + "".join(f"{c:>14s}" for c in columns))
        for row, values in zip(ROWS, m["matrix"]):
            lines.append(f"  {row:16s}" + "".join(f"{v:14d}" for v in values))
            counts += [{"system": name, "truth": row, "output": c, "intervals": int(v)} for c, v in zip(columns, values)]
        c = m["counts"]
        lines += [f"  Accuracy {m['Accuracy (%)']:.1f}% ({c['correct']}/{c['total']})   "
                  f"Pfa {m['Pfa (%)']:.1f}% ({c['falseAlarms']}/{c['nonHuman']})   "
                  f"Recall {m['Recall (%)']:.1f}% ({c['inboundAlert']}/{c['inbound']})",
                  f"  Specificity {m['Specificity (%)']:.1f}%   Precision {m['Precision (%)']:.1f}%   "
                  f"F1 {m['F1 (%)']:.1f}%   Event Pd {m['Event Pd (%)']:.1f}% ({m['events'][0]}/{m['events'][1]})   "
                  f"false alarm episodes {m['episodes']}", ""]
    lines += [f"Relative accuracy gain {result['gain']:.1f}% (target 20%), "
              f"relative false alarm reduction {result['reduction']:.1f}% (target 50%)", ""]

    table = subjectTable(data, names)
    lines.append("PER PERSON")
    for r in table.itertuples():
        lines.append(f"  {r.subject:6s} accuracy {r.pirAcc:5.1f} -> {r.fusedAcc:5.1f} ({r.gain:+5.1f}%)   "
                     f"Pfa {r.pirPfa:5.1f} -> {r.fusedPfa:5.1f} ({r.reduction:5.1f}% lower)   "
                     f"fused event Pd {r.fusedPd[0]}/{r.fusedPd[1]}")
    if len(table) > 1:
        lines.append(f"  Mean +/- SD: gain {table.gain.mean():.1f} +/- {table.gain.std():.1f}%, "
                     f"reduction {table.reduction.mean():.1f} +/- {table.reduction.std():.1f}%")

    figConfusion(result, out)
    pd.DataFrame(counts).to_csv(out / "req1_confusion_counts.csv", index=False)
    (out / "req1_confusion_metrics.tex").write_text(metricsTex(result), encoding="utf-8")
    (out / "req1_subjects.tex").write_text(subjectsTex(table), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Requirement 1 confusion matrices")
    parser.add_argument("--name", default="Requirement1_Confusion", help="Output folder in Processed/")
    main(parser.parse_args().name)
