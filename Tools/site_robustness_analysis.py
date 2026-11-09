"""Requirement 1 across every labelled recording and site (supplementary to QT1).

QT1 verifies Requirement 1 once, with the frozen model on the testing recordings.
This script asks a different question: how does the improvement over the PIR
change with the scene? It uses every labelled recording from every site:

  1. Leave-one-recording-out (LORO), per site: each recording is scored by a
     Gradient Boosting model trained on all OTHER recordings of the same site
     (the system is trained at its installation site). No recording is ever
     scored by a model that saw it.
  2. Alarm thresholds per site: tuned on the site's validation recording only
     (same rule as offlineRun.py --train), then applied to the other recordings.
  3. Threshold-free comparison per site: event Pd vs Pfa curves with the fused
     confidence threshold and the PIR voltage threshold swept, so the result
     does not depend on any chosen threshold or on how bad the PIR baseline is.
  4. Thermal clutter per recording (Tools/contrast_analysis.py): the clutter
     ceiling is the median hottest-pixel residual (C) with nobody in the zone.
     It is compared with how much the fusion improves on the PIR per recording
     (fused AUC minus PIR AUC, interval level).

scikit-learn's GradientBoostingClassifier is used for the many LORO models (offline
comparison only); it has the same settings as Fusion/models.py and gives the same
confidences as the first-principles model (Processed/Train_Driveway_2/report.txt).

    python -m Tools.site_robustness_analysis --name Site_Robustness

Outputs (Processed/<name>/): report.txt, per_recording.csv, site_operating_curves.png,
clutter_vs_benefit.png, site_results.tex, clutter_table.tex
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

from Fusion.dataset import buildDataset, getFusedFeatureNames
from Fusion.evaluation import (TARGET_ACC_GAIN, TARGET_PFA_DROP, Recordings, addOutboundLabels, improvement,
                               pirBaselineStates, runStateMachine, tuneThresholds)
from Fusion.models import buildModels
from Fusion.realtimeModel import INTRUDER, NO_INTRUDER
from Fusion import report
from Tools.contrast_analysis import auc, measureRecording, summarise


# Config: every labelled recording of each site (folders are searched for these names)
SITES = {
    "Driveway": {"folders": [Path("Datasets/training"), Path("Datasets/validation"), Path("Datasets/testing")],
                 "prefix": "Driveway", "validation": "DrivewayValid"},
    "Garden": {"folders": [Path("Datasets/old/training"), Path("Datasets/old/validation"), Path("Datasets/old/testing")],
               "prefix": "Garden", "validation": "GardenValid"},
    "Uni": {"folders": [Path("Datasets/old/training")], "prefix": "Uni", "validation": None},
}
WINDOW_S = 2.0
OUTPUT = Path("Processed")
PERMUTATIONS = 10000
RANDOM_SEED = 0
SITE_COLOURS = {"Driveway": report.SYSTEM_COLOURS[0], "Garden": report.SYSTEM_COLOURS[1],
                "Uni": report.SYSTEM_COLOURS[2]}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def siteRecordings(site):
    config = SITES[site]
    found = []
    for folder in config["folders"]:
        if folder.is_dir():
            found += [p for p in sorted(folder.iterdir())
                      if p.is_dir() and p.name.startswith(config["prefix"]) and (p / "labels.csv").is_file()]
    return found


# Features Of Every Recording Of A Site; Outbound Labels Where They Exist (needed to score Pfa)
def loadSite(site):
    pieces = []
    for path in siteRecordings(site):
        group = buildDataset(path, windowSeconds=WINDOW_S)     # one recording folder
        if (path / "labels(outbound).csv").is_file():
            group = addOutboundLabels(group)
            group["scored"] = True
        else:
            group = group.copy()
            group["outboundHuman"] = 0
            group["scored"] = False          # training-only recording: no outbound labels, never scored
        pieces.append(group)
    data = pd.concat(pieces, ignore_index=True)
    data["site"] = site
    data["name"] = data["recording"].map(lambda r: Path(r).name)
    return data


# ---------------------------------------------------------------------------
# 1. Leave one recording out
# ---------------------------------------------------------------------------

def leaveOneRecordingOut(data, features):
    scores = np.full(len(data), np.nan)
    for name in data.loc[data["scored"], "name"].unique():
        train = data["name"] != name
        model = buildModels()["Gradient Boosting"].fit(data.loc[train, features], data.loc[train, "human"])
        test = ~train
        scores[test.to_numpy()] = model.predict_proba(data.loc[test, features])[:, 1]
        print(f"    {name}", flush=True)
    return scores


# ---------------------------------------------------------------------------
# 2. Requirement 1 per site (thresholds tuned on the site's validation recording)
# ---------------------------------------------------------------------------

def siteRequirement(data, site):
    validation = SITES[site]["validation"]
    scored = data[data["scored"]].reset_index(drop=True)
    if validation is None or validation not in set(scored["name"]):
        return None
    valid = scored[scored["name"] == validation].reset_index(drop=True)
    thresholds = tuneThresholds(valid, valid["score"].to_numpy())["thresholds"]
    rest = scored[scored["name"] != validation].reset_index(drop=True)
    recordings = Recordings(rest)
    pir = recordings.core(rest["pirState"].to_numpy())
    fused = recordings.core(runStateMachine(rest, rest["score"].to_numpy(), thresholds))
    return {"thresholds": thresholds, "pir": pir, "fused": fused, "recordings": rest["name"].nunique(),
            "gain": improvement(fused["Accuracy (%)"], pir["Accuracy (%)"]),
            "reduction": improvement(fused["Pfa (%)"], pir["Pfa (%)"], reduction=True)}


# ---------------------------------------------------------------------------
# 3. Threshold-free curves
# ---------------------------------------------------------------------------

def curves(data):
    data = data[data["scored"]].reset_index(drop=True)
    recordings = Recordings(data)
    peak = data["pirThresholdIntervalPeak"].to_numpy(dtype=float)
    scores = data["score"].to_numpy()
    pir = [recordings.core(np.where(peak >= v, INTRUDER, NO_INTRUDER))
           for v in np.r_[np.arange(0.05, 1.0, 0.025), np.arange(1.0, 3.01, 0.1)]]
    fused = [recordings.core(np.where(scores >= t, INTRUDER, NO_INTRUDER)) for t in np.linspace(0.01, 0.99, 99)]
    table = lambda rows: pd.DataFrame({"pd": [m["Event Pd (%)"] for m in rows], "pfa": [m["Pfa (%)"] for m in rows]})
    pir, fused = table(pir), table(fused)
    baseline = recordings.core(data["pirState"].to_numpy())
    # Fused false alarms at the PIR's own detection rate (0.2 V)
    reachable = fused[fused["pd"] >= baseline["Event Pd (%)"]]
    matchedPfa = float(reachable["pfa"].min()) if len(reachable) else np.nan
    # Share of PIR operating points that some fused operating point beats or equals on both axes
    dominated = np.mean([((fused["pd"] >= r.pd) & (fused["pfa"] <= r.pfa)).any() for r in pir.itertuples()])
    return {"pir": pir, "fused": fused, "baseline": baseline, "matchedPfa": matchedPfa, "dominated": dominated}


# ---------------------------------------------------------------------------
# 4. Clutter vs benefit
# ---------------------------------------------------------------------------

def ranks(values):
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values))
    out[order] = np.arange(1, len(values) + 1)
    # average ties
    for value in np.unique(values):
        tie = values == value
        out[tie] = out[tie].mean()
    return out


# Spearman Rank Correlation With A Two-Sided Permutation p-Value (first principles)
def spearman(x, y, rng):
    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(PERMUTATIONS)])
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (PERMUTATIONS + 1))


def perRecording(data, recordingPaths):
    rows = []
    for name, group in data[data["scored"]].groupby("name", sort=False):
        human = group["human"].to_numpy() == 1
        if human.sum() < 10 or (~human).sum() < 10:
            continue
        measured = measureRecording(recordingPaths[name])
        if measured is None:
            continue
        clutter = summarise(measured)
        pirAuc = auc(group.loc[human, "pirThresholdIntervalPeak"].to_numpy(), group.loc[~human, "pirThresholdIntervalPeak"].to_numpy())
        fusedAuc = auc(group.loc[human, "score"].to_numpy(), group.loc[~human, "score"].to_numpy())
        rows.append({"recording": name, "site": group["site"].iloc[0],
                     "clutterC": clutter["Clutter ceiling (C)"], "personPeakC": clutter["Person peak (C)"],
                     "scr": clutter["SCR"], "thermalAuc": clutter["Separability (AUC)"],
                     "backgroundC": clutter["Background (C)"],
                     "pirAuc": pirAuc, "fusedAuc": fusedAuc, "benefit": fusedAuc - pirAuc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figCurves(results, out):
    sites = list(results)
    fig, axes = plt.subplots(1, len(sites), figsize=(report.FIG_WIDTH, 2.6), sharey=True)
    for ax, site in zip(np.atleast_1d(axes), sites):
        c = results[site]
        ax.plot(c["fused"]["pfa"], c["fused"]["pd"], color=report.SYSTEM_COLOURS[0], linewidth=2, label="Fused system")
        ax.plot(c["pir"]["pfa"], c["pir"]["pd"], color=report.PIR_COLOUR, linewidth=2, label="Thresholded PIR")
        ax.plot(c["baseline"]["Pfa (%)"], c["baseline"]["Event Pd (%)"], "s", color=report.PIR_COLOUR,
                markeredgecolor="white", markersize=7)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 102)
        ax.set_xlabel("False alarm probability (%)", fontsize=8)
        ax.set_title(site, loc="left", fontsize=9, fontweight="normal")
    np.atleast_1d(axes)[0].set_ylabel("Event Pd (%)")
    np.atleast_1d(axes)[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "site_operating_curves.png")
    plt.close(fig)


def figClutter(table, out):
    fig, ax = plt.subplots(figsize=(report.FIG_WIDTH * 0.8, 3.2))
    for site, group in table.groupby("site", sort=False):
        ax.scatter(group["clutterC"], group["benefit"], s=30, color=SITE_COLOURS.get(site), edgecolor="white", label=site)
    ax.axhline(0, color=report.TEXT_2, linestyle=":", linewidth=1)
    ax.set_xlabel("Thermal clutter ceiling with nobody present (C)")
    ax.set_ylabel("Fused AUC minus PIR AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "clutter_vs_benefit.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(runName):
    out = OUTPUT / runName
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(RANDOM_SEED)
    features = getFusedFeatureNames()
    lines = [f"SITE ROBUSTNESS: {runName}",
             "Leave-one-recording-out per site (model trained on the other recordings of the same site).",
             "Supplementary to QT1: shows how the improvement over the PIR depends on the scene.", ""]
    siteData, curveResults, requirement, recordingPaths = {}, {}, {}, {}
    for site in SITES:
        paths = siteRecordings(site)
        if not paths:
            continue
        recordingPaths.update({p.name: p for p in paths})
        print(f"{site}: {len(paths)} recordings")
        data = loadSite(site)
        data["pirState"] = pirBaselineStates(data)
        data["score"] = leaveOneRecordingOut(data, features)
        siteData[site] = data
        curveResults[site] = curves(data)
        requirement[site] = siteRequirement(data, site)

    lines.append("1. REQUIREMENT 1 PER SITE (thresholds tuned on the site's validation recording only)")
    lines.append(f"  {'Site':9s} | Rec | Entries | PIR Pd  Pfa   Acc  | Fused Pd  Pfa   Acc  | Acc gain | Pfa red")
    for site, r in requirement.items():
        if r is None:
            lines.append(f"  {site:9s} | no validation recording: threshold-free comparison only")
            continue
        p, f = r["pir"], r["fused"]
        lines.append(f"  {site:9s} | {r['recordings']:3d} | {p['Events']:7d} | {p['Event Pd (%)']:5.1f} {p['Pfa (%)']:5.1f} "
                     f"{p['Accuracy (%)']:5.1f} | {f['Event Pd (%)']:5.1f} {f['Pfa (%)']:5.1f} {f['Accuracy (%)']:5.1f} | "
                     f"{r['gain']:7.1f}% | {r['reduction']:6.1f}%   thresholds {r['thresholds']}")
    lines.append("")

    lines.append("2. THRESHOLD-FREE COMPARISON PER SITE")
    for site, c in curveResults.items():
        b = c["baseline"]
        lines.append(f"  {site:9s} PIR at 0.2 V: Pd {b['Event Pd (%)']:.1f}%, Pfa {b['Pfa (%)']:.1f}%. "
                     f"Fused Pfa at the same Pd: {c['matchedPfa']:.1f}%. "
                     f"PIR operating points matched or beaten by the fused curve: {100 * c['dominated']:.0f}%")
    lines.append("")

    table = perRecording(pd.concat(siteData.values(), ignore_index=True), recordingPaths)
    table.to_csv(out / "per_recording.csv", index=False)
    rhoBenefit, pBenefit = spearman(table["clutterC"], table["benefit"], rng)
    rhoPir, pPir = spearman(table["clutterC"], table["pirAuc"], rng)
    lines += ["3. THERMAL CLUTTER VS FUSION BENEFIT (per recording, walks with both classes present)",
              f"  {len(table)} recordings. Spearman rho(clutter ceiling, fused AUC - PIR AUC) = {rhoBenefit:.2f} "
              f"(permutation p = {pBenefit:.4f})",
              f"  Spearman rho(clutter ceiling, PIR AUC) = {rhoPir:.2f} (p = {pPir:.3f}): the PIR does not depend on thermal clutter",
              "  Median per site:"]
    for site, group in table.groupby("site", sort=False):
        lines.append(f"    {site:9s} clutter {group['clutterC'].median():.2f} C, background {group['backgroundC'].median():.1f} C, "
                     f"thermal AUC {group['thermalAuc'].median():.2f}, PIR AUC {group['pirAuc'].median():.2f}, "
                     f"fused AUC {group['fusedAuc'].median():.2f}, benefit {group['benefit'].median():+.2f}")
    lines.append("")
    lines.append("  Recording                        | Site     | Clutter C |  SCR | Thermal AUC | PIR AUC | Fused AUC | Benefit")
    for _, r in table.sort_values("clutterC").iterrows():
        lines.append(f"  {r['recording']:32s} | {r['site']:8s} | {r['clutterC']:9.2f} | {r['scr']:4.2f} | {r['thermalAuc']:11.2f} | "
                     f"{r['pirAuc']:7.2f} | {r['fusedAuc']:9.2f} | {r['benefit']:+.2f}")

    figCurves(curveResults, out)
    figClutter(table, out)

    # LaTeX tables
    rows = []
    for site, r in requirement.items():
        c = curveResults[site]
        if r is None:
            rows.append(rf"{site} & -- & -- & -- & -- & {c['baseline']['Pfa (%)']:.1f} / {c['matchedPfa']:.1f} \\")
            continue
        rows.append(rf"{site} & {r['recordings']} & {r['pir']['Events']} & {r['gain']:.1f} & {r['reduction']:.1f} & "
                    rf"{c['baseline']['Pfa (%)']:.1f} / {c['matchedPfa']:.1f} \\")
    tex = "\n".join([r"\begin{table}[H]", r"\centering", r"\fontsize{10}{12}\selectfont",
                     r"\begin{tabular}{|l|c|c|c|c|c|}", r"\hline",
                     r"\textbf{Site} & \textbf{Recordings} & \textbf{Entries} & \textbf{Accuracy gain (\%)} & "
                     r"\textbf{Pfa reduction (\%)} & \textbf{Pfa at PIR's Pd: PIR / fused (\%)} \\", r"\hline",
                     *rows, r"\hline", r"\end{tabular}",
                     r"\caption{Leave-one-recording-out comparison per site. Specification: accuracy gain $\geq "
                     rf"{TARGET_ACC_GAIN:g}$\%, Pfa reduction $\geq {TARGET_PFA_DROP:g}$\%.}}",
                     r"\label{tab:site_results}", r"\end{table}"]) + "\n"
    (out / "site_results.tex").write_text(tex, encoding="utf-8")
    clutterRows = [rf"{r['recording'].replace('_', chr(92) + '_')} & {r['site']} & {r['clutterC']:.2f} & {r['scr']:.2f} & "
                   rf"{r['thermalAuc']:.2f} & {r['pirAuc']:.2f} & {r['fusedAuc']:.2f} \\"
                   for _, r in table.sort_values("clutterC").iterrows()]
    tex = "\n".join([r"\begin{longtable}{|l|l|c|c|c|c|c|}", r"\hline",
                     r"\textbf{Recording} & \textbf{Site} & \textbf{Clutter (C)} & \textbf{SCR} & "
                     r"\textbf{Thermal AUC} & \textbf{PIR AUC} & \textbf{Fused AUC} \\", r"\hline",
                     *clutterRows, r"\hline",
                     r"\caption{Thermal clutter and separability of every labelled recording.}",
                     r"\label{tab:clutter_all}", r"\end{longtable}"]) + "\n"
    (out / "clutter_table.tex").write_text(tex, encoding="utf-8")

    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Requirement 1 across every labelled recording and site")
    parser.add_argument("--name", default="Site_Robustness", help="Output folder in Processed/")
    main(parser.parse_args().name)
