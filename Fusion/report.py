from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from Fusion.evaluation import (
    MAX_DELAY_S, TARGET_ACC_GAIN, TARGET_PFA_DROP, WARD_KEYS, REQUIREMENT_CHECKS,
    Recordings, compareToPir, evaluateStates, meetsResponseTime,
)

PIR_COLOUR = "#8a8984"

SYSTEM_COLOURS = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#0072B2", "#000000"]
INBOUND = "#2a78d6"
OUTBOUND = "#eda100"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# Set common fonts, axes and grid formatting
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
})

FIG_WIDTH = 6.3
SHOW_TITLES = False

# Add chart titles only when enabled
def _title(ax, text, **kwargs):
    if SHOW_TITLES:
        ax.set_title(text, **kwargs)

# Add figure titles only when enabled
def _suptitle(fig, text, **kwargs):
    if SHOW_TITLES:
        fig.suptitle(text, **kwargs)

WARD_NAMES = {"C": "Correct", "D": "Missed", "F": "Fragmented", "M": "Merged",
              "FM": "Frag.+merged", "I": "Inserted"}

# Describe the performance measures used in the reports
DEFINITIONS = [
    f"Pd: INTRUDER at least once within {MAX_DELAY_S:g} s of a labelled entry (event based).",
    "Pfa: MAYBE/INTRUDER with nobody inbound or outbound, or INTRUDER with a person only outbound;",
    "     percentage of non-human fusion intervals (not alarms per hour).",
    "Accuracy: alert (MAYBE or INTRUDER) vs inbound label per fusion interval.",
    "Improvements are relative to the thresholded PIR: Pd improvement = (fused - PIR) / PIR,",
    "     Pfa reduction = (PIR - fused) / PIR, accuracy gain = (fused - PIR) / PIR.",
    f"Requirement 1: accuracy gain >= {TARGET_ACC_GAIN:g}% and false alarm reduction >= {TARGET_PFA_DROP:g}% over PIR.",
    f"Requirement 4: every entry alarmed (INTRUDER) within {MAX_DELAY_S:g} s.",
    "Requirements 2 and 3 (range 1-10 m, speed 0.2-5 m/s): see the per-recording table on the testing set.",
]

# Give each system a consistent plot colour
def _colours(names):
    colours = {}
    k = 0
    for name in names:
        if name == "Thresholded PIR":
            colours[name] = PIR_COLOUR
        else:
            colours[name] = SYSTEM_COLOURS[k % len(SYSTEM_COLOURS)]
            k += 1
    return colours

# Format report numbers and missing values
def _fmt(value, width, decimals=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A".rjust(width)
    return f"{float(value):.{decimals}f}".rjust(width)

# Format each requirement check as PASS or FAIL
def _pass(ok):
    return "PASS" if ok else "FAIL"

# Build the classifier and window comparison table
def comparisonTable(results, best, bestBoosting):
    lines = ["\nVALIDATION RESULTS - every system at every fusion window "
             "(alarm thresholds tuned per classifier on validation)",
             f"{'Win s':>5} | {'System':20s} | {'Pd %':>6} | {'Pfa %':>6} | {'Acc %':>6} | "
             f"{'Acc gain %':>10} | {'Pfa red %':>9} | {'<=10 s':>7} | {'Max s':>5} | {'FA ep.':>6} | Met",
             "-" * 112]
    checks = len(REQUIREMENT_CHECKS)

    # Format each classifier and window result
    for _, row in results.iterrows():
        pir = row["System"] == "Thresholded PIR"
        met = "-" if pir else f"{int(row['Requirements Met'])}/{checks}"
        entries = f"{int(row['Detected Events'])}/{int(row['Events'])}"
        lines.append(
            f"{row['Window (s)']:5.1f} | {row['System']:20s} | {_fmt(row['Event Pd (%)'], 6)} | "
            f"{_fmt(row['Pfa (%)'], 6)} | {_fmt(row['Accuracy (%)'], 6)} | "
            f"{'-' if pir else _fmt(row['Accuracy Gain (%)'], 10):>10} | "
            f"{'-' if pir else _fmt(row['Pfa Reduction (%)'], 9):>9} | "
            f"{entries:>7} | {_fmt(row['Max Delay (s)'], 5, 1)} | "
            f"{int(row['False Alarm Episodes']):6d} | {met}")

    # Format one selected model result for the comparison report
    def summary(row):
        return (f"{int(row['Requirements Met'])}/{checks} checks met, accuracy gain {row['Accuracy Gain (%)']:.1f}%, "
                f"false alarm reduction {row['Pfa Reduction (%)']:.1f}%, "
                f"{int(row['Detected Events'])}/{int(row['Events'])} entries within {MAX_DELAY_S:g} s")
    lines += ["", f"Checks: Requirement 1 accuracy gain, Requirement 1 false alarm reduction, Requirement 4 response time ({checks} in total).",
              f"BEST CONFIGURATION: {best['System']} at {best['Window (s)']:g} s ({summary(best)})",
              f"BEST GRADIENT BOOSTING WINDOW: {bestBoosting['Window (s)']:g} s ({summary(bestBoosting)})",
              "Ranking: most checks met, then worst-case delay within 8 s, then highest false alarm reduction, "
              "then highest accuracy gain.",
              "Train the Gradient Boosting model from first principles at that window with:",
              f"    python offlineRun.py --train --window {bestBoosting['Window (s)']:g}"]
    return lines

# Plot one performance metric across fusion windows
def figComparison(results, metric, title, ylabel, target, targetKind, filename, out):
    systems = list(dict.fromkeys(results["System"]))
    colours = _colours(systems)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.4))

    # Draw each system using the same window values
    for system in systems:
        sub = results[results["System"] == system].sort_values("Window (s)")
        if metric not in sub or sub[metric].isna().all():
            continue
        pir = system == "Thresholded PIR"
        ax.plot(sub["Window (s)"], sub[metric], marker="o", markersize=5,
                linewidth=2, color=colours[system], linestyle="--" if pir else "-", label=system)
    if target is not None:
        ax.axhline(target, color=TEXT, linestyle=":", linewidth=1.2)
        ax.text(ax.get_xlim()[0], target, f" requirement {'≥' if targetKind == 'min' else '≤'} {target:g}%",
                va="bottom", ha="left", color=TEXT_2, fontsize=8)
    ax.set_xlabel("Fusion window (s)")
    ax.set_ylabel(ylabel)
    _title(ax, title, loc="left", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)

# Save model comparison results and figures
def writeComparisonReport(runName, results, best, bestBoosting, outDir, trainingPath, validationPath):
    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=False)
    lines = [f"MODEL AND WINDOW SELECTION: {runName}",
             f"Trained on {trainingPath}, evaluated on {validationPath}.",
             "Validation is used only to choose the classifier, window and alarm thresholds;",
             "the testing recordings give the verification result (offlineRun.py --test).", *DEFINITIONS]

    # Add the ranked model comparison to the report
    lines += comparisonTable(results, best, bestBoosting)
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    results.drop(columns=["Thresholds"], errors="ignore").to_csv(out / "comparison.csv", index=False)

    # Create the event detection plot
    figComparison(results, "Event Pd (%)", "Event probability of detection vs fusion window",
                  "Event Pd (%)", None, None, "compare_event_pd.png", out)

    # Create the false alarm plot
    figComparison(results, "Pfa (%)", "False alarm probability vs fusion window",
                  "Pfa (%)", None, None, "compare_pfa.png", out)
    figComparison(results, "Accuracy Gain (%)", "Accuracy gain over thresholded PIR vs fusion window",
                  "Accuracy gain (%)", TARGET_ACC_GAIN, "min", "compare_accuracy_gain.png", out)
    figComparison(results, "Pfa Reduction (%)", "False alarm reduction over thresholded PIR vs fusion window",
                  "Pfa reduction (%)", TARGET_PFA_DROP, "min", "compare_pfa_reduction.png", out)
    return text

# Compare the systems against the project requirements
def requirementTable(metrics):
    pir = metrics["Thresholded PIR"]
    lines = ["\nREQUIREMENTS (project proposal)",
             f"{'System':26s} | {'Pd %':>6} | {'Pfa %':>6} | {'Acc %':>6} | {'Acc gain %':>10} | "
             f"{'Pfa red %':>9} | {'<=10 s':>7} | {'Mean s':>6} | {'Max s':>5} | "
             f"Req 1 acc   Req 1 FA   Req 4",
             "-" * 136]

    # Format each system performance result
    for name, m in metrics.items():
        entries = f"{int(m['Detected Events'])}/{int(m['Events'])}"
        common = (f"{name:26s} | {_fmt(m['Event Pd (%)'], 6)} | {_fmt(m['Pfa (%)'], 6)} | "
                  f"{_fmt(m['Accuracy (%)'], 6)} | ")
        delays = f"{entries:>7} | {_fmt(m['Mean Delay (s)'], 6)} | {_fmt(m['Max Delay (s)'], 5, 1)} | "
        if name == "Thresholded PIR":
            lines.append(common + f"{'-':>10} | {'-':>9} | " + delays +
                         f"{'-':>10}  {'-':>9}  {_pass(meetsResponseTime(m)):>6}")
            continue
        c = compareToPir(m, pir)
        flags = f"{_pass(c['Meets Accuracy Gain']):>10}  {_pass(c['Meets Pfa Reduction']):>9}  " \
                f"{_pass(c['Meets Response Time']):>6}"
        lines.append(common + f"{_fmt(c['Accuracy Gain (%)'], 10)} | {_fmt(c['Pfa Reduction (%)'], 9)} | "
                     + delays + flags)
    lines.append(f"Entries evaluated: {int(pir['Events'])}. <=10 s = entries alarmed (INTRUDER) within 10 s.")
    return lines

# List the selected alarm thresholds for each system
def thresholdTable(systems):
    lines = ["\nALARM THRESHOLDS",
             f"{'System':26s} | maybe enter/exit | intruder enter/exit | off-delay s"]
    for name, system in systems.items():
        th = system.get("thresholds")
        if th:
            lines.append(f"{name:26s} |    {th['maybe_enter']:.3f}/{th['maybe_exit']:.3f} |"
                         f"         {th['intruder_enter']:.3f}/{th['intruder_exit']:.3f} | {th.get('hold_s', 0.0):.1f}")
    return lines

# Summarise correct, missed and fragmented alarm events
def wardTable(metrics):
    lines = ["\nWARD-STYLE EVENT QUALITY (INTRUDER output)",
             "C=correct 1:1, D=missed, F=fragmented, M=merged, FM=both, I=inserted false alarm.",
             f"Fresh Pd: new OFF->ON alarm within {MAX_DELAY_S:g} s of entry (events already in alarm excluded).",
             "Overfill = INTRUDER time outside a matched entry; underfill = time inside a detected entry "
             "without INTRUDER (mean per detected entry).",
             f"{'System':26s} | " + " | ".join(f"{k:>4}" for k in WARD_KEYS) + f" | {'Fresh Pd':>8} | {'Delay s':>7}"
             + f" | {'Overfill s':>10} | {'Underfill s':>11}",
             "-" * 120]

    # Format each system performance result
    for name, m in metrics.items():
        lines.append(f"{name:26s} | " + " | ".join(f"{int(m[f'Ward {k}']):4d}" for k in WARD_KEYS)
                     + f" | {_fmt(m['Ward Fresh Pd (%)'], 8)} | {_fmt(m['Ward Fresh Delay (s)'], 7)}"
                     + f" | {_fmt(m['Mean Overfill (s)'], 10)} | {_fmt(m['Mean Underfill (s)'], 11)}")
    return lines

# Break down detections and false alarms by recording
def perRecordingTable(evalData, systems):
    recordings = Recordings(evalData)
    lines = ["\nPER RECORDING",
             f"Detected = INTRUDER within {MAX_DELAY_S:g} s of entry; FA = false-alarm intervals / non-human intervals"]
    header = f"{'Recording':22s} | {'Events':>6}"
    widths = {name: len(name) + 4 for name in systems}
    for name in systems:
        header += f" | {name + ' det':>{widths[name]}} | {name + ' FA':>{widths[name]}}"
    lines += [header, "-" * len(header)]

    # Calculate results for every recording separately
    for item in recordings.items:
        negative = item["human"] == 0
        line = f"{Path(item['recording']).name:22s} | {len(item['entries']):6d}"
        for name, system in systems.items():
            st = np.asarray(system["states"])[item["idx"]]
            false = negative & ((st == 2) | ((st == 1) & (item["outbound"] == 0)))
            detected = 0
            for s, e in item["entries"]:
                deadline = min(item["times"][s] + MAX_DELAY_S, item["times"][e])
                if np.any((st[s:e + 1] == 2) & (item["times"][s:e + 1] <= deadline)):
                    detected += 1
            falseAlarms = f"{int(false.sum())}/{int(negative.sum())}"
            line += f" | {detected:{widths[name]}d} | {falseAlarms:>{widths[name]}}"
        lines.append(line)
    return lines

EDGE_WINDOW_S = 3.0   

# Separate false alarms near people from isolated alarms
def falseAlarmBreakdown(evalData, states):
    """Split false-alarm intervals into 'edge' (within EDGE_WINDOW_S of any
    labelled person, inbound or outbound) and 'isolated'. Returns
    {recording: (edge, isolated)} plus an 'ALL' total."""
    out = {}
    total_edge = total_iso = 0
    states = np.asarray(states)

    # Plot the score and alarm history of each recording
    for item in Recordings(evalData).items:
        t = item["times"]
        inbound = item["human"] == 1
        outbound = item["outbound"] == 1
        st = states[item["idx"]]
        false = ~inbound & ((st == 2) | ((st == 1) & ~outbound))

        # Locate labelled person activity near each alarm
        person_times = np.sort(t[inbound | outbound])
        edge = iso = 0
        for k in np.flatnonzero(false):
            if person_times.size:
                pos = np.searchsorted(person_times, t[k])
                near = min(abs(t[k] - person_times[max(pos - 1, 0)]),
                           abs(t[k] - person_times[min(pos, person_times.size - 1)]))
            else:
                near = np.inf

            # Count alarms near the beginning or end of a person event
            if near <= EDGE_WINDOW_S:
                edge += 1
            else:
                iso += 1
        out[Path(item["recording"]).name] = (edge, iso)
        total_edge += edge
        total_iso += iso
    out["ALL"] = (total_edge, total_iso)
    return out

# Build a report of false alarm causes
def falseAlarmTable(evalData, systems):
    breakdowns = {name: falseAlarmBreakdown(evalData, s["states"]) for name, s in systems.items()}
    lines = ["\nFALSE-ALARM BREAKDOWN",
             f"edge = within {EDGE_WINDOW_S:g} s of a labelled person (inbound or outbound): alarm lingering"
             " or starting early at an event boundary; isolated = everything else.",
             f"{'Recording':22s} | " + " | ".join(f"{n[:30]:>30}" for n in systems),
             "-" * (25 + 33 * len(systems))]
    recordings = [r for r in next(iter(breakdowns.values())) if r != "ALL"] + ["ALL"]
    for rec in recordings:
        cells = []
        for name in systems:
            edge, iso = breakdowns[name][rec]
            total = edge + iso
            share = f"{100 * edge / total:.0f}% edge" if total else "-"
            cells.append(f"{edge:4d} edge {iso:4d} iso ({share})".rjust(30))
        lines.append(f"{rec:22s} | " + " | ".join(cells))
    return lines, breakdowns

# Plot detection and false alarm probabilities
def figRequirements(metrics, colours, tag, out, filename="requirements.png"):
    names = list(metrics)
    pdVals = [metrics[n]["Event Pd (%)"] for n in names]
    pfaVals = [metrics[n]["Pfa (%)"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.0))
    for ax, vals, title, ylabel in [
        (axes[0], pdVals, f"Event probability of detection (alarm within {MAX_DELAY_S:g} s)",
         "Event probability of detection (%)"),
        (axes[1], pfaVals, "False alarm probability", "False alarm probability (%)"),
    ]:
        x = np.arange(len(names))
        vals = [0.0 if not np.isfinite(v) else v for v in vals]
        ax.bar(x, vals, width=0.6, color=[colours[n] for n in names], edgecolor="white", linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9)
        ax.set_xticks(x, names, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        _title(ax, title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 105)
    finitePfa = [v for v in pfaVals if np.isfinite(v)]
    axes[1].set_ylim(0, (max(finitePfa) if finitePfa else 0) * 1.25 + 1)
    _suptitle(fig, f"Fused system vs thresholded PIR ({tag})", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / filename)
    plt.close(fig)

# Plot accuracy gain and false alarm reduction against targets
def figImprovements(metrics, colours, tag, out, filename="improvements.png"):
    pir = metrics["Thresholded PIR"]
    fused = [n for n in metrics if n != "Thresholded PIR"]
    labels = ["Accuracy gain", "False alarm reduction"]
    keys = ["Accuracy Gain (%)", "Pfa Reduction (%)"]
    targets = [TARGET_ACC_GAIN, TARGET_PFA_DROP]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.9, 3.0))
    width = 0.8 / len(fused)

    # Draw each fused model next to the PIR baseline
    for k, name in enumerate(fused):
        c = compareToPir(metrics[name], pir)
        vals = [c[key] if np.isfinite(c[key]) else 0.0 for key in keys]
        x = np.arange(len(labels)) + (k - (len(fused) - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.92, color=colours[name], label=name)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top", color=TEXT, fontsize=8)
    for i, t in enumerate(targets):
        if t is not None:
            ax.plot([i - 0.45, i + 0.45], [t, t], color=TEXT, linestyle="--", linewidth=1.2)
            ax.text(i - 0.45, t, f"target {t:g}%", va="top", ha="left", color=TEXT_2, fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Relative to thresholded PIR (%)")
    _title(ax, f"Requirement 1: improvement over thresholded PIR ({tag})", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / filename)
    plt.close(fig)

# Plot the Ward event categories for each system
def figWard(metrics, colours, tag, out):
    names = list(metrics)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.0))
    width = 0.8 / len(names)
    x0 = np.arange(len(WARD_KEYS))
    for k, name in enumerate(names):
        vals = [metrics[name][f"Ward {key}"] for key in WARD_KEYS]
        x = x0 + (k - (len(names) - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.92, color=colours[name], label=name)
    ax.set_xticks(x0, [f"{WARD_NAMES[k]}\n({k})" for k in WARD_KEYS], fontsize=8)
    ax.set_ylabel("Number of events")
    _title(ax, f"Ward-style event quality, INTRUDER output ({tag})", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "ward_events.png")
    plt.close(fig)

# Show detection and false alarm tradeoffs across thresholds
def figOperatingCurve(evalData, systems, metrics, colours, tag, out):
    """Event Pd vs Pfa when INTRUDER = (confidence >= t), t swept 0..1."""
    recordings = Recordings(evalData)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.8, 3.6))
    for name, system in systems.items():
        if system.get("scores") is None:
            continue
        pts = []

        # Sweep confidence thresholds to build the operating curve
        for t in np.linspace(0.02, 0.98, 49):
            states = np.where(np.asarray(system["scores"]) >= t, 2, 0)
            m = recordings.core(states)
            pts.append((m["Pfa (%)"], m["Event Pd (%)"]))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=colours[name], linewidth=2, label=name)
        ax.plot(metrics[name]["Pfa (%)"], metrics[name]["Event Pd (%)"], "o", markersize=8,
                color=colours[name], markeredgecolor="white", markeredgewidth=2)
    pir = metrics["Thresholded PIR"]
    ax.plot(pir["Pfa (%)"], pir["Event Pd (%)"], "s", markersize=8, color=PIR_COLOUR,
            markeredgecolor="white", markeredgewidth=2, label="Thresholded PIR")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("False alarm probability (%)")
    ax.set_ylabel("Event probability of detection (%)")
    _title(ax, f"Operating curve ({tag})\nlines: single alarm threshold swept; dots: three-state system",
                 loc="left", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "operating_curve.png")
    plt.close(fig)

# Highlight labelled time intervals on the plots
def _shade(ax, times, flags, colour, alpha):
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
    for s, e in zip(starts, ends):
        ax.axvspan(times[s], times[e], color=colour, alpha=alpha, linewidth=0)

# Plot human labels, probabilities and alarm states over time
def figTimelines(evalData, systems, colours, tag, out):
    fusedName = next(n for n in systems if n != "Thresholded PIR")
    fused = systems[fusedName]
    th = fused.get("thresholds")

    # Plot the score and alarm history of each recording
    for item in Recordings(evalData).items:
        t = item["times"]
        inbound = item["human"] == 1
        outbound = item["outbound"] == 1
        score = np.asarray(fused["scores"])[item["idx"]]
        fusedStates = np.asarray(fused["states"])[item["idx"]]
        pirStates = np.asarray(systems["Thresholded PIR"]["states"])[item["idx"]]
        fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
        for ax in axes:
            _shade(ax, t, outbound & ~inbound, OUTBOUND, 0.18)
            _shade(ax, t, inbound, INBOUND, 0.15)
        axes[0].plot(t, score, color=colours[fusedName], linewidth=1.5)
        if th:
            for level in (th["intruder_enter"], th["maybe_enter"]):
                axes[0].axhline(level, color=TEXT_2, linestyle="--", linewidth=0.8)
            axes[0].text(t[0], max(th["intruder_enter"], th["maybe_enter"]) + 0.02,
                         f" enter thresholds: maybe {th['maybe_enter']:.2f}, intruder {th['intruder_enter']:.2f}",
                         color=TEXT_2, fontsize=7, va="bottom")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Fused\nconfidence")
        axes[1].step(t, fusedStates, where="post", color=colours[fusedName], linewidth=1.5)
        axes[1].set_yticks([0, 1, 2], ["Clear", "Maybe", "Intruder"], fontsize=8)
        axes[1].set_ylabel("Fused\nstate")
        axes[2].step(t, pirStates, where="post", color=PIR_COLOUR, linewidth=1.5)
        axes[2].set_yticks([0, 2], ["Clear", "Intruder"], fontsize=8)
        axes[2].set_ylabel("PIR\nstate")
        axes[2].set_xlabel("Time (s)")
        axes[0].legend(handles=[Patch(color=INBOUND, alpha=0.3, label="Person in detection zone (inbound)"),
                                Patch(color=OUTBOUND, alpha=0.4, label="Person outside zone (outbound)")],
                       loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, ncol=2)
        name = Path(item["recording"]).name
        _title(axes[0], f"{name}: {fusedName} vs thresholded PIR ({tag})", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"timeline_{name}.png")
        plt.close(fig)

# Plot false alarms near people and in isolated periods
def figFalseAlarms(breakdowns, tag, out):
    names = list(breakdowns)
    edge = [breakdowns[n]["ALL"][0] for n in names]
    iso = [breakdowns[n]["ALL"][1] for n in names]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.0))
    y = np.arange(len(names))
    ax.barh(y, edge, color="#eda100", label=f"Edge (within {EDGE_WINDOW_S:g} s of a person)",
            edgecolor="white", linewidth=2)
    ax.barh(y, iso, left=edge, color="#e34948", label="Isolated", edgecolor="white", linewidth=2)
    for yi, e, i in zip(y, edge, iso):
        ax.text(e + i, yi, f"  {e + i} ({100 * e / max(e + i, 1):.0f}% edge)", va="center", color=TEXT, fontsize=8)
    ax.set_yticks(y, names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("False-alarm fusion intervals")
    ax.set_xlim(0, max(e + i for e, i in zip(edge, iso)) * 1.35 + 1)
    _title(ax, f"Where the false alarms happen ({tag})", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "false_alarms.png")
    plt.close(fig)

# Show which sensor features contribute most to the model
def figFeatureImportance(names, importance, out, top=15):

    # Select the most influential features
    order = np.argsort(importance)[::-1][:top]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 0.25 * len(order) + 0.9))
    y = np.arange(len(order))
    colours = [SYSTEM_COLOURS[0] if names[i].startswith("thermal") else SYSTEM_COLOURS[1] for i in order]
    ax.barh(y, 100 * np.asarray(importance)[order], color=colours)
    ax.set_yticks(y, [names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of squared-error reduction (%)")
    _title(ax, "First-principles Gradient Boosting: most important features", loc="left", fontsize=10)
    ax.legend(handles=[Patch(color=SYSTEM_COLOURS[0], label="Thermopile"),
                       Patch(color=SYSTEM_COLOURS[1], label="PIR")], fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png")
    plt.close(fig)

# Compare the custom and library Gradient Boosting outputs
def figImplementationCheck(firstScores, libraryScores, metrics, firstName, libraryName, out,
                           filename="fp_vs_sklearn.png"):
    colours = {firstName: SYSTEM_COLOURS[0], libraryName: SYSTEM_COLOURS[1]}
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=TEXT_2, linestyle="--", linewidth=0.8)

    # Compare the confidence scores from both implementations
    ax.scatter(libraryScores, firstScores, s=4, alpha=0.35, color=SYSTEM_COLOURS[0], linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{libraryName} confidence", fontsize=9)
    ax.set_ylabel(f"{firstName} confidence", fontsize=9)
    ax = axes[1]
    keys = [("Event Pd (%)", "Pd"), ("Pfa (%)", "Pfa"), ("Accuracy (%)", "Accuracy")]
    names = [firstName, libraryName]
    width = 0.38
    for k, name in enumerate(names):
        x = np.arange(len(keys)) + (k - 0.5) * width
        vals = [metrics[name][key] for key, _ in keys]
        ax.bar(x, vals, width=width * 0.92, color=colours[name], label=name)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", color=TEXT, fontsize=7)
    ax.set_xticks(np.arange(len(keys)), [label for _, label in keys])
    ax.set_ylabel("%", fontsize=9)
    ax.set_ylim(0, 112)
    ax.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(Path(out) / filename, bbox_inches="tight")
    plt.close(fig)

# Write the evaluation report and generate its figures
def writeEvaluationReport(runName, evalData, systems, outDir, description, tag,
                          featureNames=None, importance=None, extraLines=(), timelines=True):
    """systems: {name: {"states", "scores" (optional), "thresholds" (optional)}};
    must contain "Thresholded PIR". The first other system is the main fused system."""
    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=False)
    recordings = Recordings(evalData)

    # Calculate all metrics using the same recording labels
    metrics = {name: evaluateStates(evalData, s["states"], recordings) for name, s in systems.items()}
    colours = _colours(systems)

    lines = [f"EVALUATION REPORT: {runName}", description, *DEFINITIONS, *extraLines]

    # Add the selected alarm settings
    lines += thresholdTable(systems)

    # Add the project requirement comparison
    lines += requirementTable(metrics)

    # Add event quality measurements
    lines += wardTable(metrics)

    # Separate and report the false alarms
    faLines, breakdowns = falseAlarmTable(evalData, systems)
    lines += faLines

    # Add results for individual recordings
    lines += perRecordingTable(evalData, systems)
    if importance is not None:

        # Select the most influential features
        order = np.argsort(importance)[::-1]
        lines += ["\nFEATURE IMPORTANCE (first-principles Gradient Boosting, share of squared-error reduction)"]
        lines += [f"{featureNames[i]:36s} {100 * importance[i]:6.2f}%" for i in order]

    figRequirements(metrics, colours, tag, out)
    figImprovements(metrics, colours, tag, out)
    figWard(metrics, colours, tag, out)
    figFalseAlarms(breakdowns, tag, out)
    figOperatingCurve(evalData, systems, metrics, colours, tag, out)
    if timelines:
        figTimelines(evalData, systems, colours, tag, out)
    if importance is not None:
        figFeatureImportance(featureNames, importance, out)

    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    return text, metrics
