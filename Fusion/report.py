"""Evaluation report: full fused system vs thresholded PIR (validation data).

Definitions (project rules):
  Pd   = event probability of detection. A labelled human event counts as
         detected only if the system reaches INTRUDER at least once within
         10 s of the labelled entry (Confirmed Event Pd).
  Pfa  = fraction of non-human fusion intervals that are false alarms:
         MAYBE or INTRUDER while nobody is inbound or outbound, or INTRUDER
         while a person is only in the outbound zone.
  Pd improvement   = (fused Pd - PIR Pd) / PIR Pd x 100
  Pfa reduction    = (PIR Pfa - fused Pfa) / PIR Pfa x 100
  Accuracy gain    = (fused acc - PIR acc) / PIR acc x 100  (used for the
                     ">= 20% better detection" requirement for now)

Outputs (in Processed/<run name>/):
  report.txt                          tables
  requirements_<win>s.png             Pd and Pfa, PIR vs fused, with limits
  improvements_<win>s.png             Pd improvement, Pfa/accuracy gains vs targets
  ward_events_<win>s.png              Ward-style event categories
  operating_curve_<win>s.png          event Pd vs Pfa as the alarm threshold moves
  timeline_<recording>_<win>s.png     labels, PIR state, fused confidence/state
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Fusion.models import (
    calculateEventMetrics, LABEL_COLUMN,
    MAYBE_ENTER, MAYBE_EXIT, INTRUDER_ENTER, INTRUDER_EXIT,
)

# Requirements (project specification)
PD_MIN = 95.0
PFA_MAX = 5.0
ACC_GAIN_MIN = 20.0
PFA_DROP_MIN = 50.0

# Palette (validated reference categorical slots + neutral baseline)
COLOURS = {"PIR": "#8a8984", "Random Forest": "#2a78d6", "RBF SVM": "#eb6834"}
INBOUND = "#2a78d6"
OUTBOUND = "#eda100"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
})

WARD_KEYS = ("C", "D", "F", "M", "FM", "I")
WARD_NAMES = {"C": "Correct", "D": "Missed", "F": "Fragmented", "M": "Merged",
              "FM": "Frag.+merged", "I": "Inserted"}


def _models(results_df):
    return list(results_df["Algorithm"])


def _row(results_df, model):
    return results_df.loc[results_df["Algorithm"] == model].iloc[0]


def _improvement(new, old, reduction=False):
    if old is None or not np.isfinite(old) or old == 0:
        return np.nan
    return 100.0 * ((old - new) if reduction else (new - old)) / old


def requirement_table(window, results_df):
    lines = [f"\nREQUIREMENTS - fusion window {window:.2f} s (validation recordings)",
             f"{'System':20s} | {'Pd %':>6} | {'Pfa %':>6} | {'Pd impr %':>9} | {'Pfa red %':>9} | "
             f"{'Acc %':>6} | {'Acc gain %':>10} | {'Delay s':>7} | Pd>=95  Pfa<=5  Acc+20  Pfa-50",
             "-" * 132]
    first = _row(results_df, _models(results_df)[0])
    pir_pd = first["PIR-Only Confirmed Event Pd (%)"]
    pir_pfa = first["PIR-Only False Alarm Probability (%)"]
    pir_acc = first["PIR-Only Accuracy (%)"]
    lines.append(f"{'Thresholded PIR':20s} | {pir_pd:6.2f} | {pir_pfa:6.2f} | {'-':>9} | {'-':>9} | "
                 f"{pir_acc:6.2f} | {'-':>10} | {first['PIR-Only Mean Event Delay (s)']:7.2f} | "
                 f"{'PASS' if pir_pd >= PD_MIN else 'FAIL':>6}  {'PASS' if pir_pfa <= PFA_MAX else 'FAIL':>6}")
    for model in _models(results_df):
        row = _row(results_df, model)
        pd_ = row["Fused Confirmed Event Pd (%)"]
        pfa = row["Fused False Alarm Probability (%)"]
        acc = row["Fused Accuracy (%)"]
        pd_imp = _improvement(pd_, pir_pd)
        pfa_red = _improvement(pfa, pir_pfa, reduction=True)
        acc_gain = _improvement(acc, pir_acc)
        checks = [pd_ >= PD_MIN, pfa <= PFA_MAX, acc_gain >= ACC_GAIN_MIN, pfa_red >= PFA_DROP_MIN]
        flags = "  ".join(f"{'PASS' if ok else 'FAIL':>6}" for ok in checks)
        delay = row.get("Fused Confirmed Mean Event Delay (s)", row["Fused Mean Event Delay (s)"])
        lines.append(f"{'Fused ' + model:20s} | {pd_:6.2f} | {pfa:6.2f} | {pd_imp:9.2f} | {pfa_red:9.2f} | "
                     f"{acc:6.2f} | {acc_gain:10.2f} | {delay:7.2f} | {flags}")
    lines.append(f"Events evaluated: {int(first['Evaluated Events'])}")
    return lines


def ward_table(window, results_df):
    lines = [f"\nWARD-STYLE EVENT QUALITY - fusion window {window:.2f} s",
             "C=correct 1:1, D=missed, F=fragmented, M=merged, FM=both, I=inserted false alert.",
             "Fresh Pd: new OFF->ON alert within 10 s of entry (events already in alert excluded).",
             f"{'System':34s} | " + " | ".join(f"{k:>4}" for k in WARD_KEYS)
             + f" | {'Fresh Pd':>8} | {'Delay s':>7} | {'Overfill s':>10} | {'Underfill s':>11}",
             "-" * 126]
    first = _row(results_df, _models(results_df)[0])

    def fmt(prefix, name, row):
        vals = " | ".join(f"{row[f'{prefix} {k}']:4.0f}" for k in WARD_KEYS)
        extra = f" | {row[f'{prefix} Fresh Event Pd (%)']:8.2f} | {row[f'{prefix} Fresh Event Delay (s)']:7.2f}"
        fill = ""
        if f"{prefix} Overfill s" in row:
            fill = f" | {row[f'{prefix} Overfill s']:10.1f} | {row[f'{prefix} Underfill s']:11.1f}"
        return f"{name:34s} | {vals}{extra}{fill}"

    lines.append(fmt("PIR-Only Ward", "Thresholded PIR", first))
    for model in _models(results_df):
        row = _row(results_df, model)
        lines.append(fmt("Fused Ward", f"Fused {model} (MAYBE/INTR.)", row))
        lines.append(fmt("Fused Ward Confirmed", f"Fused {model} (INTRUDER)", row))
    return lines


def per_recording_table(window, eval_data, pir_states, states_by_model):
    lines = [f"\nPER RECORDING - fusion window {window:.2f} s",
             "Detected = INTRUDER within 10 s of entry; FA = false-alarm intervals / non-human intervals"]
    header = f"{'Recording':22s} | {'Events':>6} | {'PIR det':>7} | {'PIR FA':>11}"
    for model in states_by_model:
        header += f" | {model[:10] + ' det':>14} | {model[:10] + ' FA':>13}"
    lines += [header, "-" * len(header)]
    for recording, idx in eval_data.groupby("recording", sort=False).indices.items():
        sub = eval_data.iloc[idx]
        negative = sub[LABEL_COLUMN].to_numpy() == 0
        outbound = sub["outboundHuman"].to_numpy() == 1

        def fa(states):
            s = np.asarray(states)[idx]
            false = negative & ((s == 2) | ((s == 1) & ~outbound))
            return f"{int(false.sum())}/{int(negative.sum())}"

        def det(states):
            confirmed = (np.asarray(states)[idx] == 2).astype(int)
            m = calculateEventMetrics(sub, confirmed)
            return int(m["Events"]), int(m["Detected Events"])

        events, pir_det = det(pir_states)
        line = f"{Path(recording).name:22s} | {events:6d} | {pir_det:7d} | {fa(pir_states):>11}"
        for model, states in states_by_model.items():
            line += f" | {det(states)[1]:14d} | {fa(states):>13}"
        lines.append(line)
    return lines


def fig_requirements(window, results_df, out):
    models = _models(results_df)
    first = _row(results_df, models[0])
    systems = ["PIR"] + models
    pd_vals = [first["PIR-Only Confirmed Event Pd (%)"]] + [_row(results_df, m)["Fused Confirmed Event Pd (%)"] for m in models]
    pfa_vals = [first["PIR-Only False Alarm Probability (%)"]] + [_row(results_df, m)["Fused False Alarm Probability (%)"] for m in models]
    labels = ["Thresholded PIR"] + [f"Fused {m}" for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, vals, title, limit, kind in [
        (axes[0], pd_vals, "Event probability of detection", PD_MIN, "min"),
        (axes[1], pfa_vals, "False alarm probability", PFA_MAX, "max"),
    ]:
        x = np.arange(len(systems))
        ax.bar(x, vals, width=0.6, color=[COLOURS.get(s, "#6250d6") for s in systems],
               edgecolor="white", linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9)
        ax.axhline(limit, color=TEXT, linestyle="--", linewidth=1.2)
        ax.text(-0.45, limit, f"requirement {'≥' if kind == 'min' else '≤'} {limit:g}%",
                va="bottom" if kind == "max" else "top", ha="left", color=TEXT_2, fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel("%")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(0, max(max(pfa_vals) * 1.25, PFA_MAX * 2))
    fig.suptitle(f"Fused system vs thresholded PIR ({window:.1f} s window, validation)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"requirements_{window:.1f}s.png")
    plt.close(fig)


def fig_improvements(window, results_df, out):
    models = _models(results_df)
    first = _row(results_df, models[0])
    metrics = ["Pd improvement", "False alarm reduction", "Accuracy gain"]
    targets = [None, PFA_DROP_MIN, ACC_GAIN_MIN]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(models)
    for k, model in enumerate(models):
        row = _row(results_df, model)
        vals = [
            _improvement(row["Fused Confirmed Event Pd (%)"], first["PIR-Only Confirmed Event Pd (%)"]),
            _improvement(row["Fused False Alarm Probability (%)"], first["PIR-Only False Alarm Probability (%)"], True),
            _improvement(row["Fused Accuracy (%)"], first["PIR-Only Accuracy (%)"]),
        ]
        x = np.arange(len(metrics)) + (k - (len(models) - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.92, color=COLOURS.get(model, "#6250d6"), label=f"Fused {model}")
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top", color=TEXT, fontsize=8)
    for i, t in enumerate(targets):
        if t is not None:
            ax.plot([i - 0.45, i + 0.45], [t, t], color=TEXT, linestyle="--", linewidth=1.2)
            ax.text(i - 0.45, t, f"target {t:g}%", va="top", ha="left", color=TEXT_2, fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set_xticks(np.arange(len(metrics)), metrics)
    ax.set_ylabel("Relative to thresholded PIR (%)")
    ax.set_title(f"Improvement over thresholded PIR ({window:.1f} s window)", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / f"improvements_{window:.1f}s.png")
    plt.close(fig)


def fig_ward(window, results_df, out):
    models = _models(results_df)
    first = _row(results_df, models[0])
    systems = [("Thresholded PIR", "PIR-Only Ward", first, COLOURS["PIR"])]
    for m in models:
        systems.append((f"Fused {m} (INTRUDER)", "Fused Ward Confirmed", _row(results_df, m), COLOURS.get(m, "#6250d6")))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / len(systems)
    x0 = np.arange(len(WARD_KEYS))
    for k, (name, prefix, row, colour) in enumerate(systems):
        vals = [row[f"{prefix} {key}"] for key in WARD_KEYS]
        x = x0 + (k - (len(systems) - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.92, color=colour, label=name)
    ax.set_xticks(x0, [f"{WARD_NAMES[k]}\n({k})" for k in WARD_KEYS], fontsize=8)
    ax.set_ylabel("Number of events")
    ax.set_title(f"Ward-style event quality ({window:.1f} s window)", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / f"ward_events_{window:.1f}s.png")
    plt.close(fig)


def fig_operating_curve(window, results_df, eval_data, pir_states, scores_by_model, out):
    """Event Pd vs Pfa when INTRUDER = (confidence >= t), t swept 0..1."""
    negative = eval_data[LABEL_COLUMN].to_numpy() == 0
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for model, scores in scores_by_model.items():
        pts = []
        for t in np.linspace(0.02, 0.98, 49):
            states = (np.asarray(scores) >= t).astype(int)
            pd_ = calculateEventMetrics(eval_data, states)["Event Pd (%)"]
            pfa = 100.0 * np.count_nonzero(negative & (states == 1)) / max(1, np.count_nonzero(negative))
            pts.append((pfa, pd_))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=COLOURS.get(model, "#6250d6"), linewidth=2, label=f"Fused {model}")
        row = _row(results_df, model)
        ax.plot(row["Fused False Alarm Probability (%)"], row["Fused Confirmed Event Pd (%)"], "o",
                markersize=8, color=COLOURS.get(model, "#6250d6"), markeredgecolor="white", markeredgewidth=2)
    first = _row(results_df, _models(results_df)[0])
    ax.plot(first["PIR-Only False Alarm Probability (%)"], first["PIR-Only Confirmed Event Pd (%)"], "s",
            markersize=8, color=COLOURS["PIR"], markeredgecolor="white", markeredgewidth=2, label="Thresholded PIR")
    ax.axvspan(0, PFA_MAX, ymin=0, ymax=1, color=INBOUND, alpha=0.06)
    ax.axhline(PD_MIN, color=TEXT, linestyle="--", linewidth=1)
    ax.axvline(PFA_MAX, color=TEXT, linestyle="--", linewidth=1)
    ax.text(PFA_MAX, 2, " Pfa ≤ 5%", color=TEXT_2, fontsize=8)
    ax.text(99, PD_MIN, "Pd ≥ 95% ", color=TEXT_2, fontsize=8, ha="right", va="top")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("False alarm probability (%)")
    ax.set_ylabel("Event probability of detection (%)")
    ax.set_title(f"Operating curve ({window:.1f} s window)\nlines: single alarm threshold swept; dots: three-state system",
                 loc="left", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"operating_curve_{window:.1f}s.png")
    plt.close(fig)


def _shade(ax, times, flags, colour, alpha):
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
    for s, e in zip(starts, ends):
        ax.axvspan(times[s], times[e], color=colour, alpha=alpha, linewidth=0)


def fig_timelines(window, eval_data, pir_states, scores_by_model, states_by_model, out,
                  thresholds_by_model=None, model="Random Forest"):
    if model not in scores_by_model:
        model = next(iter(scores_by_model))
    th = (thresholds_by_model or {}).get(model) or {
        "maybe_enter": MAYBE_ENTER, "intruder_enter": INTRUDER_ENTER}
    for recording, idx in eval_data.groupby("recording", sort=False).indices.items():
        sub = eval_data.iloc[idx]
        order = np.argsort(sub["fusionTime"].to_numpy(), kind="stable")
        sub = sub.iloc[order]
        idx = np.asarray(idx)[order]
        t = sub["fusionTime"].to_numpy(dtype=float)
        inbound = sub[LABEL_COLUMN].to_numpy() == 1
        outbound = sub["outboundHuman"].to_numpy() == 1
        score = np.asarray(scores_by_model[model])[idx]
        fused = np.asarray(states_by_model[model])[idx]
        pir = np.asarray(pir_states)[idx]
        fig, axes = plt.subplots(3, 1, figsize=(11, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1, 1]})
        for ax in axes:
            _shade(ax, t, outbound & ~inbound, OUTBOUND, 0.18)
            _shade(ax, t, inbound, INBOUND, 0.15)
        axes[0].plot(t, score, color=COLOURS.get(model, "#2a78d6"), linewidth=1.5)
        for level in (th["intruder_enter"], th["maybe_enter"]):
            axes[0].axhline(level, color=TEXT_2, linestyle="--", linewidth=0.8)
        axes[0].text(t[0], max(th["intruder_enter"], th["maybe_enter"]) + 0.02,
                     f" enter thresholds: maybe {th['maybe_enter']:.2f}, intruder {th['intruder_enter']:.2f}",
                     color=TEXT_2, fontsize=7, va="bottom")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Fused\nconfidence")
        axes[1].step(t, fused, where="post", color=COLOURS.get(model, "#2a78d6"), linewidth=1.5)
        axes[1].set_yticks([0, 1, 2], ["Clear", "Maybe", "Intruder"], fontsize=8)
        axes[1].set_ylabel("Fused\nstate")
        axes[2].step(t, pir, where="post", color=COLOURS["PIR"], linewidth=1.5)
        axes[2].set_yticks([0, 2], ["Clear", "Intruder"], fontsize=8)
        axes[2].set_ylabel("PIR\nstate")
        axes[2].set_xlabel("Time (s)")
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color=INBOUND, alpha=0.3, label="Person in detection zone (inbound)"),
                                Patch(color=OUTBOUND, alpha=0.4, label="Person outside zone (outbound)")],
                       loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, ncol=2)
        name = Path(recording).name
        axes[0].set_title(f"{name}: fused {model} vs thresholded PIR ({window:.1f} s window)", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"timeline_{name}_{window:.1f}s.png")
        plt.close(fig)


EDGE_WINDOW_S = 3.0   # false alarm within this time of a person (inbound/outbound) = "edge"


def false_alarm_breakdown(eval_data, states):
    """Split false-alarm intervals into 'edge' (within EDGE_WINDOW_S of any
    labelled person, inbound or outbound) and 'isolated'. Returns
    {recording: (edge, isolated)} plus an 'ALL' total."""
    out = {}
    total_edge = total_iso = 0
    states = np.asarray(states)
    for recording, idx in eval_data.groupby("recording", sort=False).indices.items():
        idx = np.asarray(idx)
        t = eval_data["fusionTime"].to_numpy(dtype=float)[idx]
        inbound = eval_data[LABEL_COLUMN].to_numpy(dtype=int)[idx] == 1
        outbound = eval_data["outboundHuman"].to_numpy(dtype=int)[idx] == 1
        st = states[idx]
        false = ~inbound & ((st == 2) | ((st == 1) & ~outbound))
        person_times = np.sort(t[inbound | outbound])
        edge = iso = 0
        for k in np.flatnonzero(false):
            if person_times.size:
                pos = np.searchsorted(person_times, t[k])
                near = min(abs(t[k] - person_times[max(pos - 1, 0)]),
                           abs(t[k] - person_times[min(pos, person_times.size - 1)]))
            else:
                near = np.inf
            if near <= EDGE_WINDOW_S:
                edge += 1
            else:
                iso += 1
        out[Path(recording).name] = (edge, iso)
        total_edge += edge
        total_iso += iso
    out["ALL"] = (total_edge, total_iso)
    return out


def false_alarm_table(window, eval_data, pir_states, states_by_model):
    systems = {"Thresholded PIR": pir_states}
    systems.update({f"Fused {m}": s for m, s in states_by_model.items()})
    breakdowns = {name: false_alarm_breakdown(eval_data, st) for name, st in systems.items()}
    lines = [f"\nFALSE-ALARM BREAKDOWN - fusion window {window:.2f} s",
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


def fig_false_alarms(window, breakdowns, out):
    names = list(breakdowns)
    edge = [breakdowns[n]["ALL"][0] for n in names]
    iso = [breakdowns[n]["ALL"][1] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
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
    ax.set_title(f"Where the false alarms happen ({window:.1f} s window)", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / f"false_alarms_{window:.1f}s.png")
    plt.close(fig)


def threshold_lines(window, thresholds, default_metrics, results_df, tuning, mode):
    lines = [f"\nALARM THRESHOLDS - fusion window {window:.2f} s (mode: {mode})",
             f"{'Model':16s} | maybe enter/exit | intruder enter/exit | off-delay s"]
    for model, th in thresholds.items():
        lines.append(f"{model:16s} |    {th['maybe_enter']:.3f}/{th['maybe_exit']:.3f} |"
                     f"         {th['intruder_enter']:.3f}/{th['intruder_exit']:.3f} | {th.get('hold_s', 0.0):.1f}")
    if mode == "tune":
        lines += ["Default (0.58/0.45, 0.60/0.59) vs tuned thresholds on the SAME validation data.",
                  "Tuned validation numbers are optimistic; the independent result is the --test run.",
                  f"{'Model':16s} | {'Pd default':>10} | {'Pd tuned':>8} | {'Pfa default':>11} | {'Pfa tuned':>9} | target met"]
        for model, dm in default_metrics.items():
            row = _row(results_df, model)
            info = tuning.get(model, {})
            met = ("Pd and Pfa targets" if info.get("meets_target") else
                   "Pd only (Pfa > 5% for every setting)" if info.get("meets_pd") else "NO (Pd < 95%)")
            lines.append(f"{model:16s} | {dm['Confirmed Event Pd (%)']:10.2f} | {row['Fused Confirmed Event Pd (%)']:8.2f} | "
                         f"{dm['False Alarm Probability (%)']:11.2f} | {row['Fused False Alarm Probability (%)']:9.2f} | "
                         f"{met}")
    return lines


def write_report(run_name, runs, out_dir, mode="tune", dataset="Datasets/validation"):
    """runs: list of dicts with keys window, results_df, eval_data, pir_states,
    scores, states, thresholds, default_metrics, tuning."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    purpose = {"tune": "alarm thresholds TUNED on this data (optimistic)",
               "test": "FINAL VERIFICATION with frozen thresholds",
               "default": "original fixed alarm thresholds"}[mode]
    lines = [f"EVALUATION REPORT: {run_name}",
             f"Full fused system vs thresholded PIR. Trained on Datasets/training, evaluated on {dataset}: {purpose}.",
             "Pd: INTRUDER at least once within 10 s of a labelled entry.",
             "Pfa: MAYBE/INTRUDER with nobody inbound or outbound, or INTRUDER with a person only outbound;",
             "     percentage of non-human fusion intervals (not alarms per hour).",
             f"Requirements: Pd >= {PD_MIN:g}%, Pfa <= {PFA_MAX:g}%, accuracy gain >= {ACC_GAIN_MIN:g}%, "
             f"false-alarm reduction >= {PFA_DROP_MIN:g}%."]
    for run in runs:
        w = run["window"]
        lines += threshold_lines(w, run.get("thresholds", {}), run.get("default_metrics", {}),
                                 run["results_df"], run.get("tuning", {}), mode)
        lines += requirement_table(w, run["results_df"])
        lines += ward_table(w, run["results_df"])
        fa_lines, breakdowns = false_alarm_table(w, run["eval_data"], run["pir_states"], run["states"])
        lines += fa_lines
        lines += per_recording_table(w, run["eval_data"], run["pir_states"], run["states"])
        fig_requirements(w, run["results_df"], out)
        fig_improvements(w, run["results_df"], out)
        fig_ward(w, run["results_df"], out)
        fig_false_alarms(w, breakdowns, out)
        fig_operating_curve(w, run["results_df"], run["eval_data"], run["pir_states"], run["scores"], out)
        fig_timelines(w, run["eval_data"], run["pir_states"], run["scores"], run["states"], out,
                      thresholds_by_model=run.get("thresholds"))
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    return text
