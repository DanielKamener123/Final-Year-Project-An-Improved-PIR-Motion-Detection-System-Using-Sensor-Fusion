"""Continuous-event diagnostics inspired by Ward, Lukowicz & Gellersen (2011).

Apply independently to each recording and separately to the any-alert and
confirmed-only versions of a three-state output. The original frame/event
metrics remain unchanged. No event is matched across recording boundaries.
"""
import numpy as np
import pandas as pd


def _runs(values):
    values = np.asarray(values, dtype=bool)
    if not len(values):
        return []
    start = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    end = np.flatnonzero(values & ~np.r_[values[1:], False])
    return list(zip(start.tolist(), end.tolist()))


def analyze_events(eval_data, predictions, max_delay_s=10.0):
    """Summarise event overlap topology and conservative new-alert responsiveness.

    Event categories are Ward-style: C (one-to-one), D (deleted), F
    (fragmented), M (merged), FM (both). I is a predicted event with no
    overlap. Counts reflect sampled fusion decision intervals; there is no
    tolerance or boundary masking. Overfill and underfill are the sampled
    false-positive/false-negative time adjacent to an overlapped event.

    A strict new-alert detection requires output to be OFF immediately before
    the ground-truth entry and a new OFF->ON transition in the next 10 s,
    while the person is present. An event starting in alarm is counted as
    pre-existing, *not* as a fresh detection.
    """
    required = {"recording", "fusionTime", "human"}
    if not required.issubset(eval_data.columns):
        raise ValueError(f"Missing required columns: {sorted(required-set(eval_data.columns))}")
    output = np.asarray(predictions, dtype=int)
    if len(output) != len(eval_data) or not np.isin(output, [0, 1]).all():
        raise ValueError("Predictions must be aligned binary 0/1 values")
    groups = eval_data[["recording", "fusionTime", "human"]].copy()
    groups["prediction"] = output
    counts = {k: 0 for k in ("C", "D", "F", "M", "FM", "I", "PredEvents", "TrueEvents", "Preexisting", "Eligible", "NewDetected")}
    delays = []
    overfill_s = underfill_s = 0.0
    for recording, group in groups.groupby("recording", sort=False):
        group = group.sort_values("fusionTime", kind="stable")
        times = group["fusionTime"].to_numpy(dtype=float)
        gt = group["human"].to_numpy(dtype=int)
        pred = group["prediction"].to_numpy(dtype=int)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid fusion timestamps for {recording}")
        if not np.isin(gt, [0, 1]).all():
            raise ValueError("Ground truth must be binary")
        tr = _runs(gt)
        pr = _runs(pred)
        counts["TrueEvents"] += len(tr)
        counts["PredEvents"] += len(pr)
        ground_to_pred = [set() for _ in tr]
        pred_to_ground = [set() for _ in pr]
        for gi, (a, b) in enumerate(tr):
            for pi, (c, d) in enumerate(pr):
                if max(a, c) <= min(b, d):
                    ground_to_pred[gi].add(pi)
                    pred_to_ground[pi].add(gi)
        counts["I"] += sum(not matches for matches in pred_to_ground)
        for gi, (a, b) in enumerate(tr):
            links = ground_to_pred[gi]
            if not links:
                counts["D"] += 1
            else:
                frag = len(links) > 1
                merge = any(len(pred_to_ground[p]) > 1 for p in links)
                counts["FM" if frag and merge else "F" if frag else "M" if merge else "C"] += 1
            # Exclude left-censored events from fresh-alert responsiveness.
            if a == 0:
                continue
            counts["Eligible"] += 1
            if pred[a - 1] == 1:
                counts["Preexisting"] += 1
                continue
            deadline = times[a] + max_delay_s
            for i in range(a, b + 1):
                if times[i] > deadline:
                    break
                if pred[i] and not pred[i - 1]:
                    counts["NewDetected"] += 1
                    delays.append(float(times[i] - times[a]))
                    break
        # Approximate decision-frame durations from the actual timestamp gaps;
        # last interval uses the median preceding interval (if available).
        if len(times) > 1:
            delta = np.diff(times)
            widths = np.r_[delta, np.median(delta)]
        else:
            widths = np.array([0.0])
        # Assign non-overlapping alert tails adjacent to true event edges as overfill.
        # Anything outside truth not adjacent to a matched return is an insertion;
        # fragments/merge are separately captured above. This is an accessible
        # diagnostic, not an exact reproduction of paper's 2SET categories.
        for pi, (c, d) in enumerate(pr):
            if pred_to_ground[pi]:
                overfill_s += float(np.sum(widths[c:d + 1][gt[c:d + 1] == 0]))
        for gi, (a, b) in enumerate(tr):
            if ground_to_pred[gi]:
                underfill_s += float(np.sum(widths[a:b + 1][pred[a:b + 1] == 0]))
    counts.update({
        "Overfill s": overfill_s,
        "Underfill s": underfill_s,
        "Fresh Event Pd (%)": (100 * counts["NewDetected"] / counts["Eligible"] if counts["Eligible"] else np.nan),
        "Fresh Event Delay (s)": float(np.mean(delays)) if delays else np.nan,
    })
    return counts
