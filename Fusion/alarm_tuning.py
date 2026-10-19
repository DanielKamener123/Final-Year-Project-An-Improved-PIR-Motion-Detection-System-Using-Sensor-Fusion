"""Alarm-threshold tuning for the three-state output (tuned on VALIDATION only).

The fused model outputs a confidence per fusion interval. The three-state
machine turns it into CLEAR / MAYBE / INTRUDER using four thresholds with
hysteresis, plus an optional off-delay: INTRUDER only drops once the
confidence has stayed below intruder_exit for hold_s seconds.

Search: every valid combination in the grid below. Choice rule:
  1. settings with event Pd >= TARGET_PD and Pfa <= TARGET_PFA: pick the one
     with the highest frame accuracy (spec 1 is an accuracy improvement), then
     higher Pd, then fewer false-alarm episodes
  2. otherwise, settings with event Pd >= TARGET_PD: lowest Pfa, then highest
     accuracy
  3. otherwise: highest Pd, then lowest Pfa (flagged as not meeting the target)
Accuracy = alert (MAYBE or INTRUDER) vs human label per fusion interval, the
same definition the report uses.

Tuning on validation makes the validation numbers optimistic. The thresholds
are frozen in Processed/realtime/alarm_thresholds.json and the independent
result comes from the testing recordings (offlineRun.py --test).
"""
import json
from pathlib import Path

import numpy as np

TARGET_PD = 95.0
TARGET_PFA = 5.0
MAX_DELAY_S = 10.0
THRESHOLD_FILE = Path("Processed/realtime/alarm_thresholds.json")

GRID_INTRUDER_ENTER = [round(0.50 + 0.025 * k, 3) for k in range(17)]   # 0.50 .. 0.90
GRID_INTRUDER_GAP = [0.05, 0.10, 0.15, 0.20]       # intruder_enter - intruder_exit
GRID_MAYBE_BELOW = [0.01, 0.05, 0.10]              # intruder_exit - maybe_enter
GRID_MAYBE_GAP = [0.05, 0.10, 0.15]                # maybe_enter - maybe_exit
GRID_HOLD_S = [0.0, 1.0, 2.0]

NO_INTRUDER, MAYBE_INTRUDER, INTRUDER = 0, 1, 2


def run_state_machine(scores, times, thresholds):
    """Causal three-state machine with hysteresis and INTRUDER off-delay.
    Same logic as Fusion/realtimeModel.StateMachine (one recording)."""
    me, mx = thresholds["maybe_enter"], thresholds["maybe_exit"]
    ie, ix = thresholds["intruder_enter"], thresholds["intruder_exit"]
    hold = thresholds.get("hold_s", 0.0)
    states = np.zeros(len(scores), dtype=int)
    state = NO_INTRUDER
    below_since = None
    for k in range(len(scores)):
        s = scores[k]
        t = times[k]
        if state == NO_INTRUDER:
            if s >= ie:
                state = INTRUDER
            elif s >= me:
                state = MAYBE_INTRUDER
        elif state == MAYBE_INTRUDER:
            if s >= ie:
                state = INTRUDER
            elif s < mx:
                state = NO_INTRUDER
        else:
            if s >= ix:
                below_since = None
            else:
                if below_since is None:
                    below_since = t
                if t - below_since >= hold:
                    state = MAYBE_INTRUDER if s >= mx else NO_INTRUDER
                    below_since = None
        states[k] = state
    return states


class Recordings:
    """Per-recording arrays, sorted by time, prepared once for fast scoring."""

    def __init__(self, eval_data, scores):
        self.items = []
        scores = np.asarray(scores, dtype=float)
        for _, idx in eval_data.groupby("recording", sort=False).indices.items():
            idx = np.asarray(idx)
            order = np.argsort(eval_data["fusionTime"].to_numpy()[idx], kind="stable")
            idx = idx[order]
            times = eval_data["fusionTime"].to_numpy(dtype=float)[idx]
            human = eval_data["human"].to_numpy(dtype=int)[idx]
            outbound = eval_data["outboundHuman"].to_numpy(dtype=int)[idx]
            starts = np.flatnonzero((human == 1) & (np.r_[0, human[:-1]] == 0))
            ends = np.flatnonzero((human == 1) & (np.r_[human[1:], 0] == 0))
            events = [(s, e) for s, e in zip(starts, ends) if s != 0]   # left-censored excluded
            self.items.append(dict(idx=idx, times=times, human=human, outbound=outbound,
                                   scores=scores[idx], events=events))

    def evaluate(self, thresholds):
        events = detected = 0
        negatives = false = episodes = 0
        correct = total = 0
        for r in self.items:
            st = run_state_machine(r["scores"], r["times"], thresholds)
            for s, e in r["events"]:
                events += 1
                deadline = min(r["times"][s] + MAX_DELAY_S, r["times"][e])
                for k in range(s, e + 1):
                    if r["times"][k] > deadline:
                        break
                    if st[k] == INTRUDER:
                        detected += 1
                        break
            neg = r["human"] == 0
            fa = neg & ((st == INTRUDER) | ((st == MAYBE_INTRUDER) & (r["outbound"] == 0)))
            alert = st >= MAYBE_INTRUDER
            correct += int(np.count_nonzero(alert == (r["human"] == 1)))
            total += len(st)
            negatives += int(neg.sum())
            false += int(fa.sum())
            episodes += int(np.count_nonzero(fa & ~np.r_[False, fa[:-1]]))
        pd_ = 100.0 * detected / events if events else np.nan
        pfa = 100.0 * false / negatives if negatives else np.nan
        acc = 100.0 * correct / total if total else np.nan
        return pd_, pfa, episodes, acc


def candidate_grid():
    for ie in GRID_INTRUDER_ENTER:
        for gap in GRID_INTRUDER_GAP:
            ix = round(ie - gap, 3)
            for below in GRID_MAYBE_BELOW:
                me = round(ix - below, 3)
                for mgap in GRID_MAYBE_GAP:
                    mx = round(me - mgap, 3)
                    if not (0 <= mx < me < ix < ie <= 1):
                        continue
                    for hold in GRID_HOLD_S:
                        yield {"maybe_enter": me, "maybe_exit": mx,
                               "intruder_enter": ie, "intruder_exit": ix, "hold_s": hold}


def tune(eval_data, scores):
    recordings = Recordings(eval_data, scores)
    best = None
    for th in candidate_grid():
        pd_, pfa, episodes, acc = recordings.evaluate(th)
        if pd_ >= TARGET_PD and pfa <= TARGET_PFA:
            key = (0, -acc, -pd_, episodes)
        elif pd_ >= TARGET_PD:
            key = (1, pfa, -acc, episodes)
        else:
            key = (2, -pd_, pfa, -acc)
        if best is None or key < best[0]:
            best = (key, th, pd_, pfa, episodes, acc)
    key, th, pd_, pfa, episodes, acc = best
    return {"thresholds": th, "pd": pd_, "pfa": pfa, "episodes": episodes, "accuracy": acc,
            "meets_target": key[0] == 0, "meets_pd": key[0] <= 1}


def save_thresholds(tuned, path=THRESHOLD_FILE):
    """tuned: {window: {model: tune() result}} -> JSON used by export and the ROCK."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {f"{float(w):.2f}": {m: r["thresholds"] for m, r in models.items()} for w, models in tuned.items()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_thresholds(window, model, path=THRESHOLD_FILE):
    path = Path(path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(f"{float(window):.2f}", {}).get(model)
