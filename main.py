from pathlib import Path
import argparse
import re

import pandas as pd

from Fusion.dataset import buildDataset
from Fusion.models import runModels


# Offline Fusion Configuration
HISTORY_WINDOWS_SECONDS = [1.0, 1.5, 2.0, 2.5, 3.0]

# A Sensor Is Marked Invalid If Its Latest Reading Is Older Than This Many Expected Sample Periods
MAX_PIR_AGE_PERIODS = 2.0

TRAIN_DATASET = Path("Datasets/training")
VALIDATION_DATASET = Path("Datasets/validation")

OUTPUT_ROOT = Path("Processed")


def format_compact_results(results_df, run_name):
    """Format a short report without depending on Fusion.models reporting helpers."""
    columns = [
        ("History Window (s)", "Win"),
        ("Algorithm", "Model"),
        ("PIR-Only Event Pd (%)", "PIR ePd"),
        ("Fused Event Pd (%)", "Fuse ePd"),
        ("Fused Confirmed Event Pd (%)", "Conf ePd"),
        ("PIR-Only False Alarm Probability (%)", "PIR Pfa"),
        ("Fused False Alarm Probability (%)", "Fuse Pfa"),
        ("Fused Confirmed Pfa (%)", "Conf Pfa"),
        ("False Alarm Probability Improvement (%)", "Pfa drop"),
        ("Fused Accuracy (%)", "Fuse Acc"),
        ("Accuracy Improvement (%)", "Acc gain"),
    ]
    widths = [max(len(label), 7) for _, label in columns]
    widths[1] = 23

    def format_cell(value, width, model=False):
        if model:
            return str(value).ljust(width)
        if pd.isna(value):
            return "N/A".rjust(width)
        return f"{float(value):.2f}".rjust(width)

    def join_cells(values):
        return " | ".join(values)

    heading = join_cells([label.ljust(width) for (_, label), width in zip(columns, widths)])
    rule = "-" * len(heading)
    lines = [
        f"OFFLINE VALIDATION: {run_name}",
        heading,
        rule,
    ]
    # Confirmed metrics are reported separately from all-alert metrics.
    for _, record in results_df.iterrows():
        cells = [
            format_cell(record.get(key, float("nan")), widths[i], model=(key == "Algorithm"))
            for i, (key, _) in enumerate(columns)
        ]
        lines.append(join_cells(cells))
    lines.extend([rule, "", "ePd: event detection with MAYBE or INTRUDER; Pfa: INTRUDER outside inbound or MAYBE outside both inbound and outbound.",
                  "Confirmed-event detection and confirmed-Pfa use INTRUDER alone.",
                  "Thresholds are provisional and must be tuned on validation recordings.",
                  "", "Targets: relative accuracy gain >= 20%; Pfa reduction >= 50%."])
    qualifying = results_df.loc[(results_df["Accuracy Improvement (%)"] >= 20) &
                                (results_df["False Alarm Probability Improvement (%)"] >= 50)]
    for _, row in qualifying.iterrows():
        lines.append(f"PASS: window={row['History Window (s)']:.2f}s; model={row['Algorithm']}")
    if qualifying.empty:
        lines.append("No configuration meets both relative targets.")
    # Ward-style event diagnostics in a second compact table, kept in the same TXT.
    event_columns = [
        ("History Window (s)", "Win"), ("Algorithm", "Model"),
        ("Fused Ward C", "C"), ("Fused Ward D", "D"),
        ("Fused Ward F", "F"), ("Fused Ward M", "M"),
        ("Fused Ward FM", "FM"), ("Fused Ward I", "I"),
        ("Fused Ward Preexisting", "Pre"),
        ("Fused Ward Fresh Event Pd (%)", "Fresh Pd"),
        ("Fused Ward Fresh Event Delay (s)", "Delay s"),
    ]
    lines.extend(["", "WARD-STYLE EVENT QUALITY (FUSION, MAYBE OR INTRUDER)",
                  "C=correct 1:1; D=missed; F=fragmented; M=merged; FM=both; I=inserted false alert.",
                  "Pre=events already in alert immediately before entry.",
                  "Fresh Pd counts a new OFF->ON alert within 10s; left-censored entries excluded.",
                  "Counts are independent per recording. These are overlap-based diagnostics;",
                  "not an exact implementation of every 2SET frame subcategory."])
    ew = [max(len(header), 7) if key != "Algorithm" else 23 for key, header in event_columns]
    lines.append(join_cells([header.ljust(ew[i]) for i, (_, header) in enumerate(event_columns)]))
    lines.append("-" * len(lines[-1]))
    for _, record in results_df.iterrows():
        lines.append(join_cells([format_cell(record.get(key, float("nan")), ew[i], model=key == "Algorithm")
                                 for i, (key, _) in enumerate(event_columns)]))
    return "\n".join(lines) + "\n"


def main(run_name=None):
    if not run_name:
        run_name = input("Name this results file: ").strip()
    if not run_name:
        raise ValueError("Please provide a filename for this experiment")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip()).strip("._")
    if not safe_name:
        raise ValueError("Filename must contain letters or numbers")
    if safe_name.lower().endswith(".txt"):
        safe_name = safe_name[:-4]
    if not safe_name:
        raise ValueError("Filename cannot be just .txt")
    output_path = OUTPUT_ROOT / f"{safe_name}.txt"
    if output_path.exists():
        raise FileExistsError(f"Results file already exists: {output_path}. Choose another name.")
    print(f"Experiment: {safe_name}")
    allResults = []

    for windowSeconds in HISTORY_WINDOWS_SECONDS:
        print(f"\nPIR history: {windowSeconds:.2f}s")

        print("Building training feature dataset from raw recordings...")
        trainData = buildDataset(
            TRAIN_DATASET,
            windowSeconds=windowSeconds,
            maxPirAgePeriods=MAX_PIR_AGE_PERIODS,
        )
        print(f"Training rows: {len(trainData)}")

        print("\nBuilding validation feature dataset from raw recordings...")
        validationData = buildDataset(
            VALIDATION_DATASET,
            windowSeconds=windowSeconds,
            maxPirAgePeriods=MAX_PIR_AGE_PERIODS,
        )
        print(f"Validation rows: {len(validationData)}")

        print("\nRunning machine-learning algorithms...")
        results = runModels(trainData, validationData)
        results.insert(0, "History Window (s)", float(windowSeconds))
        allResults.append(results)

    if allResults:
        comparison = pd.concat(allResults, ignore_index=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        report = format_compact_results(comparison, safe_name)
        # Write one simple tabulated text file. Never create a run folder or CSV.
        with output_path.open("x", encoding="utf-8") as file:
            file.write(report)
        print("\n" + report)
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline PIR / thermal feature-fusion validation")
    parser.add_argument("--name", help="Output filename, e.g. thermal25_no_energy (without .txt)")
    args = parser.parse_args()
    main(run_name=args.name)
