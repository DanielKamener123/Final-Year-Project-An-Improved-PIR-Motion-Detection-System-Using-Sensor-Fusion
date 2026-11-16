from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from Fusion.dataset import buildRecordingDataset, recordingFolders
from Fusion.evaluation import addOutboundLabels, runStateMachine
from Fusion.gradientBoosting import GradientBoosting

MODEL_PATH = Path('Processed/realtime/fusion_gb.json')
REVIEW_SPLITS = [Path('Datasets/validation'), Path('Datasets/testing')]


# Predictions For One Recording -> review_predictions.csv
def exportRecording(model, recording, output=None):
    recording = Path(recording)
    labelled = (recording / 'labels.csv').is_file()
    data = buildRecordingDataset(recording, windowSeconds=model.window, requireLabels=False)
    if data is None or len(data) == 0:
        print(f'  {recording.name}: no usable fusion intervals, skipped')
        return None
    score = model.predictProba(data[model.features].to_numpy(dtype=float))
    state = runStateMachine(data, score, model.thresholds)

    out = pd.DataFrame({
        'time_s': data['fusionTime'].to_numpy(float),
        'confidence': score,
        'state': state,
        'human': data['human'].to_numpy(int),
        'labelled': int(labelled),
    })
    # Project false-alarm definition when outbound labels exist
    if not labelled:
        out['false_alarm'] = 0
    elif (recording / 'labels(outbound).csv').is_file():
        outbound = addOutboundLabels(data)['outboundHuman'].to_numpy(int)
        out['false_alarm'] = ((out['human'] == 0) & ((out['state'] == 2) |
                                                      ((out['state'] == 1) & (outbound == 0)))).astype(int)
    else:
        out['false_alarm'] = ((out['human'] == 0) & (out['state'] > 0)).astype(int)
    out = out.sort_values('time_s', kind='stable').reset_index(drop=True)

    target = Path(output) if output else recording / 'review_predictions.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    note = f'{int(out["false_alarm"].sum())} false-alarm intervals' if labelled else 'not labelled'
    print(f'  {recording.name}: {len(out)} decisions ({note})')
    return target


# Predictions For Every Validation And Testing Recording
def exportAll(modelPath=MODEL_PATH, splits=REVIEW_SPLITS):
    model = GradientBoosting.load(modelPath)
    print(f'Exporting review predictions with {modelPath} ({model.window:g} s window)...')
    for split in splits:
        for recording in recordingFolders(split):
            exportRecording(model, recording)


def main():
    parser = argparse.ArgumentParser(description='Export model confidence and states for synchronised review.')
    parser.add_argument('--recording', type=Path, help='One recording (default: every validation and testing recording)')
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--output', type=Path, help='Only with --recording. Default: recording/review_predictions.csv')
    args = parser.parse_args()

    if args.recording is None:
        exportAll(args.model)
        return
    recording = args.recording.resolve()
    if not recording.is_dir():
        parser.error(f'Recording folder not found: {recording}')
    if 'training' in recording.parts:
        parser.error('Refusing to review a training recording. Use a validation/testing recording.')
    exportRecording(GradientBoosting.load(args.model), recording, args.output)


if __name__ == '__main__':
    main()
