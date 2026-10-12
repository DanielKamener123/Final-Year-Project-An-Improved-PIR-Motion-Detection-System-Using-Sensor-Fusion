"""Export real offline ML outputs for the synchronised playback review tool.

Run from your project root:
 python -m Tools.export_review_predictions --recording Datasets/validation/NAME --model 'RBF SVM' --window 1.5

The model is trained ONLY with Datasets/training, never the reviewed recording.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from Fusion.dataset import buildDataset, buildRecordingDataset
from Fusion.models import buildModels, prepareFeatures, confidenceScores, applyThreeState


def main():
    parser = argparse.ArgumentParser(description='Export model confidence and states for synchronised review.')
    parser.add_argument('--recording', type=Path, required=True, help='One labelled validation/testing recording directory')
    parser.add_argument('--training', type=Path, default=Path('Datasets/training'))
    parser.add_argument('--model', default='RBF SVM', choices=list(buildModels()))
    parser.add_argument('--window', type=float, default=1.5, help='PIR history window in seconds')
    parser.add_argument('--pir-only', action='store_true', help='Use PIR-only rather than feature fusion')
    parser.add_argument('--output', type=Path, help='Default: recording/review_predictions.csv')
    args = parser.parse_args()
    recording = args.recording.resolve()
    if args.window <= 0:
        parser.error('--window must be positive')
    if not recording.is_dir():
        parser.error(f'Recording folder not found: {recording}')
    if args.training.resolve() == recording or args.training.resolve() in recording.parents:
        parser.error('Refusing to evaluate training recordings. Use a validation/testing recording.')

    print('Building training features...')
    train = buildDataset(args.training, windowSeconds=args.window)
    print('Building reviewed recording features...')
    test = buildRecordingDataset(recording, windowSeconds=args.window)
    if test is None or len(test) == 0:
        raise ValueError('No usable labelled fusion decisions in recording')
    pir_features = [c for c in train.columns if c.startswith('pir') and c != 'pirMeanSignalEnergy']
    thermal_features = [c for c in train.columns if c.startswith('thermal') and c != 'thermalTrack_id']
    features = pir_features if args.pir_only else pir_features + thermal_features
    xtrain, xtest = prepareFeatures(train, test, features)
    model = buildModels()[args.model]
    model.fit(xtrain, train['human'])
    score = confidenceScores(model, xtest)
    state = applyThreeState(score, test)
    out = pd.DataFrame({
        'time_s': test['fusionTime'].to_numpy(float),
        'confidence': score,
        'state': state,
        'human': test['human'].to_numpy(int),
    })
    out = out.sort_values('time_s', kind='stable').reset_index(drop=True)
    out['false_alarm'] = ((out['human'] == 0) & (out['state'] > 0)).astype(int)
    target = args.output if args.output else recording / 'review_predictions.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(f'Saved {len(out)} decision rows to {target}')
    print(f'Model: {args.model} | {"PIR only" if args.pir_only else "Fusion"} | PIR window: {args.window}s')
    print('NOTE: These are model predictions from the current offline feature pipeline.')


if __name__ == '__main__':
    main()
