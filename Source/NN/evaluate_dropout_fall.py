"""Fixed synthetic missing-zone stress check; not independent event validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_physics_fall import ROOT, PhysicsFallNet, load, class_windows, probability, metrics, assign_splits


def disconnect(raw, seed):
    """Remove a row/column/rectangle for 1-3 frames, retaining usable frames."""
    rng = np.random.default_rng(seed)
    grid = raw.copy().reshape(-1, 8, 8)
    for _ in range(3):
        first = int(rng.integers(len(grid)))
        last = min(len(grid), first + int(rng.integers(1, 4)))
        y, x = int(rng.integers(8)), int(rng.integers(8))
        mode = int(rng.integers(3))
        ys = slice(y, y + 1) if mode == 0 else slice(None) if mode == 1 else slice(y, min(8, y + 2))
        xs = slice(None) if mode == 0 else slice(x, x + 1) if mode == 1 else slice(x, min(8, x + 2))
        for t in range(first, last):
            before = grid[t].copy()
            grid[t, ys, xs] = -1
            if (grid[t] >= 0).sum() < 4:
                grid[t] = before
    return grid.reshape(-1, 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'NN/artifacts/physics_fall/dropout_stress.json')
    parser.add_argument('--weights', type=Path, default=ROOT/'NN/artifacts/physics_fall/weights.npz')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = PhysicsFallNet()
    weights = np.load(args.weights)
    model.load_state_dict({k: torch.from_numpy(weights[k]) for k in weights.files})
    model.eval()
    records = assign_splits(load())
    val = class_windows(records, 'validation')
    clean = [clip for k in range(3) for clip in val[k]]
    labels = np.array([k for k in range(3) for clip in val[k]])
    clean_p = probability(model, clean)
    corrupted = [disconnect(clip, 1701 + i*4 + trial) for i, clip in enumerate(clean) for trial in range(4)]
    targets = np.repeat(labels, 4)
    corrupted_p = probability(model, corrupted)
    nonfall = targets != 1
    clean_repeated = np.repeat(clean_p, 4, axis=0)
    report = dict(clean=metrics(labels, clean_p), corrupted=metrics(targets, corrupted_p),
                  nonfall_corrupted_sequences=int(nonfall.sum()),
                  nonfall_argmax_fall=int(((corrupted_p.argmax(1)==1)&nonfall).sum()),
                  nonfall_fall_score_ge_08=int(((corrupted_p[:,1]>=.8)&nonfall).sum()),
                  new_nonfall_argmax_fall=int(((corrupted_p.argmax(1)==1)&(clean_repeated.argmax(1)!=1)&nonfall).sum()),
                  mean_absolute_fall_score_change=float(np.abs(corrupted_p[:,1]-clean_repeated[:,1]).mean()),
                  seed=1701, corruptions_per_sequence=4,
                  limitation='Synthetic row/column/2x2 losses; correlated short held-out clips, not measured false alarms per hour or full-event recall.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
