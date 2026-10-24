#!/usr/bin/env python3
"""Compare detector trial predictions to the FP32 baseline (not ground truth)."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a, b):
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=-1)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=-1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def compare(baseline, candidate, image_count, threshold=.15):
    rows = []
    for index in range(image_count):
        active_a = baseline[f'scores_{index}'] > threshold
        active_b = candidate[f'scores_{index}'] > threshold
        a, b = baseline[f'boxes_{index}'][active_a], candidate[f'boxes_{index}'][active_b]
        ious = box_iou(a, b)
        left, right = linear_sum_assignment(-ious)
        matches = [(i, j) for i, j in zip(left, right) if ious[i, j] >= .5]
        masks_a = baseline[f'masks_{index}'][active_a]
        masks_b = candidate[f'masks_{index}'][active_b]
        mask_ious, box_ious, score_errors, box_errors = [], [], [], []
        for i, j in matches:
            union = np.logical_or(masks_a[i], masks_b[j]).sum()
            mask_ious.append(float(np.logical_and(masks_a[i], masks_b[j]).sum() / union) if union else 1.)
            box_ious.append(float(ious[i, j]))
            score_errors.append(float(abs(baseline[f'scores_{index}'][active_a][i] -
                                          candidate[f'scores_{index}'][active_b][j])))
            box_errors.append(float(np.max(abs(a[i] - b[j])) * 768))
        rows.append(dict(image=index, baseline_count=len(a), candidate_count=len(b),
            unmatched_baseline=len(a)-len(matches), unmatched_candidate=len(b)-len(matches),
            mask_ious=mask_ious, box_ious=box_ious, score_errors=score_errors,
            max_coordinate_errors_at_768px=box_errors))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    base_stats = json.loads((args.directory / 'baseline.json').read_text())
    report = {}
    with np.load(args.directory / 'baseline-predictions.npz') as baseline:
        for variant in ('compiled', 'fp16', 'fp16-compiled'):
            stats_path = args.directory / f'{variant}.json'
            if not stats_path.exists():
                continue
            stats = json.loads(stats_path.read_text())
            if stats['images'] != base_stats['images']:
                raise ValueError('image counts differ; use one input set for all variants')
            with np.load(args.directory / f'{variant}-predictions.npz') as candidate:
                rows = compare(baseline, candidate, stats['images'])
            mask_ious = [v for row in rows for v in row['mask_ious']]
            report[variant] = dict(
                throughput_ratio=stats['images_per_second']/base_stats['images_per_second'],
                inference_time_reduction_percent=100*(1-stats['mean_batch_s']/base_stats['mean_batch_s']),
                peak_mlx_reduction_percent=100*(1-stats['peak_mlx_bytes']/base_stats['peak_mlx_bytes']),
                unmatched_baseline=sum(row['unmatched_baseline'] for row in rows),
                unmatched_candidate=sum(row['unmatched_candidate'] for row in rows),
                minimum_mask_iou=min(mask_ious) if mask_ious else None,
                mean_mask_iou=float(np.mean(mask_ious)) if mask_ious else None,
                images=rows)
    (args.directory / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'images'}
                      for key, value in report.items()}, indent=2))


if __name__ == '__main__':
    main()
