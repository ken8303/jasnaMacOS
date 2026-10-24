#!/usr/bin/env python3
"""Bounded batch-2 MLX trial. Run each variant in a fresh process; no app defaults change."""
import argparse
import json
from pathlib import Path
import time
import resource
import numpy as np
from rfdetr_mlx_detector import RFDetrMLXDetector, mx
from rfdetr_mlx_experimental import ExperimentalRFDetrMLXDetector
from rfdetr_mlx_postprocess import select_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='NPZ containing normalized NHWC images')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'compiled', 'fp16', 'fp16-compiled'], required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('repeats must be between 1 and 10')
    root = Path(__file__).resolve().parents[1]
    archive = root / 'Models/MLXDetector/rfdetr-vr-v1.safetensors'
    images = np.load(args.inputs)['images']
    if images.shape[0] < 2 or images.shape[0] % 2:
        parser.error('an even number of images, at least two, is required')
    model = RFDetrMLXDetector(archive) if args.variant == 'baseline' else ExperimentalRFDetrMLXDetector(
        archive, precision='float16' if args.variant.startswith('fp16') else 'float32',
        compile_forward='compiled' in args.variant)
    mx.eval(model.weights)
    batches = [mx.array(images[i:i+2]) for i in range(0, len(images), 2)]
    mx.eval(batches)
    started = time.perf_counter()
    raw = model.predict_raw(batches[0])
    mx.eval(raw)
    warmup = time.perf_counter() - started
    print(f'{args.variant}: cold forward {warmup:.3f}s', flush=True)
    del raw
    timings = []
    for repeat in range(args.repeats):
        for batch in batches:
            started = time.perf_counter()
            raw = model.predict_raw(batch)
            mx.eval(raw)
            timings.append(time.perf_counter() - started)
            del raw
        print(f'{args.variant}: timed repeat {repeat + 1}/{args.repeats}', flush=True)
    prediction_arrays = {}
    counts = []
    for index, batch in enumerate(batches):
        raw = model.predict_raw(batch)
        values, boxes, masks, valid = select_predictions(raw['pred_boxes'], raw['pred_logits'],
            raw['pred_masks'], threshold=.15, max_select=64)
        # Save all top-64 candidates to match detections by box, not unstable rank order.
        values, boxes, masks, valid = map(np.asarray, (values, boxes, masks > 0, valid))
        if not all(np.isfinite(np.asarray(value)).all() for value in raw.values()):
            raise ValueError('non-finite model output')
        for offset in range(2):
            n = index * 2 + offset
            prediction_arrays[f'scores_{n}'] = values[offset]
            prediction_arrays[f'boxes_{n}'] = boxes[offset]
            prediction_arrays[f'masks_{n}'] = masks[offset]
            counts.append(int(valid[offset].sum()))
        del raw
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / (args.variant + '-predictions.npz'), **prediction_arrays)
    result = dict(variant=args.variant, device=mx.device_info()['device_name'], mlx=mx.__version__,
        images=len(images), batch_size=2, repeats=args.repeats, cold_forward_s=warmup,
        median_batch_s=float(np.median(timings)), mean_batch_s=float(np.mean(timings)),
        images_per_second=2 / float(np.mean(timings)), peak_mlx_bytes=mx.get_peak_memory(),
        peak_process_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        detected_counts=counts, timings_s=timings)
    (args.output / (args.variant + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
