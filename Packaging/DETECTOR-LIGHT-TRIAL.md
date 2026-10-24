# Experimental lighter MLX detector — 12 October 2026

Tested on Apple M4, 16 GB unified memory, bundled MLX 0.32.2. This changes execution and weight precision, not the trained architecture. The app exposes these modes under **Experimental options**; Stable FP32 remains the default.

## Measurements

Batch size 2, 768×768 input, 14 eye images from three saved short clips. Each variant ran in a fresh process, sequentially, with one cold forward followed by 21 timed batches (three passes). Timing includes completed model inference, excludes video decoding, normalization, polygon extraction and restoration. Peak MLX allocation includes warm-up, timing and prediction comparison; it is not total system memory.

| Variant | Eye images/s | Mean batch ms | Peak MLX GiB | Cold first forward s |
|---|---:|---:|---:|---:|
| baseline | 4.59 | 435.8 | 1.86 | 0.56 |
| compiled | 5.34 | 374.5 | 1.86 | 5.77 |
| fp16 | 5.74 | 348.6 | 1.38 | 1.11 |
| fp16-compiled | 6.49 | 308.2 | 1.27 | 3.27 |

FP16 + compilation delivered 41.4% more inference throughput, 29.3% less inference time and 31.5% less peak MLX allocation in this trial. These are detector-only measurements, not full restoration speedups. Cold compilation overhead matters for very short scans.

## Prediction checks

All variants returned finite raw outputs and matched all 21 baseline detections at confidence 0.15, using one-to-one box matching at IoU ≥ 0.5. Compiled FP32 binary masks matched exactly. FP16, with or without compilation, had mean mask IoU 99.868% and minimum 98.113% relative to baseline. This is agreement with the existing model, not measured ground-truth accuracy.

The production forward refactor was checked against the original Git HEAD implementation: raw logits, boxes and masks were exactly equal for batch sizes 1 and 2. The comparison helper passed reordered, missing and empty detection checks.

## Limits and next trial

The latest FCVR source was unavailable at its original path. Saved short clips provide limited scene diversity; repeated frames are correlated. This trial does not establish performance or recall on tiny mosaics, scene cuts, difficult lighting, or long runs. Keep non-default modes experimental until representative source clips and full-pipeline quality checks pass.

To repeat with an available SBS source, run from the repository (no Homebrew or new Python installation needed):

```bash
./script/test_detector_light.sh "/path/to/source.mp4" "/path/to/new-results-directory" 60 30
```

This samples both eyes at 60, 75 and 90 seconds, compares four variants sequentially and writes JSON results. It does not restore a video. Use a new results directory each time. Choose times within the video. Do not run another GPU restoration at the same time when measuring performance.

Raw measurements: `baseline.json`, `compiled.json`, `fp16.json`, `fp16-compiled.json`. Sample manifest: `samples.json`. Per-image comparison: `comparison.json`.
