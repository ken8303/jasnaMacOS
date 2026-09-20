# Synthetic output copying

Run from the repository root:

```sh
bash script/test_synthetic_output_copy.sh --gpu-written
```

Omit the flag for the original CPU-filled shared-buffer control. An optional
results directory follows the flag. Each run saves independent build/log files
and a JSON report. A failed build, GPU command, pixel check, or report write
returns nonzero; only the final PASS line establishes completion.

The GPU experiment creates RGBA float patterns with exactly representable values.
Each partner receives a fresh GPU write before CPU access. Eight warmup pairs
precede 64 measured pairs per size, with alternating fresh/reuse order. Every
pixel and checksum is checked against an independent CPU-generated reference.
Changing values exercise stale-buffer errors; no interpolation kernel changes,
motion estimation, media, models, or application modules are involved.

Report samples separate command creation/encoding/submission/wait, output
allocation/copy, CPU checksum reading, and total elapsed time. GPU timestamps
overlap submission/wait and must not be added to host timing. Printed component
medians need not sum to the median total. Initial reusable allocation, pipeline
setup, reference generation, validation, and destination release are excluded.
Results measure synchronous single-buffer operation, not concurrent GPU load.
The earlier CPU-filled control includes fresh-array destruction, so cross-mode
comparisons with that control are descriptive rather than a controlled A/B.

All raw measured samples are retained for reviewing dispersion and execution
order. Near-zero differences should not be treated as evidence of a speed gain.

## Four-output submission comparison

```sh
bash script/test_synthetic_output_copy.sh --grouped
```

Compares three modes: four submissions with four waits, four submissions queued
before waiting on the last, and one submission containing four dispatches.
All commands use one queue, and all four completion statuses must pass before
CPU reads. A failure in an earlier command cannot be hidden by a successful last
command. Submission/wait counts are validated and saved in each sample.
All paths allocate the same four independent shared outputs and copy all four
to fresh CPU arrays after all writes complete.
Each output has a different changing pattern, verified pixel-for-pixel against
the CPU reference. Every output is checksum-consumed inside the timing window.
Twelve warmup trials precede 60 measured trials per mode. All six mode orders
occur equally often; each mode occupies each position 20 times during measurement.
Paired differences are reported for queued−separate, grouped−queued, and
grouped−separate. Older two-mode reports used 64 pairs and are not pooled with
these results.
Timings are per four-output trial, not single-output latency; setup, validation
and array release remain excluded. This measures a synchronous generated workload,
not the performance of concurrent inference or video processing.
