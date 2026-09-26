# Bounded synthetic pipeline

```sh
bash script/test_synthetic_output_copy.sh --pipeline
```

Generated patterns only. Each trial processes 12 batches of four independent
outputs. One grouped command writes each batch. Depth 1, 2 or 4 limits the ring
of pending batches. GPU work can run while the CPU copies and consumes an older
batch. A slot cannot be reused until all four outputs have been consumed.

Six warmup trials precede twelve measured trials per depth and size. All six
depth orders occur equally often. Full pixel checks run before and after timings;
every timed output has its complete checksum compared with an exact CPU
reference. Every run must drain twelve batches, preserve slot identity and stay
within the selected pending-batch bound.

Results include whole-trial time, output throughput, submission-to-consumption
latency, and actual Metal pool allocation. CPU arrays are copied and released one
image at a time. `cpuCopyPayloadBytes` is one image's payload, not process RSS.
The larger depth requires proportionally more pool storage. Resource construction
is outside the stopwatch; fresh pools per trial can affect cache conditions.
These short trials include pipeline fill and drain, not just steady state.
Copy allocation/release, CPU reading, checksum checks and command submission are
timed. Pool setup and full per-pixel validation are excluded. Raw samples and
per-batch latencies remain in the JSON report.

This measures permitted overlap on one queue. It does not prove actual GPU/CPU
overlap duration or generalize to concurrent applications, models or video.
