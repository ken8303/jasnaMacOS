# Jasna Metal proof of concept

This branch requires macOS 27 and Xcode 27 (Swift 6.4). It intentionally no
longer targets older macOS releases.

This project tests the highest-risk operation in a native Apple Silicon port of
Jasna: the modulated deformable convolution used by BasicVSR++. It contains:

- a Metal FP32 kernel checked against a standalone CPU implementation;
- an FP16 kernel benchmarked at Jasna's real `1×128×64×64 → 1×64×64×64`
  propagation shape, with 16 deformable groups;
- a SIMD-group FP16 kernel with prepacked weights;
- conversion of sixteen supported static BasicVSR++ subgraphs into both Core ML
  `.mlpackage` and Metal ML `.mtlpackage` formats;
- a native Metal 4 runtime probe that loads and compiles a generated Metal ML
  package into an `MTL4MachineLearningPipelineState`;
- deterministic inputs and a checksum for repeatable performance comparisons.

Run:

```sh
./script/build_and_run.sh
```

Fast-cut a video from 12:00 to 13:00 without re-encoding:

```sh
./script/cut_video.sh /path/to/input.mov /path/to/clip.mov
```

The optional third and fourth arguments select another start and end time, for
example `./script/cut_video.sh input.mov clip.mov 05:30 06:15`. The script uses
FFmpeg stream copy, preserves all mapped streams and metadata, refuses to
overwrite an existing output, and reports the resulting duration. This is the
fastest method for an 8K source, but cuts align to nearby keyframes and may
start slightly early; use a re-encoded path when frame-exact boundaries matter.

Run only the correctness gate:

```sh
./script/build_and_run.sh --verify
```

Probe the converted feature extractor through Metal ML:

```sh
./script/build_and_run.sh --metal-ml-probe
```

Run the feature extractor or every supported Metal ML segment benchmark:

```sh
./script/build_and_run.sh --metal-ml-benchmark
./script/build_and_run.sh --metal-ml-suite
```

Verify Metal ML and a custom compute kernel sharing one buffer-backed tensor in
the same Metal 4 command buffer:

```sh
./script/build_and_run.sh --metal-ml-interop
```

Export and benchmark the experimental Core AI version of the real Jasna feature
extractor (the generated model and fixtures remain local under `Models/CoreAI`):

```sh
python3.13 -m venv .venv-coreai
.venv-coreai/bin/pip install coreai-torch torch torchvision mmengine
.venv-coreai/bin/python tools/convert_coreai_feature_extract.py \
  --jasna-source ../../work/jasna \
  --weights Models/SourceWeights/lada_mosaic_restoration_model_generic_v1.2.pth \
  --output Models/CoreAI/feature_extract.aimodel
./script/build_and_run.sh --core-ai-feature-extract
```

The Core AI spike is a measured experiment, not the restoration default. On an
Apple M4, its GPU-specialized output matched the PyTorch reference with maximum
error `3.05e-5`. A persistent `ComputeStream` and reusable zero-copy Metal input
and output buffers reduced a 30-frame stream to `1.221 ms/frame` median, versus
about `2.8 ms` when waiting after every inference. The existing Metal ML feature
extractor still measured `0.445 ms`, so Core AI remained about 2.7× slower even
under the streamed comparison. GPU-specialized loading measured about
`13–20 ms` once cached. The project therefore keeps Metal ML for the hot
restoration graph while retaining this probe for later Xcode beta comparisons.

Run a complete first propagation body in one Metal 4 command buffer, using the
real offset and backbone packages plus the checkpoint DCNv2 weights:

```sh
./script/build_and_run.sh --propagation-smoke
./script/build_and_run.sh --propagation-suite
./script/build_and_run.sh --reconstruct-frame
./script/build_and_run.sh --zero-copy-frame
./script/build_and_run.sh --zero-copy-frame-grouped
./script/build_and_run.sh --zero-copy-frame-staged
./script/build_and_run.sh --spynet-pair
./script/build_and_run.sh --frame-with-spynet
./script/build_and_run.sh --temporal-inputs
./script/build_and_run.sh --three-frame-recurrence
./script/build_and_run.sh --three-frame-first-pass
./script/build_and_run.sh --three-frame-four-pass
./script/build_and_run.sh --variable-clip 5
./script/build_and_run.sh --variable-clip 30
./script/build_and_run.sh --single-run-clip 30
./script/build_and_run.sh --plan-sbs-video 7680 4320 60 1
./script/build_and_run.sh --inspect-sbs-video /path/to/input.mov
./script/build_and_run.sh --transcode-sbs-30 /path/to/input.mov /path/to/output.mov
./script/build_and_run.sh --transcode-sbs-30-tiled /path/to/input.mov /path/to/output.mov
./script/build_and_run.sh --restore-sbs-video /path/to/input.mov /path/to/output.mov
./script/build_and_run.sh --restore-sbs-eye /path/to/input.mov left /path/to/left.mov
./script/build_and_run.sh --restore-eye-video /path/to/one-eye.mov /path/to/restored-eye.mov
./script/restore_vr_eye_segments.sh /path/to/input.mov left /path/to/restored-left.mov
./script/restore_vr_sbs.sh /path/to/input.mov /path/to/restored-vr.mp4
```

The side-by-side planner takes `width height source-fps duration-seconds`.
It always produces a constant 30 fps timeline, dropping or duplicating source
frames as necessary without changing the duration. The dimensions are not
restricted to one 8K container shape, so both `7680×4320` and shorter SBS
layouts such as `7680×2160` can be planned.

The AVFoundation video harness now reads real file dimensions, duration, and
nominal frame rate, validates the SBS plan, decodes sequentially with
Metal-compatible pixel buffers, selects the nearest source frame for every
exact `n/30` output timestamp, and writes HEVC. A generated 512×256 SBS smoke
video converted from 60 fps to 30 fps with 30 frames written and the output
metadata re-opened and validated. Existing output files are never overwritten.
The macOS 27 path uses `AVAssetReaderOutput.Provider.next()` and asynchronous
pixel-buffer receivers throughout. It no longer uses deprecated reader/writer
adaptors or polls encoder readiness with one-millisecond sleeps.
This first path is video-only and BGRA/SDR: audio copying, HDR/10-bit color
preservation, rotated tracks, and Metal restoration insertion remain explicit
follow-up work.

The tiled I/O path now converts decoded BGRA pixels to the model's planar FP16
RGB layout, reconstructs the frame with separable feather weights, propagates
the decoder's color attachments, and then encodes. Tile positions are evenly
distributed across each eye, so the 4320-pixel axis uses 42–43-pixel overlaps
instead of concentrating a 224-pixel overlap in the last row. Every tile stores
its actual four overlap widths. A 960×256 unit test round-trips every pixel
through overlapping FP16 tiles within one byte and proves the accumulated
weight is exactly one everywhere. The end-to-end 60→30 fps tiled smoke output
decoded identically to the direct output across all 30 frames (`inf` PSNR).

The first real restored-video command now replaces those identity tiles with
the fused Metal graph: bicubic flow inputs, bidirectional SPyNet, feature
extraction, all four recurrent propagation passes, reconstruction, and frame
residual. A generated one-second 512×256 SBS source produced two independent
eye tiles and thirty restored HEVC frames. The two graph submissions took
`792.518 ms` total and used a 22.50 MiB temporary FP16 tile cache. The output is
512×256, exactly 30 fps, 30 frames, and 1.000 second; its 38.53 dB PSNR versus
passthrough confirms that actual model output reached the encoder.

The restored-video command now supports arbitrary duration using one decoder
and one HEVC writer across sequential windows of up to thirty output frames.
A two-second smoke run produced 60 frames in two windows at exactly 30 fps and
2.000 seconds, with `1471.042 ms` total GPU graph time. A 31-frame run also
passed: its final one-frame window was padded internally to the graph's
three-frame minimum, while only the real frame was encoded, yielding exactly
31 frames and 1.033333 seconds.

For full SBS VR, `restore_vr_sbs.sh` now runs a restartable sequential-eye
workflow. It decodes the source directly but retains only one cropped eye's
30-frame window, restores and encodes the complete left-eye movie, releases
that work, then does the right eye. Finally it stacks the two restored eye
streams and copies the original audio into one SBS HEVC output. It does not
create lossy raw-eye intermediates before restoration. For an 8192×4096 input,
each active eye plan is 4096×4096 with 361 tiles and about 3.96 GiB of FP16
cache, instead of 722 tiles and about 7.93 GiB in one window. Total model work
is unchanged; this phase reduces peak memory and storage pressure rather than
runtime.

The orchestrator keeps `left-restored.mov`, `right-restored.mov`, stage markers,
logs, and independent resumable caches under `<output>.vr-work/`. A stopped run
continues the incomplete eye and skips an eye already marked complete. If a
compatible older full-SBS cache is still available, its left-eye tile prefix
can be reused explicitly:

```sh
JASNA_LEFT_WORK_DIR=/path/to/old.jasna-work \
  ./script/restore_vr_sbs.sh input_30fps.mp4 restored-vr.mp4
```

The final merge copies audio and ordinary container metadata. Injection of
Spherical Video `st3d`/`sv3d` atoms is not implemented yet, so players may need
the output manually identified as left-right SBS VR.

Sparse mosaic restoration follows VR Video Toolbox CE's pre-scan design. A
YOLO detector samples each physical eye every 0.1 seconds, separates distant
detections, and emits one-second tracked regions aligned with each 30-frame
processing window. This gives BasicVSR++ five times more temporal context than
the former 0.2-second clips, which reduces residual blocks and flicker on moving
mosaics without materially increasing the total number of restored frames. Set
`JASNA_REGION_DURATION=0.2` only for an old-behaviour comparison. The detector
writes a JSON manifest, suppresses fully contained crops, and applies temporal
non-maximum suppression to strongly overlapping duplicate tracks before
restoration. This prevents both inner rectangles and stacks of projected boxes
when one moving mosaic produces several detections. Nearby independent subjects
and tracks that continue outside the winning interval remain separate.
The default overlap cutoff is 0.45 (`JASNA_REGION_NMS_IOU`), based on the
remaining 0.487–0.497 overlaps in the v6 VR fixture. Large projected regions
also scale feathering from 12 up to 64 pixels instead of exposing a narrow,
fixed-width rectangular transition.
Detection defaults to confidence 0.15 with one second of temporal
padding so short or fast-moving mosaic appearances are less likely to be
missed. `JASNA_DETECT_CONFIDENCE` and `JASNA_TEMPORAL_PADDING` can override
those quality settings for controlled comparisons. The
Metal restorer then runs only the detected 256×256 model crops. Clean
one-second windows bypass the Jasna model. Following Jasna's VR180 path, the
sparse wrapper defaults to fisheye projection: each region is flattened before
restoration, only the restored delta is inverse-projected, and a feathered mask
places that delta onto the untouched source frame. Set
`JASNA_VR_PROJECTION=raw` only when comparing against the older flat-crop path.
The v11 manifest also carries compact 128×128 soft-mask keyframes from the detector's
segmentation output. Metal multiplies the precomputed fisheye composite alpha by
an interpolated mask for every output frame, restricting restored deltas to the
moving mosaic shape instead of using a one-second union or
exposing a rectangular crop. The mask expands by 10% of its resolution before
the soft edge is applied, covering large source mosaic blocks that extend
beyond the semantic contour. The higher resolution reduces visible alpha steps
when a large VR crop maps each former 64×64 mask pixel to roughly 25 source
pixels. `JASNA_MASK_EXPANSION` can tune expansion from `0.01` through `0.25`,
and `JASNA_MASK_SIZE` accepts powers of two from 32 through 256. Older manifests
without mask data retain the rectangular feather fallback.
The v12 restoration path selectively subdivides the largest oversized region in
each eye/window before inference. A region wider or taller than 768 pixels becomes
overlapping model crops with 96 pixels of context, capped at three crops per axis.
Each child retains a remapped static and temporal segmentation mask. This raises
the effective model resolution for the large moving VR regions that otherwise
stretch one 256×256 restoration over roughly 1,300–2,000 source pixels. Only one
region per window is expanded by default to contain the performance cost. Tune
`JASNA_LARGE_REGION_MAX_BLEND`, `JASNA_LARGE_REGION_OVERLAP`,
`JASNA_LARGE_REGION_SPLIT_LIMIT`, and `JASNA_LARGE_REGION_MAX_AXIS_CROPS`, or set
`JASNA_LARGE_REGION_MAX_BLEND=0` to retain the v11 single-crop behaviour.
The Metal compositor accumulates child restoration deltas and normalizes their
overlap weights before touching the frame. A sequential overwrite experiment
exposed a cross-shaped 2×2 grid and was rejected; normalized accumulation removes
that internal seam and fails closed if a grouped texture composite is unavailable.
The v13 path additionally expands only the selected oversized region's parent mask
before subdivision. Its parent blend envelope grows by the same bounded amount so
the existing rectangular feather cannot clip that new coverage. The default 5%
growth and 2.5% soft transition cover residual large mosaic blocks outside the
detector contour without broadening every small restoration. Tune
`JASNA_LARGE_REGION_MASK_GROWTH` and
`JASNA_LARGE_REGION_MASK_FEATHER`, or set growth to `0` for the v12 mask behaviour.
Grid dimensions remain based on the original detected envelope, preventing mask
growth near a threshold from unexpectedly changing a 2×2 restoration into 3×3.
The v14 path adds a square-distance halo around that semantic mask to target the
axis-aligned corners left by large mosaic blocks, while a half-strength adjacent
keyframe union protects quickly moving mask edges. The defaults add 4% block
coverage and one temporal keyframe on each side without adding model crops. Tune
`JASNA_LARGE_REGION_BLOCK_GROWTH` and
`JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS`; set either to `0` to disable that part.
The v15 path adds one 576-pixel detail crop centred on the lower boundary of the
selected semantic mask. It joins the same normalized compositor group as the 2x2
base grid, increasing sampling density only where the large VR block ring remained.
This adds one model crop per selected oversized region instead of the five extra
crops required by a full 3x3 grid. Tune `JASNA_LARGE_REGION_DETAIL_CROPS` from
`0` through `2` and `JASNA_LARGE_REGION_DETAIL_DIMENSION` from 256 pixels up to
the configured large-region maximum.
Metal 4/MPSGraph allocations on the macOS 27 beta are released reliably only when
the restoration process exits. The macOS 27 asynchronous AVFoundation receiver
can also retain queued 8K BGRA frames until an HEVC part finishes. Direct SBS
restoration therefore defaults to one temporal window per subprocess, writes a
validated HEVC part, exits to release `IOAccelerator`, `IOSurface`, and encoder
buffers, and concatenates the parts without another video encode. Set
`JASNA_METAL_WINDOWS_PER_PROCESS` from 1 through 30 only to make an explicit
peak-memory/startup tradeoff. Completed larger parts are validated and reused if
the limit is lowered while resuming an interrupted run.
Each work directory also has its own process lock. A second command targeting the
same resume data fails clearly, while unrelated restorations are left running.
Every isolated restoration subprocess logs `Runtime memory: peak resident` at
exit. This uses Darwin's per-process high-water mark and makes six- versus
twelve-window memory comparisons visible in the persistent restoration log.
Set `JASNA_LOG_PEAK_MEMORY=0` only when this telemetry is not wanted. The former
12-window default measured 5.57–5.82 GiB before the macOS 27 receiver migration,
but the new receiver allowed retained encoder surfaces to grow beyond 32 GiB in
a long 8K run. One-window isolation is the safe default until that beta behavior
is fixed or a bounded writer handoff is available.
The current sparse VR path decodes and encodes 8-bit BGRA/SDR. A Main 10 or HDR
source therefore does not retain its original bit depth or HDR transfer
characteristics; do not use this path when HDR preservation is required.
Detector setup is isolated from Swift:

```sh
./script/setup_mosaic_detector.sh
```

The setup downloads the public 6 MB
`lada_vr_mosaic_detection_model_v2_fast.pt` model used by VR Video Toolbox CE
and installs Ultralytics in `.venv-mosaic`. Neither environment nor model is
tracked by Git. Test the left eye with:

```sh
./script/restore_vr_eye_sparse.sh \
  /path/to/input_30fps.mp4 left /path/to/restored-left.mov
```

Detection manifests, source segments, restored windows, caches, and the log
remain beside the requested output, so interrupted work is restartable. A real
4096×4096 one-second left-eye fisheye proof detected three regions. The Metal
model used 1.31 seconds of GPU time for all three crops; compositing and hardware
HEVC encoding brought the post-build work to roughly six seconds. Its output was
validated as exactly 30 frames at 30 fps. Projection mode is included in the
resume-cache key, so an older raw crop can never be reused for a fisheye run.
The key also fingerprints source-segment and model-file paths, sizes, and
modification times, preventing stale crops after an input or model replacement.

Persistent crop caches are flushed every five completed regions and at the end
of every window. After an unexpected restart, at most four small regions are
recomputed. Set `JASNA_REGION_CHECKPOINT_INTERVAL=1` to force per-region
durability, at the cost of more disk synchronization.

Each sparse window also reports its model-frame count and separate timings for
crop extraction, graph wall time, GPU time, cache writes, compositing/writer
submission, and encoder finish. These end-to-end phases determine whether the
next optimization should pipeline CPU preparation, reduce cache traffic, or
remain focused on the Metal graph without changing restoration quality.

Sparse output keeps the one-second recurrence boundaries but, by default, feeds
up to 120 consecutive windows to one HEVC writer. This produces one restored
file per two minutes and reduces hardware-encoder startup and drain work while
preserving restartability: model caches remain per-window, so an interrupted
writer can rebuild its current output without rerunning completed model crops.
Existing one- and five-window outputs are detected and reused. Set
`JASNA_ENCODER_WINDOWS_PER_SEGMENT=5` for the earlier five-second behavior,
`JASNA_ENCODER_WINDOWS_PER_SEGMENT=1` for one file per recurrence window, or
choose another positive segment size. A three-window 4096×4096 fixture reduced
encoder-finish time from 1.431 seconds to 0.497 seconds and produced one
validated 90-frame, 30 fps segment. The 30-second production log measured about
0.48 seconds per encoder drain; grouping a two-minute segment avoids up to 23
extra drains, or roughly 11 seconds per eye on that workload.

Metal ML crop execution is serialized. The first 30-frame crop builds the
retained graph and every later crop reuses it; attempting to construct two
first-use graphs concurrently proved unstable in the macOS 27 beta runtime.
`JASNA_REGION_CONCURRENCY` is therefore ignored for production restoration.

The optimized production path retains its first complete 30-frame Metal graph
for the lifetime of the eye-restoration process. Later crops reuse the same
tensors, buffers, argument tables, and residency set under a lock. Sequential
Metal ML dispatches also share scratch heaps per pipeline/level instead of
allocating hundreds of identical heaps per crop. A 4096×4096 one-second proof
measured about 1.10 seconds for the initial graph build and execution, then
about 0.45 seconds per reused 30-frame crop including roughly 0.38 seconds of
GPU work. Decoded frame hashes matched the pre-cache output exactly.

Restoration uses batch 1 by default. Set `JASNA_MODEL_BATCH=2` to experiment with
grouping two mosaic crops of the same temporal length in one retained graph when
`Models/MetalMLBatch2` is present. Incompatible or odd final crops stay on batch
1. The first batch failure retries both crops individually and disables batch 2
for the remainder of that process, preventing a bad graph from imposing another
minute-long timeout on every later crop. Batch-2 package initialization can take
roughly three minutes in a fresh process under Xcode 27 beta.

The retained graph clears its unpadded main buffers only on first use because
every later destination is fully overwritten. SPyNet's padded-row tensors are
still cleared for every crop; skipping those clears changed output and was
rejected. On a five-region production fixture, the accepted change removed
about 26 ms of non-GPU work per reused crop. The decoded 30-frame output was
byte-identical, and the full PyTorch oracle remained at 70.49 dB PSNR.

Long sparse eye restorations also submit every pending 30–120 second physical
segment to one sequential app process. This keeps that retained graph alive
across segment boundaries while each segment continues to use its own output,
resume cache, and completion marker. A two-job production fixture reduced the
second job's first-crop wall time from 786 ms to 101 ms; both decoded 30-frame
outputs had identical frame hashes. Restarting still skips validated windows
and resumes an interrupted crop from its segment-specific persistent cache.
The 30-second SBS test coordinates both eyes through the same batch as well.
On an 8192×4096 end-to-end fixture, the right eye's first crop reused the graph
and fell from 1.32 seconds to 456 ms. A second invocation skipped both completed
eyes and the final SBS output without running the model or encoder again.

Sparse fisheye compositing runs the delta sampling and feather blending on
Metal. Its default zero-copy path wraps the decoder and encoder pixel buffers
as Metal textures, avoiding two 64 MiB CPU frame copies for every 4096×4096
eye frame. Two independent output frames are prepared in parallel on Macs with
at least 16 GB of memory, then submitted to AVFoundation in presentation order.
Set `JASNA_COMPOSITE_CONCURRENCY=1` for the lowest-memory path; values above two
are capped. `JASNA_METAL_TEXTURE_COMPOSITOR=0` selects the Metal buffer-copy
fallback, while `JASNA_METAL_COMPOSITOR=0` selects the CPU fallback.

In an alternating warmed comparison, zero-copy reduced steady compositor time
from 20.2 to 8.3 ms/frame and reduced user/system CPU time from 0.93/0.95 to
0.78/0.74 seconds for a one-second proof. Whole-job time remained about 4.2
seconds because model inference and HEVC finalization dominate and overlap the
saved work. A raw-frame cross-check differed in only 8 of 67,108,864 bytes,
each by 1/255. Set `JASNA_VERIFY_METAL_COMPOSITOR=1` to repeat that first-frame
diagnostic.

Run an end-to-end 30-second test of both eyes and rebuild an SBS preview with:

```sh
./script/test_vr_sparse_30s.sh \
  /path/to/input-sbs.mp4 /path/to/restored-vr-test.mov 00:12:00
```

After validating quality, restore the complete SBS source with the production
wrapper:

```sh
./script/restore_vr_sparse_sbs.sh \
  /path/to/input-sbs.mp4 /path/to/restored-full.mp4
```

It does not apply the test harness's 30-second cut. It uses persistent
120-second source/restoration segments, fisheye sparse regions, direct SBS
output, and the stable batch-1 model path by default. The launcher limits each
Metal process to twelve temporal windows so beta-runtime allocations are released
regularly. Set `JASNA_MODEL_BATCH=2` only for an explicit batch-2 experiment. A
restart reuses completed source segments, region manifests, restored windows,
and validated SBS segment files. Existing and newly written final SBS outputs
must also match the source dimensions, HEVC codec, 30 fps frame count, expected
duration, and a real first-frame decode before they are accepted.

The start time is optional and defaults to the beginning. The script prepares
an exact 30 fps test clip, restores the left and right eyes sequentially, copies
the test clip's audio, and keeps all intermediate files and logs beside the
output. Repeating the same command resumes incomplete work and skips validated
eye outputs.

Limit a longer quality and stability test to exactly five minutes with:

```sh
JASNA_TEST_SECONDS=300 ./script/test_vr_sparse_30s.sh \
  /path/to/input-sbs.mp4 /path/to/restored-vr-5m.mov 00:12:00
```

The test duration is capped at 300 seconds. A five-minute run uses physical and
restored eye segments of 120, 120, and 60 seconds while model recurrence and
resume checkpoints remain one second apart.

The test wrapper reuses a fresh release executable when available, otherwise it
builds once and shares that executable across both eyes. When the source is
already 30 fps and the requested start is zero, it copies the requested test
interval without an unnecessary 8K re-encode. VideoToolbox
speed-priority mode is enabled by default; set `JASNA_FAST_ENCODE=0` to compare
its output with the slower quality-priority encoder. Set
`JASNA_FAST_SOURCE_COPY=0` to force regeneration of the 30 fps test source.
Paired eye segments now restore directly into one 8192×4096 HEVC stream. The
Metal compositor assembles both eye surfaces and applies their fisheye deltas in
place, so the final join copies video packets and audio instead of encoding two
eye movies and then encoding their SBS stack again. Each direct SBS segment is
still independently restartable and remains in the persistent work directory.
Set `JASNA_DIRECT_SBS_OUTPUT=0` to compare with the legacy three-encode path.
Sparse mosaic scans decode HEVC sequentially and infer two sampled frames at a
time. Set `JASNA_DETECT_BATCH_SIZE=1` to minimize memory, or
`JASNA_DETECT_DECODE_MODE=seek` to compare with the former random-seek path.
Detector logs separate neural inference from decode/batching time. On a
60-second 4096×4096 M4 fixture, the accepted 2048-pixel, 0.1-second-stride,
batch-2 scan took 66.9 seconds: 62.6 seconds inference and 4.3 seconds decode.
Batch 4 regressed to 75.0 seconds; an alternating repeat kept batch 2 ahead of
batch 1 by 67.95 to 70.90 seconds. Reducing inference size to 1792 or 1536, or
sampling every 0.2 seconds, was rejected because each missed five or six
restoration regions from the quality baseline. A shared decode feeding two eye
encoders improved a 10-second preparation fixture by only 5.6%, so the simpler
restartable per-eye preparation remains the production path.
An opt-in Core ML export of the same segmentation checkpoint was also rejected:
with a fixed 2048-pixel batch-1 input it took 73.0 seconds (69.3 seconds of
inference), about 9% slower than the accepted PyTorch/MPS path. It produced the
same total of 128 restoration regions, but FP16 box differences changed temporal
tracking and left two baseline blend regions without a same-window overlap. The
scanner now declares the segmentation task explicitly so exported backends can
be evaluated without silently interpreting mask coefficients as detections, but
the `.pt` model remains the production default.
Fisheye sampling coordinates and interpolation weights are calculated once per
mosaic region and reused across its active frames.
Inactive region/frame cache slots are sparse file holes and are skipped during
compositing, reducing physical cache I/O without changing resumable offsets.
A conservative contained-track merge was rejected on the 30-second production
fixture: it preserved all original blend coverage but reduced 346 crops to only
343 and saved just 65 of 10,047 model frames (0.65%). Reusing the extracted
original crops during direct compositing was pixel-identical but increased the
one-window runtime by 4–8% from retained-memory pressure. Moving compositor work
ahead of HEVC backpressure was also neutral (1.33 versus 1.32 seconds for the
one-second fixture), so neither experiment is in the production path.

When the supporter-only Jasna SD1.5 checkpoint is unavailable, the public
DeepMosaics BVDNet checkpoint can be tested on a persistent 30-second left-eye
clip:

```sh
./script/test_deepmosaics_left.sh \
  /path/to/input_30fps.mp4 /path/to/restored-left-30s.mov
```

The runner automatically uses PyTorch MPS when the M4 GPU is accessible and
falls back to CPU otherwise. It batches all detected regions in each one-second
window, stores every region result before encoding, validates every 30-frame
HEVC window, and resumes completed work. On an M4, a three-region window fell
from about 87 seconds on the original sequential CPU path to 38.2 seconds with
MPS batching. GPU and CPU outputs had a 99th-percentile difference of 1/255.
The 30-second test completed as 900 validated 4096x4096 frames; its public
checkpoint produces plausible smoothing rather than recovery of true hidden
detail.

For the lower-risk physical-file workflow, test one eye first with
`restore_vr_eye_segments.sh`. It decodes and crops the selected SBS half into
real, persistent 30 fps HEVC source files of 60 seconds each, restores each
file as independently validated one-second model-window movies, and joins the
completed windows and restored segments without another encode. Every source
segment, one-second restored window, restored segment, completion marker, model
cache, and log stays beside the requested output. Re-running the same command
skips completed one-second windows and resumes only the incomplete window:

```sh
./script/restore_vr_eye_segments.sh \
  /path/to/input_30fps.mp4 left /path/to/restored-left.mov
```

The segment length defaults to 120 seconds. Use 60-second physical files when
you prefer smaller restart units:

```sh
JASNA_SEGMENT_SECONDS=60 ./script/restore_vr_eye_segments.sh \
  /path/to/input_30fps.mp4 left /path/to/restored-left.mov
```

After the left-eye result has been inspected, run the same command with
`right` and a different output name. Combining those two eye outputs back into
SBS is intentionally left for the next validation step.

Restoration commands use an optimized Swift release binary. Metal ML pipeline
states, the compiled Metal shader library, and DCNv2 weight buffers are cached
inside the process and reused across tiles. A four-tile 768×256 production
smoke test reduced warm per-tile wall time from about six seconds to about one
second while retaining approximately 0.37-second GPU graph time. The same test
encoded 30 composited frames in about 0.07 seconds. The encoder disables frame
reordering so independently completed windows concatenate at exactly 30 fps;
a 33-frame partial-window test produced exactly 1.100000 seconds and resumed by
skipping both validated window files.

Each window retains its decoded BGRA frames, writes restored FP16 tiles to
temporary storage, composites only one full frame at a time, and removes the
window cache before decoding the next one. At 7680×4320 the peak tile cache is
about 7.47 GiB rather than growing with video duration, and temporary-disk
headroom is checked before every window. The remaining quality limitation is
the hard recurrence reset every thirty frames; temporal window overlap is the
next refinement.

Some Apple Silicon VideoToolbox configurations cannot create a decoder for
`8192×4096`, HEVC Main 10 Level 6.1 at 59.94 fps (`-12906`, decoder not found),
even though FFmpeg's software HEVC decoder can read it. Prepare that source
before restoration while preserving its full resolution and Main 10 format:

```sh
./script/prepare_8k_30fps.sh input.mp4 input_30fps.mp4
./script/build_and_run.sh --restore-sbs-video input_30fps.mp4 restored.mov
```

Preparation decodes HEVC in software, selects 30 fps, copies audio, and uses
Apple VideoToolbox to encode 40 Mbit/s HEVC Main 10. A one-second 8192×4096
sample prepared this way decoded and re-encoded successfully through the
AVFoundation video path. The original 8192×4096 spatial resolution is not
reduced.

Restoration runs keep their diagnostics beside the requested output. For an
output named `restored.mov`, the launcher creates:

- `restored.jasna.log` with all build output, timestamps, window progress, tile
  progress, GPU time, errors, and later sessions appended;
- `restored.jasna-work/` as the persistent tile-cache root.

The work directory is not under macOS temporary storage, so a system restart
does not erase an interrupted window cache. Successfully encoded window caches
are removed to reclaim space; a cache involved in a handled failure is
preserved and its exact path is written to the log. Re-running the same command
automatically finds the most complete cache for each window, trims all frame
files to their common completed-tile boundary, and continues at the next tile.
Cache files are synchronized and checkpointed every eight tiles. An unfinished
output movie is moved beside the output with an `interrupted-TIMESTAMP` name
before a fresh writer starts, so it is not silently overwritten. Metal graph
objects are scoped to a per-tile autorelease pool to prevent IOSurface buildup
during hundreds of 8K tiles. Because an unfinished HEVC writer cannot itself be
continued, windows encoded before the interrupted window are rendered again;
the expensive tiles in the preserved interrupted window are reused.

If a real-content tile overflows FP16 in the 30-frame recurrence, restoration
logs the exact eye, coordinates, branch, frame, and element, then retries that
tile with balanced 10-, 5-, and 3-frame chunks. A tile that remains unstable is
restored as independent zero-motion frame triplets. If every Metal recovery mode
fails, production restoration now stops rather than silently retaining the
original mosaic. Set `JASNA_ALLOW_PASSTHROUGH=1` only to create an explicitly
logged, known-degraded diagnostic output. Diagnose one tile without creating an
output movie with:

```sh
./script/build_and_run.sh --diagnose-sbs-tile input_30fps.mp4 90
```

Inspect the real four-pass temporal traversal, validate every converted package
boundary, and allocate the complete buffer-backed clip arena:

```sh
./script/build_and_run.sh --schedule 5
./script/build_and_run.sh --validate-package-graph
./script/build_and_run.sh --allocate-frame-graph 5
./script/build_and_run.sh --validate-deform-weights
./script/build_and_run.sh --benchmark-real-weights
```

## Measured result

On a 10-GPU-core Apple M4 with Xcode 27 beta 4, 20-sample runs across all four
checkpoint directions measured the shared-coordinate gather plus fused
SIMD-group-GEMM FP16 deformable convolution at 0.534–0.550 ms median. The tiled
scalar reduction measured 1.178–1.186 ms, the first SIMD version about 9.1 ms,
and the direct baseline about 16.2 ms. The Metal-4-compatible path is about 54%
faster than tiled and 29–30× faster than the direct kernel at the median. Its
maximum FP16 difference from the baseline was `0.000977`, and its FP32
implementation passed the CPU oracle with a maximum absolute error of about
`3e-8`.

The converted feature extractor executes in about 0.42 ms median. The offset,
propagation-backbone, reconstruction, and six split SPyNet convolution packages
also execute successfully through Metal ML. The custom SPyNet border-warp and
flow-upsample kernel matches its CPU oracle exactly for the deterministic test.
On macOS 27, these tests use `MTLTensor` instances backed by ordinary
`MTLBuffer` storage with both compute and machine-learning usage, proving that
the custom kernels and Metal ML stages can share tensors without CPU copies.
The single-timeline interop test runs the feature extractor, applies a custom
compute operation after an explicit ML-to-dispatch barrier, and checks all
262,144 output values. With deterministic nonzero input, the model output
reached magnitude 2.47 and the post-compute comparison stayed within one FP16
rounding step (`0.000977`).

The temporal scheduler reproduces Jasna's `backward_1`, `forward_1`,
`backward_2`, and `forward_2` passes, including flow indices, second-order
history, and 128/192/256/320-channel backbone inputs. Runtime reflection
validates all 16 supported packages and 32 tensor bindings against this graph.
The real buffer-backed arena uses 14.50 MiB for five frames and 174.34 MiB for
60 frames (478 persistent tensor slots), all with shared compute and Metal ML
usage.

`tools/export_deform_weights.py` extracts the four learned deformable-
convolution parameter sets from the public checkpoint and writes the packed
FP16 `[input channel, kernel element, output channel]` layout consumed directly
by the Metal kernels. Each direction is 147,584 bytes including bias. All four
load into Metal buffers and benchmark at 0.534–0.550 ms median through the
shared-coordinate gather plus fused SIMD-group GEMM, with a maximum delta of
`0.000977` from the direct FP16 implementation. The custom
offset/mask stage implements Jasna's `10*tanh`, interleaved flipped-flow add,
and sigmoid mask and passes its CPU oracle with maximum error `0.00195`.

The integrated `backward_1` propagation test now records the actual hybrid
sequence in one Metal 4 command buffer: Metal ML offset prediction, custom
offset/mask transform, checkpoint-weight tiled DCNv2, backbone-input assembly,
Metal ML propagation backbone, and the residual add. Explicit `MTLResidencySet`
tracking keeps every raw GPU-address buffer resident alongside the buffer-backed
Metal ML tensors. The recorder is generalized across the real 128/192/256/320-
channel backbone widths. On the same M4, `backward_1`, `forward_1`, `backward_2`,
and `forward_2` measured `3.697`, `3.726`, `3.593`, and `3.621 ms`, respectively,
for a four-pass total of `14.637 ms`. Every branch checked all 262,144 output
elements and reproduced its full result with zero error across repeated runs.

This proves the hybrid architecture is viable; it is not yet a complete video
restoration application. The earlier 4.7 ms figure covered DCNv2 alone across
four passes; the measured full hybrid propagation bodies total 14.637 ms. The
complete-frame smoke test now concatenates the spatial feature and all four
propagation results in Jasna's original order, runs the real reconstruction and
upsampling package, and adds the input-frame residual. It checks all 196,608
RGB values with zero repeat error and `0.000488` maximum residual-add error.
Feature extraction, reconstruction, upsampling, and the residual measured
`1.430 ms`; the propagation plus reconstruction estimate was `17.984 ms` on
that run.

The zero-copy frame graph removes that CPU staging and records feature
extraction, all four propagation bodies, reconstruction, and the frame residual
in one Metal 4 command buffer backed by a 55.44 MiB residency set. The original
interleaved schedule measured 19.780 ms median. Grouping ready offset networks
reduced it to 17.435 ms; additionally staging the four DCNv2 alignments before
the dependent backbones produced 15.939–16.785 ms medians. The latest seven-run
sample measured 15.939 ms median (15.688–16.068 ms), or 62.7 theoretical FPS,
with the same output checksum and zero repeat error. Fusing residual adds into
the next backbone assembly was tested but regressed to 17.326 ms, so the staged
schedule remains preferred.

The SPyNet-fed staged graph replaces its synthetic first-order fields with the
checkpoint-validated backward and forward flows. Two repeated measurements put
the combined SPyNet plus frame graph at 21.240–23.081 ms (43.3–47.1 theoretical
FPS), with zero repeat error and the same `55.216187` frame checksum. The frame
portion alone measured 18.924–20.603 ms. This is the more realistic current
baseline: learned offsets reduce DCNv2 cache locality, so the earlier isolated
17.975 ms estimate was optimistic.

This remains a deterministic two-frame scheduling probe, not a complete
temporal clip. It currently transfers the learned flow arrays between two
command buffers, and a two-frame pair has no second-order predecessor. The
full-size temporal-preparation kernels now implement Jasna's real
`flow_n1 + warp(previous_flow, flow_n1)` composition, zero-padded feature
warps, 196-channel offset condition, and 128-channel DCNv2 input. They pass a
CPU oracle exactly on the deterministic FP16 test. Preparing both learned-flow
directions measured 0.284–0.289 ms median in the user's repeated full-size
runs, with zero repeat error and an exactly zero second-order field on the first
recurrence step.

The three-frame recurrence probe completes that binding for the first real
BasicVSR++ branch. It extracts three independent frame features, initializes
`backward_1`, then executes first- and second-order alignment through the real
offset package, checkpoint DCNv2 weights, and three dependent backbone calls in
one Metal 4 command buffer. With two distinct adjacent SPyNet fields, it
measured 9.548 ms median in the user's run, with a nonzero 1.9150 second-order flow maximum, zero
repeat error, and a stable `18.276567` checksum across all 262,144 final feature
values. The adjacent learned-flow checksums are `-15.971924` and `-2.434631`;
the second pair shares the first pair's middle frame. The recurrence timing
excludes the separately measured SPyNet graph.

The staged three-frame first-pass probe adds the dependent `forward_1`
traversal. Every forward backbone input contains the current spatial feature,
the corresponding persistent `backward_1` frame feature, and its own aligned
feature. In the user's run it measured 8.915 ms for `backward_1` and 8.488 ms
for `forward_1`, or 17.403 ms staged, with zero repeat error and stable
`18.276567` / `33.585654` output checksums. This deliberately uses two command
buffers while validating the branch boundary; `forward_1` reuses the three
spatial tensors extracted by `backward_1`.

The four-pass probe adds `backward_2` and `forward_2`, preserving the per-frame
prefix order and real 128/192/256/320-channel backbone widths. Its fused path
records three bicubic flow-input downscales, two adjacent bidirectional SPyNet
pairs (24 Metal ML residual-block calls), three feature extractions, all four
recurrent branches, twelve backbone calls, eight offset/DCNv2 alignments, three
reconstruction/upsampling networks, the input-frame residuals, and every
dependency barrier in one Metal 4 command buffer. Two 20-sample runs measured
30.349 ms and 31.137 ms medians. Their P10–P90 intervals were 28.980–31.200 ms
and 29.554–31.796 ms, with 0.802 ms and 0.885 ms standard deviations. The graph
starts with three 256×256 input frames and ends with all three restored 256×256
RGB frames.

All four fused flows, all twelve fused propagation tensors, and all three
restored frames matched their separately submitted oracles bit-for-bit and had
zero repeat error. The residual add had `0.000488` maximum error, and frame
checksums were `389.891747`, `388.891373`, and `387.192463`.

The complete Metal output is also checked against an independent CPU execution
of the original PyTorch BasicVSR++ generator, rather than only against staged
Metal implementations. Across all 589,824 restored values, the measured maximum
absolute error was `0.008633`, mean error `0.000203`, P99 error `0.001040`, and
RMSE `0.000299`, for 70.49 dB PSNR. The command fails unless maximum error is at
most `0.02`, mean error at most `0.0005`, P99 error at most `0.002`, and PSNR at
least 60 dB.

The same executor is no longer restricted to three frames. A five-frame run
creates four adjacent bidirectional SPyNet pairs, follows the generated
backward/forward traversal for every branch, uses second-order history from the
third position onward, and reconstructs all five outputs. Two 20-sample runs
measured 56.349 ms and 58.364 ms medians, with 54.935–61.335 ms and
55.938–62.518 ms P10–P90 intervals. That is 85.7–88.7 restored frames/s within
the measured clip and uses 14.50 MiB of persistent clip tensors. All flow and
repeated-output errors were zero. Against 983,040
independent PyTorch output values, maximum error was `0.004133`, mean error
`0.000183`, P99 error `0.000773`, and PSNR 72.08 dB.

A production-length 30-frame graph also fits in one Metal 4 command buffer. Two
20-sample runs measured 374.788 ms and 381.354 ms medians, with
370.827–382.505 ms and 377.944–383.127 ms P10–P90 intervals. This is
78.7–80.0 restored frames/s within the clip, with 87.16 MiB of persistent clip
tensors. Flow, propagation, and restored-frame repeat errors were all zero.
Across 5,898,240 independent PyTorch values, maximum
error was `0.022371`, mean error `0.000373`, P99 error `0.002583`, RMSE
`0.000647`, and PSNR 63.79 dB. The maximum occurred in frame 1 rather than at
the end of the recurrent sequence. For clips longer than five frames the gate
allows maximum/P99 errors of `0.03` / `0.003`, while retaining the `0.0005` mean
error and 60 dB PSNR requirements. This explicitly accounts for repeated FP16
rounding without weakening the distribution-wide accuracy checks.

The production target is now 8K side-by-side input with constant 30 fps output.
Because the converted model is fixed at 256×256, the full-resolution path uses
32-pixel-overlapped tiles and plans each eye independently; no model tile can
cross the stereo boundary. A `7680×4320` frame produces 340 tiles per eye, or
680 tiles total, and one second of 30 fps output maps to 680 executions of the
validated 30-frame temporal graph. The planner covers the right and bottom
edges exactly and reports the 126.56 MiB BGRA output-frame footprint. It is now
covered by deterministic tests for eye isolation, edge coverage, 60→30 frame
selection, slower-source frame duplication, and temporal-window counts.

This is the scheduling and memory contract for the upcoming video reader,
tile blending, and encoder. The fused executor now has a production mode that
submits one graph execution with no benchmark warmups or repeats. It does not
yet claim end-to-end 8K file conversion: pixel-buffer conversion and a 30 fps
`AVAssetWriter` path still need to be connected, and independent 30-frame
windows will need a small temporal overlap to hide recurrence resets at clip
boundaries.

On the same M4, the first production-mode submission restored thirty 256×256
frames in `370.904 ms`. At 680 overlapping tiles, that is approximately 252
seconds of graph time for one second of `7680×4320` / 30 fps output, before
decode, blending, and encode. Therefore “30 fps output” currently means the
encoded timeline rate, not real-time processing. The present architecture is
an offline converter target; reaching real-time 8K would require a fundamental
throughput change rather than only video-I/O tuning.

The bidirectional SPyNet probe now builds normalized 2/4/8/16/32/64 pyramids,
uses padded row strides required by Metal ML at the small levels, runs twelve
real checkpoint convolution blocks, and performs custom border warp, flow
upsampling, and residual addition. Both directions together measured 2.202 ms
median (2.189–2.248 ms over seven samples), with zero repeat error and 0.0122
maximum difference from a PyTorch checkpoint oracle generated by
`tools/export_spynet_oracle.py`. This remains useful as an isolated component
measurement, but the learned-flow frame graph above supersedes simply adding
it to the synthetic-flow frame time.

The 64×64 SPyNet input is faithful to Jasna rather than a reduced-resolution
shortcut: the original `BasicVSRPlusPlusNet.forward` bicubic-downsamples each
256×256 LQ frame by 0.25 before calling `compute_flow`. The 256×256 path is used
by `feat_extract`; flow is intentionally computed from the 64×64 copy. The
three-frame probes now perform that exact quarter-scale bicubic operation on
the same input frames used by feature extraction before evaluating both
adjacent bidirectional SPyNet pairs. The Metal downsampler matches an independent
CPU implementation with zero FP16 difference; the older deterministic 64×64
inputs remain only for the checkpoint-oracle SPyNet unit probe.
For the corrected synthetic three-frame clip, the two separately submitted
bidirectional SPyNet oracle pairs measured 2.415 ms and 2.411 ms median in the
combined-graph run. Their backward
checksums were `-21.818604` / `4.351471`, and their forward checksums were
`0.474731` / `-8.876831`; the fused graph reproduced all four exactly.

The specialized tiled DCNv2 kernel now applies the shape's stride and dilation
when forming sample coordinates, and the Swift dispatch path rejects unsupported
channel/kernel/group shapes before encoding instead of relying on a shader
early return. An experiment splitting each output-channel reduction across two
threads regressed from 1.175 ms to 2.059 ms and was rejected. The replacement
materializes the 4,096×1,152 deformable im2col matrix, multiplies it by the
1,152×64 packed checkpoint weights using 8×8 SIMD-group matrix instructions,
and converts the FP32 accumulator back to NCHW FP16. The gather now calculates
the 144 unique offset/mask coordinates once per output pixel in threadgroup
memory instead of reloading them for each of eight input channels. Bias,
FP16 conversion, and NCHW scattering are fused into the matrix dispatch, which
also removes the 1 MiB FP32 output matrix. Across four real checkpoint weight
sets, the combined median improved from 0.743–0.749 to 0.701–0.704 ms; the
stage-separated medians were about 0.347 ms gather and 0.354 ms matrix work.
The matrix kernel now reuses each loaded 8×8 weight tile across four row blocks,
producing 32×64 output tiles per threadgroup. Alternating 20-sample comparisons
across the same four checkpoints measured 0.250–0.254 ms matrix work and
0.604–0.614 ms combined DCNv2, versus 0.276–0.279 ms and 0.625–0.636 ms for the
16-row kernel. The change also corrects a three-frame probe dispatch that
launched twice the required threadgroups. The full four-pass oracle still
passes at 70.49 dB, and decoded production frames match the 16-row output
exactly. It works inside the same Metal 4 command buffer as the Metal ML
recurrence stages.

The gather kernel now precomputes each shared offset location's four input-plane
neighbor indices and interpolation fractions once per threadgroup. Reusing
those values across the offset group's eight input channels avoids repeating
coordinate flooring and boundary checks. Alternating 20-sample runs reduced
gather from 0.347–0.348 ms to 0.280–0.289 ms and combined DCNv2 from
0.599–0.607 ms to 0.534–0.550 ms. The full-model oracle and decoded production
frames remain unchanged.

Production recurrence runs now check restored FP16 outputs for NaN and Infinity
with a GPU atomic flag before returning the shared buffers. This replaces a
second Swift scalar pass over 5.9 million values for every 30-frame crop without
weakening the numerical-failure fallback. On the five-crop production fixture,
normalized non-GPU graph overhead fell by about 68 ms while model GPU time and
decoded output remained unchanged. A dedicated Metal test verifies finite,
Infinity, and NaN inputs.

## Model conversion

`tools/convert_basicvsrpp.py` splits the public Jasna/Lada BasicVSR++ checkpoint
into feature extraction, six SPyNet convolution levels, offset prediction,
propagation backbone, and upsampling packages. Deformable convolution and
SPyNet's dynamic border warp remain in custom Metal kernels.

The converter accepts `--batch-size N` for fixed-batch feasibility packages.
Batch 2 passed the M4 microbenchmark gate: compared with two batch-1 executions,
feature extraction fell from 0.748 to 0.590 ms, the 64×64 SPyNet block from
0.536 to 0.422 ms, offset prediction from 0.724 to 0.567 ms, representative
propagation backbones from 1.454 to 1.273 ms and 1.522 to 1.322 ms, and
reconstruction from 2.354 to 2.124 ms. The complete bidirectional six-level
SPyNet graph is now batch-aware: batch 2 takes 2.608 ms versus 4.500 ms for two
batch-1 executions, a 42% reduction, with finite deterministic flows and zero
repeat error. The retained feature/propagation/reconstruction tensor arena is
now batch-aware in the experimental full-graph probe: a three-frame
batch-2 restoration takes 51.893 ms versus 65.050 ms for two batch-1 executions,
a 20% reduction, and matches both independent batch-1 outputs exactly. Xcode 27
beta takes roughly three minutes to initialize all batch-2 packages in a fresh
process, but the retained graph pays that cost once. Video restoration remains
batch 1 until the crop scheduler supplies two regions to this graph. The custom
temporal preparation and DCNv2 offset stages now pass exact batch-2 isolation
tests. Gather plus SIMD-group GEMM also matches the general FP16
kernel within 0.000244: one batch-2 dispatch takes 1.047 ms versus 1.070 ms for
two batch-1 dispatches. This removes correctness blockers even though DCNv2's
own batching gain is only about 2%; the larger expected win remains in the
Metal ML packages above.

The converter intentionally emits the iOS 18 Core ML operation set, even though
the runtime target is macOS 27. Xcode 27 beta 4's Metal package builder crashes
on the newer `ios19.add` representation; the equivalent `ios18.add` compiles.

After conversion, `tools/build_metal_packages.sh` turns each `.mlpackage` into
an Xcode 27 `.mtlpackage`. The script contains a workspace-local workaround for
the beta's incorrect `coremlcompiler` lookup and does not modify Xcode.

Generate the fixed batch-2 packages once in persistent project storage with:

```sh
python3 tools/convert_basicvsrpp.py \
  --jasna-source ../../work/jasna \
  --weights Models/SourceWeights/lada_mosaic_restoration_model_generic_v1.2.pth \
  --output Models/CoreMLBatch2 \
  --batch-size 2

./tools/build_metal_packages.sh Models/CoreMLBatch2 Models/MetalMLBatch2
```

Both directories are ignored by Git because generated packages and downloaded
weights are not source artifacts. `build_and_run.sh` detects the Metal batch-2
directory automatically.

The old monolithic `spynet.mtlpackage` is retained only as evidence of the beta
limitation: Metal ML rejects its dynamic `sample_grid` operation at runtime.
The six `spynet_level_*.mtlpackage` files are the supported replacement.

The model converter expects the public Lada/Jasna checkpoint path as its first
argument. Use `--help` for all output-path and validation options. Model weights
are intentionally not copied into this repository's source history.

Generate the independent three-frame full-model oracle with the same Python
environment, Jasna checkout, and checkpoint used for conversion:

```sh
python tools/export_full_model_oracle.py \
  --jasna-source /path/to/jasna \
  --weights /path/to/lada_mosaic_restoration_model_generic_v1.2.pth \
  --frames 5 \
  --output Models/FullModelOracle/5
```

The exporter reproduces the Metal probe's deterministic FP16 input
quantization, then saves both FP32 and FP16 restored tensors. Oracle data and
model weights remain excluded from version control. Replace `--frames` and the
final output-directory component with `30` or another desired clip length.

## License

This Jasna-derived project is distributed under the GNU Affero General Public
License version 3. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Downloaded model
weights and third-party assets remain subject to their respective terms.
