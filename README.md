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
On Xcode 27 beta 5 (`27A5237l`), the same correctness gate still passed with
maximum error `3.05e-5`, but streamed execution measured `1.302 ms/frame` and
specialization plus loading took `369 ms`. Core AI therefore remains parked for
the rollout path.

A separate direct Core ML microbenchmark compares three representative existing
`.mlpackage` segments with their Metal ML equivalents without changing the
restoration runtime:

```sh
JASNA_COREML_BENCHMARK_ITERATIONS=20 \
  ./script/build_and_run.sh --core-ml-comparison
```

On the same M4 and Xcode 27 beta 5, direct Core ML with `.all` and explicit
CPU+Neural Engine produced identical outputs. A 20-sample confirmation measured
warm direct Core ML medians of `0.415 ms` for feature extraction, `0.220 ms` for
the 64×64 SPyNet convolution, and `0.698 ms` for `backbone_backward_1`. The
equivalent isolated Metal ML wall medians were `0.547`, `0.478`, and `0.907 ms`,
so direct Core ML was 24%, 54%, and 23% faster at the standalone API boundary.
Metal ML device-only medians were `0.414`, `0.359`, and `0.758 ms`; its production
graph avoids most standalone wall overhead by fusing many packages into one
command buffer. Forced CPU+GPU Core ML was 4–8× slower and changed FP16 outputs
slightly. The result justifies a bounded interop spike, not a runtime switch:
every SPyNet level is separated by a custom Metal warp, so the complete six-level
Core ML/Metal graph must beat the existing fused Metal ML graph before adoption.

Xcode 27 beta 6 (`27A5252f`) preserved the isolated result: direct Core ML
medians were `0.414`, `0.217`, and `0.735 ms`, while isolated Metal ML wall
medians were `0.533`, `0.476`, and `0.892 ms`. The required complete-graph test
therefore keeps all six Core ML models loaded, shares FP16 buffers with the real
Metal pyramid/warp/add kernels, and measures every synchronization boundary:

```sh
JASNA_COREML_SPYNET_ITERATIONS=7 \
  ./script/build_and_run.sh --core-ml-spynet
```

The complete bidirectional graph rejected the Core ML path. Its seven-sample
median was `4.303 ms` versus `3.520 ms` wall time and `2.309 ms` device time for
the fused Metal ML graph, making Core ML 22.2% slower after interop. Correctness
still passed: repeat error was zero, maximum difference from Metal ML was
`0.02539`, and maximum difference from the PyTorch oracle was `0.03711`. Core ML
therefore remains a benchmark-only experiment on beta 6; production SPyNet stays
on the single Metal timeline.

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

Sparse mosaic restoration follows VR Video Toolbox CE's pre-scan design. The
detector samples each physical eye every 0.1 seconds, separates distant
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
those quality settings for controlled comparisons. An experimental adaptive mode
checks one center frame per second at a sensitive 0.05 gate threshold, then runs
0.1-second temporal mask tracking at the normal confidence only within flagged
seconds plus one second on either side. Gate detections never become restoration
masks directly. Enable it with `JASNA_ADAPTIVE_DETECT=1`, or tune
`JASNA_DETECT_COARSE_STRIDE`,
`JASNA_DETECT_COARSE_CONFIDENCE`, `JASNA_DETECT_SAMPLE_STRIDE`, and
`JASNA_DETECT_REFINE_PADDING` for controlled comparisons. It is not the default:
both one- and two-snapshot-per-second gates missed a short moving mosaic in the
accepted 30-second coverage fixture, while the normal 10 Hz scan found it. The
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
The v16 quality profile keeps that compact behaviour for ordinary regions, but
uses up to four grid crops per axis and two distributed 576-pixel detail crops
when the moving mask boundary spans more than three detail-crop widths. This
targets the 3,000–4,000-pixel-wide VR strips that still produced dark block
smears after their rectangular blend outline had been removed. The extra model
work is limited to the single largest oversized region selected in each window.
Post-subdivision crop telemetry reports exact reusable geometry, high-overlap
pairs, and contained detail passes without changing scheduling. On the
three-second beta 6 fixture all 71 crops were geometrically unique, confirming
that exact-result reuse would save no inference. The contained pairs were the
intentional high-resolution detail passes. An opt-in
`JASNA_LARGE_REGION_DETAIL_CROPS=1` A/B reduced crops from 71 to 65, model-stage
wall time from about 32.1 to 30.0 seconds, and end-to-end time from 47 to 45
seconds. Visual validation accepted the moving-boundary quality, so one detail
crop is now the rollout default. Set `JASNA_LARGE_REGION_DETAIL_CROPS=2` to
restore the wider two-detail-crop coverage for unusually difficult boundaries.
The v17 profile fills the temporal gaps between detector samples with a polygon
mask at every output frame. Each generated mask follows the interpolated tracked
box, translating and scaling the nearest real segmentation polygon instead of
cross-fading two stationary masks. This prevents fast mosaic motion from passing
through an uncovered midpoint when a one-second region contains only a few real
segmentation keyframes.
The v18 compositor bounds the full-resolution source detail added back to the
model prediction. The previous unrestricted residual could preserve dark mosaic
blocks as a smooth black smear even when detection and temporal masks were
correct. The validated default retains detail within ±0.03 normalized RGB while using the
model result for larger differences. Tune `JASNA_MOSAIC_DETAIL_RESIDUAL_LIMIT`
from `0` for direct model replacement through `1` for the legacy unrestricted
detail path.
The v19 detector post-processing makes nested and overlapping suppression
segmentation-aware. A large tracking rectangle no longer deletes a smaller
detection unless its actual mask covers that smaller subject. This fixes missed
restoration where the crop was scheduled but its retained mask was zero over a
separate moving mosaic subject. The v19 cache namespace deliberately forces
fresh manifests instead of reusing v18's already-suppressed detections.
A v19 30-second A/B reduced left-eye residual detector area from 2.947% to
1.775%. The rejected v20 experiment compared every temporal mask before
suppression and disabled the bounded source-detail residual. It raised scheduled
blend coverage to 35.696%/40.224% for the two eyes without removing the visible
pattern, making restored areas look less clean. v21 returns to v19's balanced
mask suppression and the validated 0.03 detail residual. The subsequent v21
experiment subdivided the two largest oversized subjects per window. It raised
some windows from roughly 16 to 28–30 model crops without a visible recovery
gain, so v22 restores the single-region limit and records the cleaner v19
behaviour as the stable profile. The remaining prominent pattern is a recovery
limit of the current BasicVSR++ weights; stronger blending and additional crops
make the area less clean without reconstructing the hidden detail.

For fast-moving subjects whose one-second tracked union grows much larger than
the mosaic itself, an experimental temporal-mask-guided crop mode can preserve
more model-input detail without globally raising the subdivision limit:

```sh
JASNA_TEMPORAL_CROP_FRAMES=10 JASNA_TEMPORAL_CROP_PADDING=128 \
./script/test_vr_finetuned_30s.sh input.mp4 motion-tight-test.mov 00:25:57
```

The mode is off by default. It activates only for regions at least 1024 pixels
wide or high whose per-frame mask centre moves by at least 8% of the mask grid,
splits their timeline into sections of at least three frames, and remaps masks
into a tighter padded crop for both inference and compositing. Stationary and
small regions retain the existing full temporal sequence. The four controls are
`JASNA_TEMPORAL_CROP_FRAMES`, `JASNA_TEMPORAL_CROP_PADDING`,
`JASNA_TEMPORAL_CROP_MIN_DIMENSION`, and `JASNA_TEMPORAL_CROP_MOTION`.

On macOS 27 beta, an otherwise healthy crop can occasionally hit the Metal GPU
watchdog. The launcher now recognizes that specific timeout, restarts a fresh
Metal process, and resumes the preserved per-crop checkpoint up to two times.
Other failures still stop immediately. Tune the operational retry count with
`JASNA_GPU_TIMEOUT_RETRIES`; it does not change restoration output quality.
Metal 4/MPSGraph allocations on the macOS 27 beta are released reliably only when
the restoration process exits. The macOS 27 asynchronous AVFoundation receiver
can also retain queued 8K BGRA frames until an HEVC part finishes. Isolated
decoders now seek directly to their first requested frame instead of rescanning a
120-second source segment from frame zero. Direct SBS restoration now locks each
8K source frame once and copies both 4K eyes through reusable pixel-buffer pools
in one pass. On the dense six-second M4 fixture, this reduced end-to-end wall
time from 163 to 158 seconds (3.1%) with all 180 output frames validated; the six
30-frame decode/split phases totalled 3.405 seconds and peak resident memory
remained bounded at 6.99 GiB.
Direct SBS restoration uses a
validated two-window subprocess default, then stream-copies completed windows
into validated two-minute HEVC batches and removes the superseded short media
parts. This halves model-process initialization compared with one-window
isolation while still forcing regular release of Metal, IOSurface, and encoder
resources. The default `2` is the balanced mode used by the completed
31.5-minute M4 validation run; use `1` for minimum peak memory. Four windows is
available as an experimental faster mode, while values from 5 through 30 remain
experimental on the current beta.
Completed parts are validated and reused when this limit changes while resuming
an interrupted run.
Each work directory also has its own process lock. A second command targeting the
same resume data fails clearly, while unrelated restorations are left running.
Every isolated restoration subprocess logs `Runtime memory: peak resident` at
exit. This uses Darwin's per-process high-water mark and makes six- versus
twelve-window memory comparisons visible in the persistent restoration log.
Set `JASNA_LOG_PEAK_MEMORY=0` only when this telemetry is not wanted. The former
12-window default measured 5.57–5.82 GiB before the macOS 27 receiver migration,
but the new receiver allowed retained encoder surfaces to grow beyond 32 GiB in
a long 8K run. Two-window process isolation is the validated default and
one-window isolation is the conservative fallback if a particular clip approaches
the machine's memory limit. Four-window isolation is an experimental speed mode.
Two-minute batching happens after encoding and does not retain
3,600 BGRA frames.
An Xcode 27 beta 5 five-minute 8K run completed all 300 one-second windows with
four-window isolation, zero compositor fallbacks, and per-process peaks from
about 3.4 through 6.75 GiB. A later v18 dense-region fixture hit the GPU watchdog
during its third window despite a 6.06 GiB peak, so four windows is no longer the
rollout default. This points to beta-driver stability rather than memory pressure.
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
tracked by Git. Jasna v0.10's newer `rfdetr-vr-v1` detector is supported as the
default, quality-focused Apple-MPS path. It is bundled inside the official split AMD
archive rather than offered as a separate model download. Extract and install
it from an already downloaded archive with:

```sh
./script/setup_rfdetr_detector.sh "/path/to/download-directory"
```

`rfdetr-vr-v1` remains the quality-focused VR180 default. Set
`JASNA_DETECTOR=rfdetr-v6` to try Jasna's faster general-purpose v6 model,
which uses its native medium 432×432 graph and a 0.35 confidence default.
The two checkpoints and their caches remain separate, so switching to v6 does
not overwrite or invalidate the proven VR model. Set
`JASNA_DETECTOR=yolo-v2-fast` when faster scanning is more important than
maximum coverage. RF-DETR contours are traced at their native 192×192 resolution
and only polygon coordinates are scaled back to 4K. All meaningful disconnected
mask contours are retained, and up to 64 unique object queries per frame are
considered by default; set `JASNA_RFDETR_MAX_DETECTIONS` from 1 through 200 for
unusually crowded footage. This avoids the upstream
convenience path's 12.5 GiB full-resolution mask allocation. On the known
00:28 left-eye fixture, RF-DETR found four tracked regions and 35 temporal mask
keyframes in 4.08 seconds; YOLO found three regions and 18 keyframes in 1.93
seconds. RF-DETR is therefore the coverage-oriented option rather than the
speed default. The bounded batch-2 scan peaked at 1.84 GiB resident memory with
zero swap activity on the M4, compared with the rejected wrapper path's
12.5 GiB allocation request for one frame.

Detector device selection defaults to `JASNA_DETECT_DEVICE=auto`: it uses MPS
when available and switches to CPU if an MPS inference fails. The transition
releases the failed model and empties the MPS cache before constructing the CPU
model, avoiding both copies being retained at once. Set
`JASNA_DETECT_DEVICE=cpu` to force the detector onto the CPU for diagnosis.
An RF-DETR-v6 comparison using ten samples from the 30-second 8K fixture measured
1.69 seconds of MPS inference versus 5.38 seconds on CPU, so MPS remains the
faster default on the M4. Both runs produced the same region, mask, and coverage
counts. This setting affects detection only; the complete BasicVSR++ CPU code in
this repository is a correctness oracle rather than a production video path.

On the same 30-second 4K left-eye VR fixture, `rfdetr-v6` scanned in 30.6
seconds versus 85.6 seconds for `rfdetr-vr-v1` (2.79× faster). At its normal
0.35 threshold, however, v6 scheduled only 2.275% blend area versus 9.083% for
the VR model and covered only about 15% of the VR model's scheduled rectangles.
Reducing v6 to 0.05 raised its scheduled area to 18.941% and 362 regions while
still covering only about 42% of the VR model geometry. It is therefore an
experimental general-purpose speed comparison, not a safe VR180 replacement.
A second detector-only experiment retained `rfdetr-vr-v1` but reduced sampling
from 10 Hz to 5 Hz. Scan time fell from 85.6 to 52.4 seconds, while scheduled
blend area fell from 9.083% to 7.666% and only 91.6% of the 10 Hz scheduled
geometry remained covered. Because the known failure mode is a missed moving
mosaic, the quality path continues to use the full 10 Hz scan.

On the same 120-second 4096×4096 beta-5 eye clip, RF-DETR scanned in 317.1
seconds and scheduled 3.288% blend area per eye-frame. YOLO scanned in 162.5
seconds, but scheduled 3.571% and still marked all 120 windows active. YOLO was
1.95× faster at detection without creating clean-window bypass on this fixture;
RF-DETR remains the quality rollout default while YOLO remains the explicit
throughput option.

Detector logs also report average active regions per frame and scheduled blend
area per eye-frame. These normalized values make RF-DETR and YOLO coverage load
comparable even when they emit different region counts. Direct SBS runs finish
with a stereo compositor fallback summary; `fused 0, CPU 0` confirms that the
Metal beta fallback chain was not exercised during that session.

The direct SBS workflow groups consecutive one-second windows with no detected
regions and copies those packets from the prepared 30 fps SBS source. Only active
ranges enter the Metal restoration writer; the validated bypass and restored
segments are joined before source audio is copied. If a source cut is not
keyframe-exact, the workflow automatically falls back to the Metal writer for
that range instead of accepting a damaged or incorrectly timed segment.
Before restoration, a conservative stereo reconciliation checks for complete
one-second detection gaps: when one eye has regions and the counterpart eye has
none, it transfers those regions using disparity measured from nearby matched
tracks. Windows already active in both eyes are unchanged. Set
`JASNA_STEREO_MANIFEST_RECONCILE=0` for an A/B comparison.

Test the left eye with:

```sh
./script/restore_vr_eye_sparse.sh \
  /path/to/input_30fps.mp4 left /path/to/restored-left.mov
```

Or run the coverage-oriented detector on a short SBS test:

```sh
./script/test_vr_sparse_30s.sh \
  /path/to/input.mp4 /path/to/rfdetr-test.mov 00:00:28
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

Sparse output keeps the one-second recurrence boundaries and, by default, feeds
four consecutive windows to one HEVC writer. Once that four-second output is
validated, its FP16 model caches are removed. This bounded an 8K eye-by-eye
five-minute run at 5.75 GiB peak resident memory and avoided the former
120-window behavior retaining more than 7 GiB of crop caches. Existing one- and
five-window outputs are still detected and reused. Set
`JASNA_ENCODER_WINDOWS_PER_SEGMENT=1` for one file per recurrence window, or
choose another positive segment size. Larger values reduce hardware-encoder
startup and drain work, but proportionally increase temporary disk use until
the encoded segment is validated.

All restartable source clips, manifests, restored windows, and eye videos remain
beside the output by default. After visually checking a completed output, they
can be removed manually. Set `JASNA_CLEAN_WORK_ON_SUCCESS=1` to remove that work
directory automatically, but only after the final SBS output passes codec,
dimensions, frame-count, duration, and decode validation. The log and final
video remain beside one another.

Metal ML crop execution is serialized. The first normal 30-frame crop or
35-frame crop with temporal warm-up builds the retained graph and every later
crop of that temporal shape reuses it; attempting to construct two
first-use graphs concurrently proved unstable in the macOS 27 beta runtime.
`JASNA_REGION_CONCURRENCY` is therefore ignored for production restoration.

The optimized production path retains one complete Metal graph per model batch
for the lifetime of the eye-restoration process. A change between the 30-frame
and 35-frame temporal shapes evicts the former graph before constructing the
replacement, preventing the warm-up optimization from accumulating multiple
large graphs. Later crops reuse the same
tensors, buffers, argument tables, and residency set under a lock. Sequential
Metal ML dispatches also share scratch heaps per pipeline/level instead of
allocating hundreds of identical heaps per crop. A 4096×4096 one-second proof
measured about 1.10 seconds for the initial graph build and execution, then
about 0.45 seconds per reused 30-frame crop including roughly 0.38 seconds of
GPU work. Decoded frame hashes matched the pre-cache output exactly.

Xcode 27 beta 6 exposed that the former exact-30-frame gate bypassed this cache
for nearly every production crop after five-frame temporal warm-up was enabled.
The bounded 35-frame cache reduced the identical three-second batch-2 fixture's
model-stage wall time from `34.8` to `32.1` seconds (7.8%) and completed the
full restoration-only path in 47 seconds. Peak resident memory was 7.45 GiB.
An alternating-input FP16 probe matched a fresh graph bit-for-bit, including
hash `9fce8e1d284bfdc2`; decoded HEVC A/B differences remained compression-level
(about 59–67 dB PSNR). Set `JASNA_RETAINED_GRAPH=0` for an immediate rollback.

Restoration uses batch 2 by default, grouping two mosaic crops of the same
temporal length in one retained graph when `Models/MetalMLBatch2` is present.
Incompatible or odd final crops stay on batch 1. The first batch failure retries
both crops individually and disables batch 2 for the remainder of that process,
preventing a bad graph from imposing another minute-long timeout on every later
crop. Set `JASNA_MODEL_BATCH=1` for the conservative path. Batch-2 package
initialization can take roughly three minutes in a fresh process under Xcode 27
beta, making the larger process boundary particularly valuable.

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

Each top-level SBS run and standalone eye run holds an atomic workflow lock for
its complete lifetime, including source splitting and detector scanning. Resume
configuration records the source metadata, detector checkpoint, restoration
model metadata, implementation fingerprint, and every output-affecting quality
option. A changed source, model, implementation, or quality setting therefore
requires a new output name instead of silently reusing old media or manifests.

The direct SBS launcher prepares both physical 4096×4096 eye-segment streams
from one shared 8192×4096 decode. Both VideoToolbox encoders receive their crop
from the same decoded frame, eliminating the former second full source decode.
Completed `source.done` markers remain compatible with resume, so an existing
run does not regenerate its eye segments after this optimization.
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

Direct 8K SBS output fuses left/right assembly and mosaic compositing into one
Metal command buffer, removing one submission and synchronous wait per frame.
Each 30-frame window logs writer wall time plus the preparation and encoder-wait
sums. A bounded one-frame lookahead was tested and rejected: an exact M4 A/B
measured 1,265.5 ms versus 1,228.5 ms serial because concurrent Metal
preparation contended with VideoToolbox on the shared memory fabric.

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

For the optimized quality-preserving profile, use:

```sh
./script/test_vr_fast_30s.sh \
  /path/to/input-sbs.mp4 /path/to/restored-vr-fast.mov 00:12:00
```

To validate the same paired 10 Hz path with an MP4 source, one shared SBS MP4
working file, no physical eye exports, and a final MP4 output, use:

```sh
./script/test_vr_direct_sbs_mp4_30s.sh \
  /path/to/input-sbs.mp4 /path/to/restored-direct-sbs.mp4 00:25:57
```

The original 59.94 fps Main-10 source still requires one hardware-normalized
30 fps SBS working file: Apple VideoToolbox has rejected direct AVFoundation
decode of this source profile in prior tests, and the restoration manifests use
an exact 30 fps timeline. That working file remains stereo and is never exported
as separate physical left/right movies. Per-eye paths are links plus independent
region manifests and caches. Internal resumable Metal segments remain MOV; the
user-facing result is MP4.

This profile keeps RF-DETR sampling at 10 Hz and the validated batch-2 model,
but avoids temporary left/right source encodes. Both eye jobs reference the
same SBS segment. The default shared stereo detector loads RF-DETR once, decodes
each sampled 8K frame once, submits its left/right 4K crops through the retained
model, and writes the same separate eye manifests used by restoration. Set
`JASNA_STEREO_DETECT=0` to retain the legacy two-process detector for diagnosis.
The diagnostic `script/test_vr_alternating_30s.sh` halves detector images, but
its first 30-second comparison left 8.7%/9.2% of paired left/right region-frames
with less than half spatial coverage. It therefore did not pass the promotion
gate. Paired 10 Hz detection remains the validated default.
The stereo restoration path also decodes each SBS frame once. Frame-exact,
decodable packet-copy
segments are reused directly. If a long-GOP source cannot be split exactly at a
segment boundary, the launcher automatically performs one hardware SBS encode
with forced boundary keyframes instead of producing mismatched eye timelines.
Crop results use a bounded 512 MiB in-memory handoff when possible and fall back
automatically to the persistent disk cache for larger or resumed windows.
On the dense 30-second 8K fixture, this path validated all 900 frames with zero
stereo-compositor fallbacks. The worst two-window subprocess peak was 7.48 GiB,
no FP16 crop files were left behind, and total wall time was 616 seconds
including a 31.7-second release rebuild. The prior physical-eye path took 634
seconds with an already-built executable; excluding the one-time rebuild, the
new run saved about 50 seconds while RF-DETR also scheduled slightly more source
geometry from the cleaner, non-re-encoded input.

On the dense six-second M4 fixture, shared detection preserved the exact 33/27
left/right region counts, 22.272%/24.276% scheduled coverage, and all 1,629
temporal mask keyframes. Detector scan time fell from 36.462 to 33.231 seconds,
while complete end-to-end wall time fell from 154 to 149 seconds. The output
passed as 180 8K frames with zero compositor fallbacks, and peak restoration
memory remained bounded at 7.20 GiB. The one-second startup-heavy fixture fell
from 48 to 40 seconds with the same manifests, showing the additional saving
from avoiding a second RF-DETR model load.

After validating quality, restore the complete SBS source with the production
wrapper:

```sh
./script/restore_vr_sparse_sbs.sh \
  /path/to/input-sbs.mp4 /path/to/restored-full.mp4
```

When the mosaic time ranges are already known, use the manual-range wrapper:

```sh
./script/restore_vr_sparse_ranges.sh \
  /path/to/input-sbs.mp4 /path/to/restored-full.mp4 \
  "00:12:00-00:14:00,00:20:30-00:22:00"
```

For rollout, use the preset wrapper instead of assembling environment flags.
It fixes the validated runtime profile: the baseline `MetalML` restoration
packages, RF-DETR, batch 2 when `MetalMLBatch2` is present, full mask-hole
recovery, five temporal warm-up frames, two Metal windows per process,
direct/shared SBS output, a bounded 512 MiB crop handoff, stereo reconciliation,
peak-memory telemetry, and adaptive detection disabled. It deliberately ignores
a pre-existing `JASNA_MODELS_DIR`, keeping fine-tuned weights out of runtime
sign-off; use the dedicated fine-tuned or restoration-only A/B wrappers for
candidate models. It falls back to batch 1 when the baseline batch-2 packages
are unavailable. Its optional third argument supplies known mosaic ranges:

```sh
./script/restore_vr_rollout.sh \
  /path/to/input-sbs.mp4 /path/to/restored-full.mov \
  "00:12:00-00:14:00,00:20:30-00:22:00"
```

Times refer to the original source timeline. RF-DETR and Jasna run only inside
those intervals. With direct SBS output, completely clean 120-second segments
are not converted into left/right eye videos at all: their prepared SBS packets
are copied directly into the final timeline. Only intersecting segments are
cropped into eye videos, and clean one-second windows inside boundary segments
are still packet-copy bypassed. Repeating the identical command resumes safely;
changing the ranges requires a new output filename so cached manifests cannot
be mixed.
Before detection, every cached left/right source pair is checked for matching
30 fps frame counts, dimensions, codec, shared-file identity when applicable,
and a decodable first frame. If valid paired segments survive but one completion
marker is missing, both markers are recovered without repeating the encode. A
partial pair or an oversized output from an interrupted/beta encoder session is
archived together with its shared source and dependent eye cache, then
regenerated, preventing mismatched timelines from reaching stereo manifest
reconciliation.

If a script update interrupts a run after compatible manifests or restored
windows have already been written, set `JASNA_ALLOW_IMPLEMENTATION_RESUME=1`
for the first restart. The launcher accepts this only when the input, model,
manual ranges, and all quality settings still match; it then records the new
implementation identity so later restarts do not need the override.

It does not apply the test harness's 30-second cut. It uses persistent
120-second source/restoration segments, fisheye sparse regions, direct SBS
output, and the full-run-validated batch-2 model path by default. The launcher
limits each Metal process to two temporal windows so beta-runtime allocations
are released regularly. Set `JASNA_METAL_WINDOWS_PER_PROCESS=1` for the most
conservative memory mode, or `JASNA_MODEL_BATCH=1` for the conservative model
path. A restart reuses completed source segments, region manifests, restored windows,
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
The isolated eye-by-eye test profile makes that lower-memory path process one
120-second 4096x4096 eye segment per Metal process. The process exits after
each validated segment, releasing its Metal/ANE state before the next eye job.
Matching restored eye segments are immediately encoded as independently
validated 8192x4096 SBS checkpoints. The final stage joins those SBS checkpoints
without first creating complete left- and right-eye movies, and it copies source
audio without re-encoding:

```sh
./script/restore_vr_eye_by_eye.sh \
  /path/to/input-sbs.mp4 /path/to/restored-eye-by-eye.mov
```

Use the 30-second eye-by-eye loop while tuning performance or quality. Keep the
same start time and use a new output filename for each configuration so the
wall-time measurements remain comparable:

```sh
./script/test_vr_eye_by_eye_30s.sh \
  /path/to/input-sbs.mp4 /path/to/restored-eye-by-eye-30s.mov 00:12:00
```

The log reports the total wall time after output validation. Once a change is
faster and its image quality is acceptable, run the bounded five-minute
comparison before committing a full source:

```sh
./script/test_vr_eye_by_eye_5min.sh \
  /path/to/input-sbs.mp4 /path/to/restored-eye-by-eye-5m.mov 00:12:00
```

For the full-source wrapper, known mosaic ranges can be supplied as the optional
third argument. The 30-second and five-minute test wrappers use their third
argument as the start time. Use a new output filename when comparing this profile with direct SBS so their
persistent work directories and configuration identities remain separate.
Sparse mosaic scans decode HEVC sequentially and infer two sampled frames at a
time. Set `JASNA_ADAPTIVE_DETECT=1` only for speed/coverage comparisons; the full
10 Hz scan remains the quality default. Set `JASNA_DETECT_BATCH_SIZE=1` to
minimize memory, or
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
The v22 compositor keeps subdivision accumulators in private GPU memory and
clears them with a Metal kernel instead of allocating shared buffers and
zero-filling hundreds of megabytes from the CPU for every frame. On the same two
8K windows from the v19 production fixture, writer/compositor time fell from
`16.241 s` to `4.946 s` (69.5%), while the complete cached two-window job fell
from about 50 seconds to 31 seconds. Peak resident memory was 4.63 GiB, no
fallback fired, and all 60 decoded output frames were pixel-identical to the
previous compositor (`inf` PSNR). Region geometry varies, so the full-run gain
will depend on how many large subdivided subjects are active.
A follow-up attempt to combine every restored crop, original crop, and mask into
one aligned upload buffer per frame was rejected on the same fixture. Writer
time regressed from `4.946 s` to `5.244 s` and peak resident memory rose from
4.63 GiB to 5.43 GiB, so v22 retains the smaller per-region shared uploads.
Reusing one private accumulator set across the sequential left/right subdivision
groups was also rejected. Although two repeats reduced writer time to about
`4.02 s`, peak memory rose to 6.93–7.40 GiB and the encoded result no longer
matched the accepted output (47.35 dB PSNR). Independent group storage therefore
remains required by the current queued Metal resolve path.

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

This full-frame tile plan is the historical upper-bound baseline. The later
sparse VR pipeline now connects decoding, stereo eye preparation, temporal
restoration, normalized blending, hardware encoding, clean-window bypass, and
resumable final assembly end to end. On Xcode 27 beta 5, a v22 fully automatic
five-minute `8192×4096` run completed in about 78.5 minutes (15.7 seconds of
wall time per source second). Stereo reconciliation bypassed 73 of 300
one-second windows and restored 227. The output passed as 9,000 HEVC frames at
30 fps with an exact 300-second duration; every isolated compositor summary
reported zero fused and CPU fallbacks. Peak resident memory across the
two-window restoration subprocesses was 8.07 GiB, with no GPU timeout,
non-finite output, or process abort. Approximately 12.5 minutes were source and
eye preparation, 27.7 minutes were the conservative 10 Hz RF-DETR scans, and
38.3 minutes were restoration plus final assembly. This remains an offline
workflow, but validates clean-window bypass and bounded memory over a sustained
mixed clean/dirty range rather than only a short fixture.

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

## Experimental restoration fine-tuning

The remaining strong pattern on some moving mosaics is a limitation of the
public `lada_mosaic_restoration_model_generic_v1.2.pth` checkpoint. Increasing
mask coverage or adding more overlapping crops cannot reconstruct detail that
the primary model never predicts. `tools/prepare_finetune_dataset.py` and
`tools/finetune_basicvsrpp.py` provide a conservative same-architecture
fine-tuning path. SPyNet stays frozen and accepted checkpoints retain the exact
parameter layout consumed by the existing Metal converter, so model inference
cost does not increase.

Training requires clean, unmosaicked source video. Already-mosaicked output is
not ground truth and must not be used as the target. Prepare isolated train and
validation clips on the Mac; `--sbs-eyes` prevents a crop from crossing the eye
boundary of a clean SBS source:

```sh
python3 tools/prepare_finetune_dataset.py \
  /path/to/clean/videos \
  --output /path/to/jasna-finetune-data \
  --clips-per-video 20 \
  --frames 30 \
  --source-crop-size 1024 \
  --exclude-ranges 00:25:57-00:26:27 \
  --sbs-eyes
```

The extractor records every source/time/crop choice in `dataset.json` and
refuses to overwrite a non-empty dataset. Synthetic moving mosaics are generated
from those clean clips during training, so their block size and position change
without duplicating ground-truth frames on disk. `--source-crop-size 1024`
downscales a larger 8K-eye area into the model's 256-pixel input, approximating
the scale of an oversized VR restoration region. `--exclude-ranges` accepts the
same comma-separated source timeline syntax as manual restoration ranges and
adds a two-second guard on both sides by default. It is intended for known
mosaic intervals in an otherwise clean source; every extracted frame still
needs visual review before training. At least two clips are required per input
video; each source is represented independently in both train and validation
so the promotion score cannot accidentally omit an entire source. Run the
sensitive VR detector audit as a
mandatory second gate:

```sh
./script/audit_finetune_dataset.sh /path/to/jasna-finetune-data
```

For a larger candidate dataset, the first audit can quarantine every flagged
clip instead of requiring manual moves. Quarantine is recoverable: clips move
under `rejected/train` or `rejected/validation` and are never deleted. This
also rebalances a surviving clip when a source loses all of its train or
validation representatives, recording both its original and current split in
`dataset.json`. This
first command intentionally exits with status 2 when it moves anything; run a
second audit without quarantine to certify the remaining exact clip set:

```sh
JASNA_FINETUNE_QUARANTINE=1 \
JASNA_FINETUNE_AUDIT_STRIDE=10 \
./script/audit_finetune_dataset.sh /path/to/jasna-finetune-data

./script/audit_finetune_dataset.sh /path/to/jasna-finetune-data
```

The audit samples the first, last, and intermediate frames of every clip and
writes `mosaic-audit.json`. It also rejects black/empty and very low-detail
clips that do not add useful restoration targets. Any rejected clip makes the
command fail and the dataset must not be trained. The trainer independently
requires a passing audit whose exact clip list still matches the dataset. This
deliberately prefers rejecting a questionable clean clip over teaching the
restoration model to reproduce mosaic blocks.

Full BasicVSR++ back-propagation is intended for a Python 3.12+ CUDA training
machine with Jasna's development dependencies installed. The
M4 remains the validation and Metal deployment target; MPS/CPU training is an
explicit diagnostic option because deformable-convolution backward support and
speed are not validated there. Start with a small run before renting a longer
GPU session:

```sh
python tools/finetune_basicvsrpp.py \
  --jasna-source /path/to/jasna \
  --weights Models/SourceWeights/lada_mosaic_restoration_model_generic_v1.2.pth \
  --dataset /path/to/jasna-finetune-data \
  --output /path/to/jasna-finetune-run \
  --steps 1000 \
  --frames 10
```

The CUDA wrapper keeps the complete console log beside the checkpoints and
provides environment overrides for clip length, learning rate, validation
frequency, and an explicit resume checkpoint:

```sh
JASNA_TRAIN_PYTHON=/path/to/jasna-venv/bin/python \
./script/run_finetune_basicvsrpp.sh \
  /path/to/jasna \
  /path/to/jasna-finetune-data \
  /path/to/jasna-finetune-run \
  1000
```

An Apple-silicon Mac can run the already validated CPU diagnostic path when a
CUDA machine is unavailable. Start with three frames and 150 steps; CPU is
slower, but avoids relying on unsupported MPS deformable-convolution backward
operations:

```sh
JASNA_FINETUNE_DEVICE=cpu \
JASNA_FINETUNE_FRAMES=3 \
JASNA_FINETUNE_VALIDATE_EVERY=25 \
JASNA_FINETUNE_SAVE_EVERY=50 \
./script/run_finetune_basicvsrpp.sh \
  /path/to/jasna-source \
  /path/to/audited-dataset \
  /path/to/mac-pilot-run \
  150
```

A fresh run refuses a non-empty output directory. Resume an interrupted run
with `JASNA_FINETUNE_RESUME=/path/to/checkpoint-000500.pth` and a larger target
step; the original v1.2 baseline remains immutable across resumes.

The first three-frame Mac candidate improved its synthetic validation PSNR from
23.183 to 23.412 dB and temporal MAE from 0.017968 to 0.016200 by step 500, but
its fixed real moving-mosaic A/B looked almost unchanged. That test had already
scheduled blend coverage of 44.5% for the left eye and 40.3% for the right eye,
so increasing detector coverage again would only add work. Continue from the
accepted checkpoint with the harder `moving-vr` corruption profile instead.
It uses five-frame samples, faster curved motion, changing pixel-block sizes,
non-rectangular masks, and occasional independent overlapping subjects. Because
this validation task differs from the first recipe, the quality baseline must
be measured again and the result must go into a new run directory:

```sh
JASNA_FINETUNE_DEVICE=cpu \
JASNA_FINETUNE_RESUME="/path/to/mac-pilot-v1/checkpoint-000500.pth" \
JASNA_FINETUNE_CORRUPTION_PROFILE=moving-vr \
JASNA_FINETUNE_RESET_BASELINE=1 \
JASNA_FINETUNE_FRAMES=5 \
JASNA_FINETUNE_LEARNING_RATE=0.000005 \
JASNA_FINETUNE_TEMPORAL_WEIGHT=0.20 \
JASNA_FINETUNE_VALIDATE_EVERY=25 \
JASNA_FINETUNE_SAVE_EVERY=50 \
./script/run_finetune_basicvsrpp.sh \
  /path/to/jasna-source \
  /path/to/audited-dataset \
  /path/to/mac-motion-v2 \
  750
```

On resume, the wrapper's requested learning rate now overrides the optimizer
checkpoint's old rate. This prevents a staged recipe from silently continuing
with the previous learning rate. Do not train the old three-frame recipe beyond
step 500 unless the fixed real-video A/B shows a benefit.

The moving-VR step-750 candidate also passed its synthetic gate but remained
almost unchanged in the same real 30-second A/B: the step-500 and step-750
outputs measured 49.40 dB PSNR against each other. Inspection of the source
mosaic found stable averaged grid blocks with hard or lightly softened edges.
The earlier trainer generated blocks directly at model resolution and omitted
Lada's post-mosaic video degradation. It also retained EMA decay 0.999, meaning
only 22.1% of the resumed EMA at step 750 came from the new 250-step stage.

The `lada-vr` profile more closely follows Lada's training recipe: one block
grid is retained through a temporal sample, square and rectangular average or
sampled blocks are selected through whole mask cells, optional feathering is
applied, and lightweight blur plus one or two quantization passes approximate
post-mosaic video compression. Continue into another isolated run with longer
temporal context and a faster EMA that can actually incorporate the stage:

```sh
JASNA_FINETUNE_DEVICE=cpu \
JASNA_FINETUNE_RESUME="/path/to/mac-motion-v2/checkpoint-000750.pth" \
JASNA_FINETUNE_CORRUPTION_PROFILE=lada-vr \
JASNA_FINETUNE_RESET_BASELINE=1 \
JASNA_FINETUNE_FRAMES=8 \
JASNA_FINETUNE_LEARNING_RATE=0.000005 \
JASNA_FINETUNE_TEMPORAL_WEIGHT=0.20 \
JASNA_FINETUNE_EMA_DECAY=0.995 \
JASNA_FINETUNE_VALIDATE_EVERY=25 \
JASNA_FINETUNE_SAVE_EVERY=50 \
./script/run_finetune_basicvsrpp.sh \
  /path/to/jasna-source \
  /path/to/audited-dataset \
  /path/to/mac-lada-vr-v3 \
  1000
```

At decay 0.995, 250 optimizer updates give the new stage about 71.4% EMA
contribution instead of 22.1%. The trainer prints this estimate at startup so a
future short staged run cannot appear successful while barely changing the
exported EMA weights.

The trainer initializes both working and EMA generators from v1.2, freezes
SPyNet, uses masked Charbonnier reconstruction plus a low-weight clean-background
term and temporal-delta loss, clips gradients, and writes resumable optimizer
checkpoints atomically. A candidate becomes `best-inference.pth` only when its
held-out masked PSNR improves by at least 0.10 dB and its temporal MAE remains
within 2% of the starting checkpoint. These are synthetic-data gates, not an
automatic production promotion: the candidate must still pass the full-model
oracle and a fixed real 30-second visual A/B before replacing any generated
Metal package.

After an accepted candidate has been converted into
`Models/MetalMLFineTuneMac1000`, run the isolated visual test without replacing
the production packages:

```sh
./script/test_vr_finetuned_30s.sh \
  /path/to/input-sbs.mp4 \
  /path/to/fine-tuned-test.mov \
  00:25:57
```

This test forces batch 1 until a matching fine-tuned batch-2 package set is
converted. It defaults to a 0.10 detector threshold and 18% segmentation-mask
expansion because the first real A/B retained intermittent moving-edge mosaic
at the normal 0.15/10% settings. It also enables the approved delta-gated
mask-hole recovery for ordinary regions and five temporal warm-up frames.
Override any setting explicitly for comparison. `JASNA_MODELS_DIR` and any
selected batch-2 model directory are included in restoration cache identity, so
a baseline or differently batched cache cannot be reused for the candidate
accidentally. Set `JASNA_FINETUNED_MODELS_DIR` to compare a different converted
candidate, such as `Models/MetalMLFineTuneMac750`.

Fine-tuned packages remain an experimental quality track even after their
synthetic training gate passes. They are not selected by
`restore_vr_rollout.sh` or `test_vr_direct_sbs_mp4_30s.sh`. Promotion requires a
fixed-manifest baseline/candidate A/B, objective quality measurements, and visual
acceptance on moving-mosaic fixtures; runtime validation is signed off
separately with the baseline packages.

Once one 30-second test has prepared the SBS source and RF-DETR manifests, use
the restoration-only harness for subsequent model or crop-geometry A/B runs:

```sh
JASNA_MODELS_DIR="$PWD/Models/MetalMLFineTuneMac1000" \
JASNA_MODEL_BATCH=1 \
./script/test_vr_restore_only_30s.sh \
  "/path/to/previous-test.jasna-vr30-v22-work" \
  "/path/to/restoration-only-candidate.mov"
```

This reuses the exact prepared 900-frame 8K clip and reconciled left/right
detector manifests. It does not rerun source preparation or RF-DETR, so detector
coverage is held constant and the log measures only restoration, clean-window
bypass, and the final packet join. Each output receives an independent cache,
which prevents model results from being reused across candidates. Set
`JASNA_RESTORE_ONLY_VALIDATE=1` first to check the reference assets without
building or writing video. For faster troubleshooting, select only the affected
source-relative seconds inside the prepared 30-second clip:

```sh
JASNA_RESTORE_ONLY_START_SECOND=8 JASNA_RESTORE_ONLY_SECONDS=2 \
JASNA_MODELS_DIR="$PWD/Models/MetalMLFineTuneMac1000" \
JASNA_MODEL_BATCH=1 \
./script/test_vr_restore_only_30s.sh \
  "/path/to/previous-test.jasna-vr30-v22-work" \
  "/path/to/restoration-only-seconds-8-10.mov"
```

The start is relative to the prepared clip, not the original video's absolute
timeline. Omitting the duration restores from the selected second through the
end of the 30-second fixture.

For a controlled baseline-versus-fine-tuned comparison, run both model sets
against that same source and those same manifests with one shared executable:

```sh
JASNA_RESTORE_ONLY_START_SECOND=0 JASNA_RESTORE_ONLY_SECONDS=3 \
./script/test_vr_restore_ab.sh \
  "/path/to/previous-test.jasna-vr30-v22-work" \
  "/path/to/recovery-ab"
```

This writes `recovery-ab-baseline.mov` with `Models/MetalML` and
`recovery-ab-candidate.mov` with `Models/MetalMLFineTuneMac1000`. Override
`JASNA_AB_BASELINE_MODELS_DIR` or `JASNA_AB_CANDIDATE_MODELS_DIR` to compare
other converted package sets. Detector preparation remains frozen for both
outputs, so the only intended A/B variable is the restoration model.

To inspect whether the model itself retains mosaic structure before final
compositing, enable recovery diagnostics on a fresh restoration-only output:

```sh
JASNA_RECOVERY_DIAGNOSTIC_DIR="/path/to/recovery-diagnostics" \
JASNA_RECOVERY_DIAGNOSTIC_REGION=all \
JASNA_RECOVERY_DIAGNOSTIC_FRAME=15 \
JASNA_RESTORE_ONLY_START_SECOND=0 JASNA_RESTORE_ONLY_SECONDS=1 \
JASNA_MODELS_DIR="$PWD/Models/MetalMLFineTuneMac1000" \
JASNA_MODEL_BATCH=1 \
./script/test_vr_restore_only_30s.sh \
  "/path/to/previous-test.jasna-vr30-v22-work" \
  "/path/to/recovery-diagnostic.mov"
```

Large subdivided mosaic regions also use the raw model delta to recover pixels
that a moving temporal mask misses. The default activation threshold is `0.025`
in normalized RGB and is intentionally limited to subdivision groups. Raise it
to make mask-hole recovery more conservative, or lower it when a verified raw
reconstruction is still being suppressed:

```bash
JASNA_MOSAIC_MASK_RECOVERY_THRESHOLD=0.025
```

Raw-crop diagnostics can also show a strong restoration for an ordinary region
whose resolved temporal mask is empty. Test delta-gated recovery for those
non-subdivided regions without changing detection or model weights:

```bash
JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=1
```

This remains opt-in for experimental launchers, while the visually approved
fine-tuned test and rollout wrappers enable it. The same spatial feather and
delta threshold still bound the additional blend.

One-second crop windows normally start with fresh temporal recurrence. To test
whether that reset causes weak recovery immediately after a window boundary,
prepend a small number of already-decoded source frames to each crop model call:

```bash
JASNA_TEMPORAL_WARMUP_FRAMES=5
```

Warm-up defaults to 5 frames, is clamped to that limit, and does not add output
frames. Set `JASNA_TEMPORAL_WARMUP_FRAMES=0` for an immediate rollback. The
preceding frames are passed through the restoration graph, then discarded before
the intended 30 output frames are cached and composited. Later windows reuse a
small in-memory decode tail, so this does not seek or decode the 8K source twice.

On the fixed three-second 8K restoration-only fixture, both three- and five-frame
warm-up passed 90-frame validation with zero compositor fallbacks. Three frames
finished in 59 seconds; five frames took about 64 seconds after excluding its
one-time release rebuild. The five-frame result produced the stronger bounded
change during the first three frames after both recurrence boundaries, while the
first 30-frame window remained bit-identical. After visual approval, five frames
became the production default.

The matching fine-tuned batch-2 package set subsequently passed the same
three-second 8K restoration-only fixture with all 90 decoded frames pixel-identical
to batch 1 and zero compositor fallbacks. Its warm run finished in 54 seconds
versus 64 seconds for batch 1 after excluding the latter's one-time release build,
a 15.6% end-to-end improvement. Measured GPU work fell from 31.825 to 28.812
seconds (9.5%), while peak resident memory rose from 5.79 to 7.39 GiB. The rollout
wrapper now selects this package set automatically when present.

For each selected crop this writes the raw 256x256 model input and output,
an amplified input/output difference, the resolved segmentation mask, and JSON
geometry/delta telemetry. Use a non-negative zero-based region index instead of
`all` to limit output. Diagnostics are opt-in and do not alter compositing.

The `lada-vr-v4` fine-tuning recipe targets the observed conservative recovery:
it synthesizes larger, faster, more heavily compressed moving mosaics and adds a
masked spatial-gradient loss that penalizes retained block edges. Resume it into
a new output directory so the earlier candidate remains reproducible:

```sh
JASNA_TRAIN_PYTHON="$PWD/.venv-coreai/bin/python" \
JASNA_FINETUNE_DEVICE=cpu \
JASNA_FINETUNE_RESUME="/path/to/checkpoint-001000.pth" \
./script/run_finetune_lada_vr_v4.sh \
  "/path/to/jasna-source" \
  "/path/to/audited-dataset" \
  "/path/to/mac-lada-vr-v4" \
  1250
```

The wrapper defaults to eight frames, a `5e-6` learning rate, 0.15 temporal
weight, 0.25 gradient weight, EMA 0.995, validation every 25 steps, and a reset
quality baseline for resumed recipes. All values remain overridable through the
existing `JASNA_FINETUNE_*` environment variables.

Convert an accepted candidate into separate experimental directories first:

```sh
python tools/convert_basicvsrpp.py \
  --jasna-source /path/to/jasna \
  --weights /path/to/jasna-finetune-run/best-inference.pth \
  --output Models/CoreMLFineTune \
  --validate

python tools/convert_basicvsrpp.py \
  --jasna-source /path/to/jasna \
  --weights /path/to/jasna-finetune-run/best-inference.pth \
  --output Models/CoreMLFineTuneBatch2 \
  --batch-size 2 \
  --validate
```

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
Internal graph-phase telemetry is available with
`JASNA_GRAPH_PHASE_TELEMETRY=1`. On a dense batch-2 30-frame window, steady
submissions spent about 1 ms uploading inputs, 7 ms encoding commands, 625–661
ms waiting for completion, and 0.5 ms reading outputs; reported GPU work was
roughly 566–592 ms. Crop-pair packing and result splitting together added only
about 10–18 ms per eye/window. This rules out CPU prefetch and array packing as
meaningful targets: the remaining host gap is predominantly Metal/driver queue
latency.

The crop scheduler keeps stable same-length pairs ahead of odd leftovers. A
single odd temporal-length group can therefore no longer shift a later even
group off the batch-2 boundary. On the dense six-second M4 fixture, the same 139
crops required 77 graph submissions instead of 80; graph wall time fell from
59.508 to 58.794 seconds and end-to-end wall time from 158 to 154 seconds. Peak
resident memory remained bounded at 7.28 GiB, and all 180 output frames passed
the direct SBS validation gate.

A fixed batch-4 experiment was also rejected on Xcode 27 beta 5. All 16 Core ML
and Metal ML packages converted successfully, but the first restoration graph
remained blocked for more than six minutes in synchronous ANE
`compileModel` specialization, with a 4.8 GiB physical footprint and no first
submission. The batch-4 scheduler was removed; batch 2 remains the validated
ceiling on this beta.

The converter intentionally emits the iOS 18 Core ML operation set, even though
the runtime target is macOS 27. Xcode 27 beta 5's Metal package builder crashes
on the newer `ios19.add` representation with `Missing pattern to rewrite MIL op
ios19.add`; the equivalent iOS 18 packages compile. The isolated feature-extract
probe was repeated with Xcode 27 beta 6 (`27A5252f`) and its matching downloaded
Metal Toolchain, and aborted on the same `ios19.add` rewrite exception. Pass
`--minimum-deployment-target ios26` only for isolated future-beta probes.

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
