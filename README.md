# Jasna Metal proof of concept

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
./script/build_and_run.sh --restore-sbs-window /path/to/input.mov /path/to/output.mov
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

This command currently accepts exactly one 30-output-frame window. It retains
the decoded BGRA window, but writes restored FP16 tiles to temporary storage
and composites only one full frame at a time. At 7680×4320 the tile cache is
about 7.47 GiB, so the command checks temporary-disk headroom before starting.
Supporting arbitrary durations requires a sequential multi-window driver and
temporal overlap at each 30-frame boundary.

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

On a 10-GPU-core Apple M4 with Xcode 27 beta 4, a 20-sample run measured the
gather-plus-SIMD-group-GEMM FP16 deformable convolution at 0.742 ms median.
Its 20-sample P10–P90 interval was 0.741–0.745 ms; one 1.723 ms outlier raised
the standard deviation to 0.214 ms. The tiled scalar reduction measured
1.175 ms, the first SIMD version 9.166 ms, and the direct baseline 16.572 ms.
The Metal-4-compatible path is 37% faster than tiled and 22.3× faster than the
direct kernel at the median. Its maximum FP16 difference from the baseline was
`0.000244`, and its FP32 implementation passed the CPU oracle with a maximum
absolute error of about `3e-8`.

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
load into Metal buffers and benchmark at 0.741–0.742 ms median through gather
plus SIMD-group GEMM. Their P10–P90 intervals stay within 0.740–0.747 ms, with
a maximum delta of `0.000977` from the direct FP16 implementation. The custom
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
and converts the FP32 accumulator back to NCHW FP16. It works inside the same
Metal 4 command buffer as the Metal ML recurrence stages.

## Model conversion

`tools/convert_basicvsrpp.py` splits the public Jasna/Lada BasicVSR++ checkpoint
into feature extraction, six SPyNet convolution levels, offset prediction,
propagation backbone, and upsampling packages. Deformable convolution and
SPyNet's dynamic border warp remain in custom Metal kernels.

The converter intentionally emits the iOS 18 Core ML operation set, even though
the runtime target is macOS 26+. Xcode 27 beta 4's Metal package builder crashes
on the newer `ios19.add` representation; the equivalent `ios18.add` compiles.

After conversion, `tools/build_metal_packages.sh` turns each `.mlpackage` into
an Xcode 27 `.mtlpackage`. The script contains a workspace-local workaround for
the beta's incorrect `coremlcompiler` lookup and does not modify Xcode.

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
