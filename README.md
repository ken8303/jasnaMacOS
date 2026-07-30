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
records three feature extractions, all four recurrent branches, twelve backbone
calls, eight offset/DCNv2 alignments, three reconstruction/upsampling networks,
the input-frame residuals, and every dependency barrier in one Metal 4 command
buffer. Over 20 samples the directly measured feature-to-restored graph took
27.098 ms median, with a 24.876–27.991 ms P10–P90 interval and 1.125 ms standard
deviation. This timing starts with input frames and precomputed flows and ends
with all three restored 256×256 RGB frames; SPyNet remains separately measured.

All twelve fused propagation tensors and all three restored frames matched the
separately submitted staged oracles bit-for-bit and had zero repeat error. The
residual add had `0.000488` maximum error, and frame checksums were `389.915688`,
`390.335297`, and `387.072357`. In the same run the four staged branch medians
totaled 40.641 ms and the independent reconstruction oracle measured 3.394 ms.

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
by `feat_extract`; flow is intentionally computed from the 64×64 copy.

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
