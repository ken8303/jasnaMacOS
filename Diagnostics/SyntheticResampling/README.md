# Standalone CPU / Metal resampling diagnostic

This is an independent Swift package using generated RGBA ramps and moving
checkerboards. It does not import the application, open videos, load models, or
change restoration settings. It tests ordinary bilinear sampling, not a model,
fisheye projection, mask compositing, or the Neural Engine.

## Run

From the repository root on macOS 27 with the Swift 6.4 toolchain:

```sh
./script/test_synthetic_metal.sh
```

For the new mixed-size experiment only, without repeating the earlier fixed-size
timings:

```sh
./script/test_synthetic_metal.sh --mixed-only
```

This still runs the identity, moving-pattern, and resident-safety gates. Its JSON
records `requestedBenchmarks: "mixed-only"`; the two fixed-size result arrays are
empty because they were skipped, not because they passed. Without the flag, all
four standard timing sections run. The optional results directory also works after any
focused flag. To test new sizes and changing reuse counts, without repeating
earlier timing suites:

```sh
./script/test_synthetic_metal.sh --held-out-only
```

This records `requestedBenchmarks: "held-out-only"`, omits `mixedBenchmarks`, and
leaves the fixed-size arrays empty. It retains the correctness/safety gates.
`--mixed-only` omits the held-out section. Focused flags cannot be combined.

For a per-job breakdown of the held-out workload, including paired measurements
with instrumentation on and off:

```sh
./script/test_synthetic_metal.sh --job-profile-only
```

This records `requestedBenchmarks: "job-profile-only"` and `jobProfileBenchmarks`.
Earlier timing sections are skipped, but the correctness/safety gates remain.
Profiling is explicitly opt-in; a no-flag run still runs the four standard
sections without per-job clocks. No selector or application defaults change.

To compare the frozen selector with a separately named large/high-reuse
candidate, plus fixed CPU and Metal controls:

```sh
./script/test_synthetic_metal.sh --selector-candidate-only
```

This records `requestedBenchmarks: "selector-candidate-only"` and
`selectorCandidateBenchmarks`. The candidate retains the existing 512×512 rule,
adds Metal for exactly 640×512 at four/eight uses, and leaves one/two uses on
CPU. It exists only in this focused diagnostic; standard modes and application
settings still use the frozen rule.

To test a predeclared pixel-area candidate on independently shaped outputs just
below and above the 512×512 area boundary:

```sh
./script/test_synthetic_metal.sh --boundary-holdout-only
```

This records `requestedBenchmarks: "boundary-holdout-only"` and
`boundaryHoldoutBenchmarks`. It uses unseen 576×448 and 704×560 outputs at
one/two/four/eight uses. The former remains on CPU; the latter changes to Metal
only at four/eight uses. This candidate is diagnostic-only and does not change
the frozen selector, standard runs, or application behavior.

An optional directory keeps reports beside another set of test results:

```sh
./script/test_synthetic_metal.sh "/Volumes/New Volume/synthetic-metal-results"
```

Each invocation creates a new persistent run directory containing `report.json`,
`comparison.log`, and `summary.txt`. The wrapper builds in release mode and uses
an isolated, rebuildable compiler cache under `/private/tmp` to avoid the earlier
Finder-metadata signing problem. Reports do not live in that temporary cache.
Deleting or losing the cache only requires a rebuild. The script does not delete
old builds, reports, or any application work.

`PASS` requires both the executable and logging to succeed, plus a saved passing
report. A failed correctness gate stops further measurements. Incomplete runs,
including an interruption or missing report, must not be treated as validation
passes. Existing reports are never overwritten. Debug executables refuse timing.

## Correctness scope

- A separate Double-precision weighted-sum oracle checks the scalar, SIMD, and
  four-chunk SIMD Float CPU implementations and Metal's linear texture sampler.
- Integer pixel centres, distinct RGBA channels, corners, out-of-bounds clamping,
  and repeated identical GPU input are checked (625 positions).
- Twenty-four generated moving frames exercise fractional coordinates, rotation,
  checkerboard boundaries, and frame-to-frame differences (184,320 positions).
- All four inputs for every mode and timed size must also pass before timing
  starts, and are checked again afterwards. Resident and grouped outputs must
  match the upload-per-call outputs exactly.
- CPU maximum absolute error must be below `1e-6`. The Metal identity budget is
  `1e-5`; fractional sampling allows `1/255`, and temporal-difference error allows
  `2/255`. These are explicit normalized image-value budgets, not bit-exact parity.
- Forty-six CPU-only unit tests cover known bilinear values, clamping, channels,
  moving fixtures, non-finite rejection, timing-summary calculations, balanced
  execution order, per-image normalization, reuse work budgets, and consuming
  every output. The optimized CPU tests also cover alpha interpolation, malformed
  byte counts, dimension overflow, single-pixel sources, odd output counts,
  repeatability, and disjoint worker chunks covering every output exactly once.
  Mixed-workload tests cover fixed routing, complete work budgets, balanced
  policy order and input banks, invalid fixture dimensions, and focused options.
  Held-out tests add changing reuse, new-size CPU fallback, routing controls, and
  joint balance across policy position, phase, input banks, and job order.
  Profiling tests add nonoverlapping stage accounting, omitted CPU/GPU fields,
  partial GPU timestamp rejection, signed paired differences, and joint balance
  including which partner runs first.
  Candidate-selector tests prove its exact size/reuse boundary, unchanged
  one/two-use and original-policy routes, complete work counters, and balanced
  four-policy position/input/order scheduling with equal candidate/control
  precedence.
  Boundary-holdout tests prove the pixel-area/reuse boundary on independently
  shaped outputs, exact per-phase GPU work, unchanged original policy sets, and
  balanced four-policy scheduling with equal candidate/control precedence.
- Six runtime resident-input checks cover empty groups, uninitialized inputs,
  duplicate output slots, non-finite uploads, stale input after a failed upload,
  and recovery after a valid re-upload.

This package's pass does **not** resolve or supersede failures in the separate
application-level synthetic projection diagnostics.

## Measurement scope

The current report uses **schema version 9**, adding the opt-in
`boundaryHoldoutBenchmarks` object. Schema 8 added
`selectorCandidateBenchmarks`, while schema 7 added `jobProfileBenchmarks`,
while schema 6 added `heldOutBenchmarks` alongside the
schema-5 mixed section and explicit requested scope. The steady-state
and upload-inclusive measurements
retain the six modes introduced in schema 4. Historical schemas 1–3 below predate
the SIMD CPU modes. In the steady-state section, each size has
four warm-up rounds and 24 measured rounds. Every mode processes the same four distinct generated
frames in each round. Rotating order puts each of six modes in each position four
times during measurement. JSON contains raw batch and per-image samples,
medians, and P10–P90 intervals.

Per-image timings divide a whole four-output round by four: they are amortized
costs, **not single-image latency**. Earlier schema-1 reports used individual calls
with three warm-ups and 21 pairs, so don't interpret cross-version differences
as a performance improvement.

Sizes include the original 257×193 → 96×80 case, plus a constant 1024×1024 source
resampled to 96×80, 128×128, 256×256, 384×384, and 512×512. Keeping source size
constant in the latter five cases separates output-size effects from upload size.

- **CPU scalar:** the original serial Swift Float bilinear sampler, including
  output-array allocation, retained as a baseline.
- **CPU SIMD:** direct RGBA8 gathers and four-lane Float arithmetic on one thread.
  It retains the scalar sampler's coordinates, clamping, and float output format.
  Input validation and output-array allocation are inside the stopwatch. No
  preconverted source image or interpolation-plan cache is prepared off the clock.
- **CPU SIMD×4:** the same SIMD sampler split into up to four disjoint chunks of
  output pixels, using synchronous GCD work dispatch. Scheduling and joining are
  included. Four source images are still processed sequentially; this mode splits
  each image's output, rather than batching four CPU images concurrently. Four
  chunks do not guarantee four physical cores or thread affinity. Input pointers
  stay within their owners' lifetimes until all chunks finish; each output has
  exactly one writer. These CPU implementations are not Accelerate/vImage or a
  claim of the best possible CPU performance.
- **Metal upload:** input validation, uploads, command encoding/submission,
  completion wait, and copying the result into a Swift output array. Texture and
  buffer allocations are reused. The scalar CPU implementation does not perform
  the extra finite-coordinate validation on every timed call; both SIMD modes
  and Metal upload do. All timed inputs are prevalidated for every path.
- **Metal resident:** four independent texture/coordinate/output slots are
  allocated and uploaded once before timing; each output still requires a
  separate submission, completion wait, and array copy. Initial upload duration
  is reported separately. Upload counters must remain unchanged during timing.
- **Metal resident group4:** the same four resident inputs are sampled in one
  command buffer. Each has its own output buffer, and all four output arrays are
  copied back after completion. This is command grouping, not a larger model
  batch or four dependent iterations on one image.
- **Metal GPU-only:** command-buffer execution timestamps, not end-to-end time.
  Missing timestamps are omitted, never reported as zero. Separate-submission
  rounds require all four timestamps before a summed GPU duration is reported.
- Fixture generation, Double-oracle validation, and pipeline/resource creation
  are excluded from the timings. Pipeline setup is reported separately.
- Memory fields report Metal resource allocations and known host array payloads.
  The upload mode has one reusable slot; resident modes share four separate
  slots. The combined allocation field includes all five slots alive in the
  experiment, not five slots required by one backend.
  They exclude driver/compiler overhead, temporary validation arrays, and process
  peak RSS. They are not a whole-application memory limit. Shared-memory numbers
  should not be interpreted as separate VRAM and RAM pools.

### Upload-inclusive reuse experiment

The `reuseBenchmarks` section tests 96×80, 256×256, and 512×512 outputs from
1024×1024 sources. For each size it measures **1, 2, 4, and 8 uses per input**.
Each trial starts with four inputs, and alternates between two different
generated input sets. Every mode processes all `4 × uses` outputs.

- Resident modes **upload all four inputs inside the stopwatch at the start of
  every trial**, then sample each input the requested number of times. They do
  not receive free pre-uploaded inputs from the previous trial.
- Upload-per-call mode uploads before every output. CPU starts with the same
  prepared source arrays; there is no CPU upload step to charge.
- Every use copies outputs into CPU arrays and consumes their checksums. The
  checksum and lightweight workload counters are included in these timings;
  outputs are not discarded on the GPU. One four-output batch is retained at a
  time rather than retaining all 32 outputs for the eight-use case.
- Each case has four warm-up trials and 24 measured trials per mode, with
  rotating execution order. JSON includes complete trial times and per-output
  times, raw samples, medians, and P10–P90 intervals.
- Upload and actual command-submission counts must match the expected workload
  on every trial. Checksums must match the selected input set. Full-pixel oracle
  and Metal-mode comparisons run on **every use of both input sets**, before and
  after timing. The timed checksum is not a replacement for those pixel checks.
- Pipeline and buffer allocation still happen outside the stopwatch. This tests
  freshly uploaded data in an already initialized process, **not a cold launch**
  or uncached RAM. New setup costs must not be inferred from it.
- GPU resource counts stay at one upload slot plus four resident slots regardless
  of reuse count. Host fixtures, reference arrays, and output arrays are reported
  separately; none of these metrics claim to measure process peak RSS.

For example, at eight uses a trial returns 32 outputs. CPU submits nothing;
upload-per-call makes 32 uploads and 32 submissions; resident-separate makes
4 uploads and 32 submissions; resident-group4 makes 4 uploads and 8 submissions.
Per-output time divides the **entire measured trial, including uploads**, by 32.

### Mixed-size experiment

Three policies process identical, predeclared jobs: parallel CPU only, grouped
Metal only, and an **experimental fixed hybrid rule**. The rule selects Metal
only for exactly 512×512 outputs with at least four known uses per input, using
CPU otherwise. It was set from the preceding isolated M4 measurements, not
learned or adjusted from these mixed trials. It is not an application default
or a general size threshold; unknown sizes are not extrapolated to Metal.

Every source is 1024×1024 RGBA8, and each job has four source images. The
`mixed-reuse` sequence is:

| Job | Output size | Uses per input | Hybrid backend |
| --- | --- | --- | --- |
| 1 | 96×80 | 1 | CPU |
| 2 | 512×512 | 8 | Metal |
| 3 | 256×256 | 2 | CPU |
| 4 | 384×384 | 4 | CPU |
| 5 | 512×512 | 1 | CPU |
| 6 | 256×256 | 8 | CPU |
| 7 | 96×80 | 8 | CPU |
| 8 | 512×512 | 4 | Metal |

The second sequence, `fresh-every-job`, uses the same sizes but resets every
reuse count to one. The first returns **144 output arrays per trial**, the second
32. Every output must be allocated/copied and checksum-consumed; no outputs are
skipped. Only one four-output batch is retained at a time.

- The **whole trial** is timed, including backend selection, resource lookup,
  synchronous CPU worker scheduling, upload, command submission/wait, output
  allocation/copy, checksums, and workload counting. This is a serial dispatcher,
  not overlapped CPU/GPU execution or a runtime autotuner.
- Every Metal job uploads all four source/coordinate inputs inside the stopwatch,
  even if a previous job used that pool. Resident reuse exists only within one
  job. Source arrays are already available for CPU, just as in the prior tests.
- Two generated input banks alternate with trial and job index. These are not
  arbitrary external data, freshly decoded frames, or an uncached-RAM benchmark.
  Jobs run forward/reverse on alternating trials. Each policy sees exactly the
  same input/order combination per trial.
- Four warm-ups precede 24 measured trials per policy. Three-policy ordering
  rotates: each policy occupies each position eight times, and each position /
  input-order parity combination four times. Whole-trial raw samples, medians,
  and P10–P90 intervals are reported. Per-output numbers divide the whole trial
  by its output count; because sizes differ, these are **not single-image
  latency, model throughput, or video FPS**.
- Full-pixel Double-oracle validation runs on every use, for both input-bank
  assignments and job orders, before and after timing. CPU keeps its `1e-6`
  error budget, Metal `1/255`. Timed job checksums must exactly match their
  independently prevalidated backend/bank expectations. CPU and Metal need not
  be bit-identical. Error fields for an unused backend are omitted, not zero.
- Actual upload, command-submission, output-array, output-pixel, and route counts
  are checked on every trial. For the reuse mix, Metal-only makes 32 uploads /
  36 submissions; hybrid makes 8 / 12 and routes six jobs to CPU, two to Metal.
  For the fresh-input mix, hybrid routes all eight jobs to CPU with no GPU work.
- One four-slot GPU pool per distinct size is initialized outside timing and
  shared by the three policies. All these pools remain allocated during CPU-only
  measurement too. Fixture generation, oracle construction, resource allocation,
  and initial validation have a separate setup duration. There is no unbounded
  pool growth or cross-job cache hit credited to Metal.

This experiment tests only these two workload mixes on generated patterns.
Known reuse counts, all inputs being ready, and four outputs per job are explicit
assumptions. It does not validate backend switching quality in an application,
predict performance under unrelated CPU/GPU load, or establish a production rule.

### New sizes and changing reuse

The held-out section keeps the existing selector **unchanged**: exactly 512×512
with at least four known uses selects Metal; everything else selects CPU. Thus
all four new shapes use the hybrid's CPU fallback. The point is to expose this
rule's limitations against CPU-only and Metal-only, not to fit another threshold
to the same results.

The eight jobs have the following output sizes, in base order. Each job has four
1024×1024 source images. Reuse counts rotate by one position after each trial:

| Job | Output size | Phase 0 uses | Phase 1 | Phase 2 | Phase 3 |
| --- | --- | --- | --- | --- | --- |
| 1 | 192×128 (new) | 1 | 2 | 4 | 8 |
| 2 | 512×512 (control) | 2 | 4 | 8 | 1 |
| 3 | 320×256 (new) | 4 | 8 | 1 | 2 |
| 4 | 640×512 (new) | 8 | 1 | 2 | 4 |
| 5 | 448×384 (new) | 1 | 2 | 4 | 8 |
| 6 | 192×128 (new) | 2 | 4 | 8 | 1 |
| 7 | 512×512 (control) | 4 | 8 | 1 | 2 |
| 8 | 640×512 (new) | 8 | 1 | 2 | 4 |

Each phase returns 120 output arrays, but **total pixels differ by phase**. All
policies receive the same jobs, phase, bank assignment, and order in a trial.
Aggregate timing therefore includes a deliberately mixed workload distribution;
per-phase timing is reported separately to avoid hiding different amounts of
pixel work behind one median.

- The source banks use frame offsets 24 and 40, not the earlier 0 and 8. This
  still uses the same generated ramp/checkerboard family, not independent real
  images, a model-quality dataset, or a video.
- Sixteen warm-up trials cover four phases × two bank parities × two job orders.
  Forty-eight measured trials rotate three-policy order. For each policy, each
  policy-position / phase / bank-parity / forward-or-reverse combination occurs
  exactly once. Each phase has twelve measured samples; the aggregate has 48.
- Hybrid routes 1 / 2 / 1 / 0 jobs to Metal in phases 0 / 1 / 2 / 3, with
  4 / 12 / 8 / 0 submissions and 4 / 8 / 4 / 0 uploads. Metal-only always makes
  32 uploads and 30 submissions; CPU-only makes none. Actual arrays, pixels,
  checksums, uploads, submissions, and route counts are verified every trial.
- Schedule and plan selection are inside the whole-trial stopwatch, along with
  the existing routing, scheduling, upload, execution, and output costs. Reuse
  is known metadata, not estimated from future measurements. Every Metal job
  refreshes all four inputs. There is no cross-job reuse or CPU/GPU overlap.
- Every pixel on every use is validated before and after timing in all sixteen
  phase/bank/order scenarios, for every policy. CPU and Metal budgets stay at
  `1e-6` and `1/255`; backend switching is not claimed to be bit-exact. Checksums
  use the prevalidated backend-specific reference, not an averaged tolerance.
- One four-slot pool per shape is reused. No pool grows with trial count. The
  report records resource and host-array payloads; these are not peak RSS.
  CPU-only trials still coexist with the GPU comparison pools. Setup, fixture
  generation, Double oracles, and initial checks remain outside the timings.

Raw trials include their phase, bank/order parities, elapsed time, actual work,
and job checksums. GPU time is absent for CPU-only or unavailable-timestamp
trials, never filled with zero. Per-phase error fields are absent when that
backend was unused. The selector is not tuned after observing this test.

### Per-job instrumentation and overhead

The opt-in profiler reuses the held-out shapes, source banks, four reuse phases,
and **unchanged** selector. It records each job separately, including both
640×512 jobs at every reuse count. JSON and console use **zero-based** indices:
these are jobs 3 and 7 (rows 4 and 8 in the table above). Do not pool them as if
their position in the workload were identical.

- Sixteen warm-up **pairs** precede 96 measured pairs per policy. Each pair runs
  the same whole workload once without job clocks and once with them. Both
  partners refresh inputs at every Metal job and return/checksum every output.
  Three policy positions × four reuse phases × two input-bank assignments ×
  two job orders × two within-pair orders are jointly balanced once per policy.
  Each job/reuse phase has 24 profiled observations. These pairs are adjacent,
  not simultaneous, and are not statistically independent cold starts.
- Each job has an enclosing wall-clock measurement. Metal jobs report upload
  time (including shape/coordinate validation), host encoding/submission/wait
  time, and output-buffer-to-Swift-array copy time. CPU jobs report compute,
  output allocation, and worker scheduling together; there is no separate CPU
  upload or output-copy operation to time. Inapplicable fields are omitted.
- GPU execution timestamps are a separate, **overlapping subset** of Metal
  encoding/submission/wait. Never add GPU milliseconds to the host stages.
  A job's GPU total is omitted unless every submission has a valid timestamp;
  the available timestamp count is still reported.
- `otherHostMS` is the remaining job time: resource lookup/routing, checksum
  consumption, counters, output lifetime cleanup, and instrumentation. It is
  **not pure routing overhead**. `outsideJobsMS` is the remaining whole-trial
  time, including collection/validation of the profile records and other trial
  bookkeeping. Every enclosing-clock/stage sum is checked; negative residuals
  beyond one nanosecond fail instead of being silently hidden.
- Raw profiled-minus-plain differences and P10/median/P90 describe instrumentation
  **plus scheduling/cache noise**. Negative differences are preserved. This is
  not a calibrated timer cost; no overhead correction is subtracted from job
  stages. Plain partners still execute disabled instrumentation branches. Do
  not infer a speedup from comparing this binary with an earlier version.
- All 16 input/order/reuse scenarios and three policies retain full-pixel oracle
  gates before and after measurement. Additionally, profiled sampling must be
  bit-identical to plain sampling for every shape/bank/backend before and after
  timing (40 four-output comparisons total). Both partners' actual workloads
  and exact job checksums must match; full-pixel comparisons are off the clock.
- Whole-job and individual-stage summaries have raw samples and dispersion.
  Medians of separate stages do not necessarily add to median job time.
  Results preserve all jobs/phases in JSON; console output highlights the two
  640×512/eight-use jobs and whole-trial paired differences.

Fixtures, pools, and oracles are reused only as in the prior held-out experiment;
setup remains excluded, outputs remain CPU-readable, and there is no new
cross-job input cache, CPU/GPU concurrency, routing rule, or application change.

## Initial M4 results — schema 1, 2026-09-03

macOS 27 build `26A5425a`, Apple Swift `6.4.0.33.1`, two consecutive release runs.
Both passed. Identity/edge error was zero. Moving-pattern Metal error was
`0.002060354`; temporal-difference error was `0.003891468`. CPU error was at most
`1.1921e-7`. The larger benchmark input's Metal error was `0.002368867`, also
within the declared budget. Checksums were stable between runs, but CPU and Metal
checksums are not identical because their interpolation precision differs.

Times below are **median [P10–P90] milliseconds**, not projected application FPS:

| Source → output | Run | CPU | Metal total | GPU execution median |
| --- | --- | --- | --- | --- |
| 257×193 → 96×80 | 1 | 0.060 [0.056–0.068] | 0.438 [0.208–0.568] | 0.009 |
| 257×193 → 96×80 | 2 | 0.052 [0.052–0.060] | 0.219 [0.213–0.454] | 0.010 |
| 1024×1024 → 512×512 | 1 | 1.863 [1.830–1.909] | 1.039 [0.783–1.374] | 0.125 |
| 1024×1024 → 512×512 | 2 | 1.968 [1.876–2.013] | 1.002 [0.780–1.130] | 0.124 |

Reusable Metal allocations were 0.52 MiB and 10.03 MiB respectively. Pipeline
setup took 888 ms in the first run and 34 ms in the repeat; setup is not included
in the table. Do not treat this as a controlled cold-start benchmark.

The small image favored CPU once GPU handoff costs were counted. The larger
image favored this Metal implementation by about 1.8–2.0× versus this plain CPU
implementation. Two runs are preliminary evidence, not a hardware crossover
threshold or a reason to switch an application backend.

Verification also confirmed that the release executable rejects an existing
report without modifying its bytes. A direct debug invocation refused timing,
but macOS denied its failure-report write outside the build directory; that
negative reporting check is incomplete. No signing or privacy permissions were
changed to work around it. Both release-script report writes succeeded.

The follow-up schema-2 experiment adds the intermediate sizes and resident modes
described above. Repeated resident inputs are an explicitly favorable reuse case;
one-time upload cost still matters for workloads with little reuse. Compare with
an optimized CPU baseline before drawing broader performance conclusions.

## Resident-input results — schema 2, 2026-09-03

Two expanded M4 runs passed all six sizes, all six resident safety checks, and
the existing identity/motion gates. Seven unit tests also passed. Resident modes
made no additional uploads (`4 → 4`), and their outputs matched upload-per-call
outputs exactly before and after measurement. Maximum reference error across
the size sweep was `0.002604843`, below `1/255`.

The table shows the second run: **amortized median [P10–P90] ms per image**.
Every row returns all four output arrays; initial uploads are excluded from
resident timing, not silently charged to the CPU path.

| Source → output | CPU | Metal upload | Resident, separate submissions | Resident, group4 |
| --- | --- | --- | --- | --- |
| 257×193 → 96×80 | 0.057 [0.054–0.073] | 0.218 [0.204–0.316] | 0.200 [0.186–0.227] | 0.065 [0.061–0.119] |
| 1024×1024 → 96×80 | 0.075 [0.067–0.080] | 0.450 [0.442–0.904] | 0.234 [0.228–0.242] | 0.071 [0.068–0.074] |
| 1024×1024 → 128×128 | 0.155 [0.145–0.182] | 0.488 [0.466–0.579] | 0.250 [0.244–0.258] | 0.088 [0.085–0.093] |
| 1024×1024 → 256×256 | 0.520 [0.489–0.534] | 0.678 [0.611–0.785] | 0.324 [0.314–0.434] | 0.151 [0.145–0.157] |
| 1024×1024 → 384×384 | 1.109 [1.055–1.140] | 0.789 [0.681–1.408] | 0.390 [0.381–0.446] | 0.213 [0.205–0.236] |
| 1024×1024 → 512×512 | 1.954 [1.910–1.961] | 0.888 [0.776–1.383] | 0.473 [0.460–0.951] | 0.288 [0.276–0.319] |

At 512×512, the first run measured 0.912 / 0.481 / 0.300 ms for upload /
resident separate / resident group4, respectively. The second measured
0.888 / 0.473 / 0.288 ms: about a 3× difference between grouped resident and
upload-per-call medians in both runs. GPU-only times remain separate in JSON.

Four resident slots at this size allocate 40.12 MiB versus the upload slot's
10.03 MiB; both coexist during comparison, totaling 50.16 MiB of Metal resources.
These are **not process peak memory**. All four source uploads must still happen
at least once. In the first run that preparation cost 2.42 ms for the four-image
set, so the steady-state result is not a 3× promise for single-use inputs.

CPU remains attractive for tiny isolated operations. For the tested resident
reuse case, grouping submissions reduced overhead without changing sampled
pixels. There is no universal size threshold here: source dimensions, reuse,
output-copy costs, memory headroom, and timing dispersion all matter.

Recorded runs (relative to the repository root):

- `.test-results/synthetic-metal/run-20260903T092049Z-praLgQ/report.json`
- `.test-results/synthetic-metal/run-20260903T092201Z-qV9dfU/report.json`

No application backend, shader, model, or default was modified. A next standalone
comparison should use an optimized CPU baseline and explicitly account for
input-reuse frequency before making broader recommendations.

## Upload-inclusive results — schema 3, 2026-09-03

Two full runs passed the six steady-state sizes and all twelve upload-inclusive
cases. Ten unit tests passed. Every recorded upload/submission count matched its
workload, and all Metal modes matched upload-per-call pixels exactly. Maximum
reference error in the reuse cases was `0.002507091`, within the declared budget.

For 1024×1024 → 512×512, the second run measured the following **median [P10–P90]
ms per output, including the initial input uploads**:

| Uses per input | CPU | Metal upload each use | Resident, separate | Resident, group4 |
| --- | --- | --- | --- | --- |
| 1 | 2.124 [1.934–2.316] | 0.907 [0.868–1.285] | 0.946 [0.928–1.071] | 0.790 [0.746–1.068] |
| 2 | 2.112 [1.946–2.316] | 0.884 [0.815–1.286] | 0.728 [0.690–1.180] | 0.521 [0.506–0.590] |
| 4 | 2.127 [1.963–2.305] | 0.933 [0.821–1.013] | 0.618 [0.582–0.732] | 0.407 [0.389–0.504] |
| 8 | 2.119 [1.977–2.300] | 0.936 [0.844–0.963] | 0.574 [0.513–0.659] | 0.351 [0.332–0.392] |

The first run's grouped medians were 0.734 / 0.514 / 0.432 / 0.360 ms for
1 / 2 / 4 / 8 uses. Counting uploads reduced the apparent benefit at one use;
the one-use timing distributions overlap, so this is not a firm latency promise.
At eight uses, grouped mode was about 2.6–2.7× faster than upload-per-call by the
per-output medians in both runs. This remains a comparison of these particular
implementations, not an application speedup or a cold-launch measurement.

For 96×80 outputs, CPU won at every tested reuse count in both runs. In the
second run, even eight uses cost 0.113 ms/output in grouped mode versus CPU's
0.070 ms. For 256×256, grouped medians fell from 0.576 ms at one use to 0.232 ms
at eight uses; CPU measured 0.688 and 0.571 ms respectively. Individual samples
vary, and the fixtures alternate between two different input sets. Do not
hard-code a universal crossover point from these three tested sizes.

At 512×512, the resource budget remains 10.03 MiB for the upload slot and
40.12 MiB for four resident slots, irrespective of reuse count. The comparison
also retains 48 MiB of generated source/coordinate payloads and 64 MiB of
reference arrays; one four-output batch adds 16 MiB of array payload. These
component counts are not a measured process peak or separate VRAM/RAM pools.
The eight-use case does not retain all 32 output images together.

Recorded runs (relative to the repository root):

- `.test-results/synthetic-metal/run-20260903T094127Z-Dm1g4I/report.json`
- `.test-results/synthetic-metal/run-20260903T094349Z-y1ionL/report.json`

The schema-4 comparison below adds optimized CPU implementations under the same
sampling, output, and correctness requirements. No application integration or
automatic backend policy is included here.

## Optimized CPU results — schema 4, 2026-09-03

Two release runs on the same Apple M4 passed all six steady-state sizes and
twelve upload-inclusive cases, each with six backends and 24 measured rounds.
All fifteen unit tests passed. The five-test optimized CPU suite also passed
separate AddressSanitizer and ThreadSanitizer runs; these checks are not a proof
of memory or thread safety for every possible input. Sanitizers were not enabled
during timing. Both reports record ten active logical processors; four chunks
are requested, without assigning particular cores.

All three CPU implementations stayed within `1.1921e-7` of the Double oracle
across the measured fixtures, below the `1e-6` gate. Serial and parallel SIMD
results were identical in the repeated-motion unit test. Metal's existing
precision budgets and resident safety gates were unchanged and passed. Every
reuse upload/submission count matched the expected workload.

For 1024×1024 → 512×512, the second run measured the following **amortized median
[P10–P90] ms per output, including input upload and output allocation/copy**:

| Uses per input | CPU scalar | CPU SIMD serial | CPU SIMD×4 | Metal resident group4 |
| --- | --- | --- | --- | --- |
| 1 | 2.170 [1.962–2.388] | 1.435 [1.219–1.708] | 0.521 [0.424–0.682] | 0.738 [0.714–0.874] |
| 2 | 2.159 [2.001–2.382] | 1.451 [1.223–1.752] | 0.519 [0.409–0.655] | 0.516 [0.508–0.585] |
| 4 | 2.176 [1.991–2.308] | 1.474 [1.221–1.690] | 0.518 [0.407–0.582] | 0.402 [0.387–0.576] |
| 8 | 2.131 [1.993–2.287] | 1.433 [1.239–1.668] | 0.493 [0.425–0.574] | 0.384 [0.342–0.453] |

The first run's SIMD×4 medians were 0.531 / 0.514 / 0.516 / 0.523 ms and its
grouped Metal medians were 0.726 / 0.518 / 0.400 / 0.354 ms for 1 / 2 / 4 / 8
uses. The broad pattern repeated: SIMD×4 led for single-use inputs, the two-use
medians were effectively tied, and grouped Metal led by the four/eight-use
medians. Some P10–P90 intervals overlap; these are not significance tests or
latency guarantees. Even these optimized CPU modes do not represent an
exhaustive search of CPU implementations or scheduling strategies.

For 96×80 and 256×256 outputs, SIMD×4 had the lowest upload-inclusive median at
every tested reuse count in both runs. At 256×256 with eight uses, it measured
0.159 / 0.154 ms versus grouped Metal's 0.216 / 0.219 ms. The steady-state
256×256 comparison, which excludes initial resident uploads, was much closer:
SIMD×4 0.142 / 0.150 ms versus grouped Metal 0.148 / 0.145 ms. Thus a threshold
based on output dimensions alone would lose important reuse and upload context.

The new CPU modes allocate no Metal resources or prepared source caches. Output
arrays are allocated in every call, including inside the SIMD×4 measurements;
thread/runtime overhead is not captured by the existing resource-byte fields.
The standalone benchmark still retains Metal resources for its comparison modes.
It does not measure whole-process peak memory, concurrent application load, video
throughput, or model performance. No production setting should be inferred from
these microbenchmarks alone.

Recorded runs (relative to the repository root):

- `.test-results/synthetic-metal/run-20260903T101538Z-bYl5kH/report.json`
- `.test-results/synthetic-metal/run-20260903T101640Z-2FSzW3/report.json`

Application code, models, and defaults remain unchanged by this experiment.

## Mixed-size results — schema 5, 2026-09-03

Twenty-one unit tests passed, including six new mixed-plan/option tests. Two
focused release runs (`--mixed-only`) passed the identity/motion/resident gates
and both mixed workloads with all three policies. Their work counters, job
checksums, and reference errors matched between runs. The earlier fixed-size
timing suites were intentionally not rerun in these focused reports.

The table reports **whole-trial median [P10–P90] milliseconds**, including routing,
uploads, output arrays, and checksum consumption. Each policy has 24 measured
trials per row; these are not single-output or video-frame latencies.

| Workload | Run | CPU SIMD×4 | Metal group4 | Experimental hybrid |
| --- | --- | --- | --- | --- |
| Mixed reuse, 144 outputs | 1 | 40.057 [34.740–46.640] | 45.984 [44.036–49.138] | 33.843 [32.106–38.307] |
| Mixed reuse, 144 outputs | 2 | 40.536 [34.815–46.285] | 41.778 [40.213–43.287] | 32.975 [31.529–35.913] |
| Fresh every job, 32 outputs | 1 | 9.650 [8.955–10.559] | 22.267 [19.940–24.219] | 10.042 [9.140–10.933] |
| Fresh every job, 32 outputs | 2 | 10.001 [8.976–10.801] | 19.441 [19.186–20.057] | 9.660 [8.923–10.960] |

For this declared reuse-heavy mix, hybrid's median wall time was about 15–19%
below CPU-only and 21–26% below Metal-only. Its six CPU jobs and two Metal jobs
returned the same number of arrays/pixels, with 8 uploads and 12 GPU submissions
per trial. Some timing intervals overlap; these are preliminary microbenchmark
results, not statistical significance or a guaranteed application improvement.

For fresh inputs, hybrid correctly selected CPU for every job. CPU-only and
hybrid exchanged the lower median between runs, with heavily overlapping
intervals. This does **not** demonstrate a speed advantage from routing when
all jobs belong on CPU, nor isolate a precise routing overhead. Metal-only's
between-run variation is another reason not to promote a fixed threshold from
these measurements alone.

The four shape pools together allocate 108.75 MiB of Metal resources, and remain
alive for every comparison policy. Host payloads are 32 MiB of source arrays,
29.47 MiB of coordinates, and 58.94 MiB of oracle arrays; the largest four-output
batch adds 16 MiB. These figures exclude runtime/driver/compiler costs and are
not process peak memory. Setup including fixtures/oracles, pool creation, and
initial checks took 126.760 / 125.071 ms, excluded from trial timings.

Recorded runs (relative to the repository root):

- `.test-results/synthetic-metal/run-20260903T105317Z-uH9kl2/report.json`
- `.test-results/synthetic-metal/run-20260903T105415Z-groTPi/report.json`

Only this standalone diagnostic and its wrapper changed. No application backend,
model, or default was changed, and no video or model package was opened.

## Held-out results — schema 6, 2026-09-03

All 27 unit tests passed. Two focused release runs passed the new workload's
full-pixel checks before and after timing, covering all sixteen reuse / input-bank
/ job-order combinations for each policy. Both retained the standard identity,
moving-pattern, and resident-safety checks. Timed job checksums, counters,
schedule metadata, and reference errors matched between runs. Maximum reference
errors were `1.1921e-7` for CPU and `0.002603948` for Metal, within the unchanged
budgets. No selector or sampling kernel was adjusted after these measurements.

The second run's **whole-trial median [P10–P90] ms** by reuse phase is below.
Each cell has twelve measured trials. Every trial returns 120 arrays, but pixel
counts differ across phases: compare policies within a row, not processing rate
between different rows.

| Reuse phase | CPU SIMD×4 | Metal group4 | Frozen hybrid |
| --- | --- | --- | --- |
| 0 | 58.856 [56.737–62.225] | 49.249 [47.209–50.624] | 58.611 [56.294–61.711] |
| 1 | 43.035 [41.929–47.203] | 42.888 [42.064–43.685] | 36.053 [35.017–37.470] |
| 2 | 40.074 [38.013–42.092] | 41.080 [38.422–42.309] | 36.174 [33.966–37.571] |
| 3 | 43.488 [40.529–45.060] | 42.574 [40.597–43.680] | 43.780 [41.193–44.769] |

The first run showed the same important limitation: in phase 0, Metal-only took
50.324 ms versus hybrid's 57.537 ms. Phase 0 assigns eight uses to both new
640×512 jobs, which the frozen rule still sends to CPU. This is consistent with
a limitation of the conservative new-size fallback, but the experiment does
not isolate each job's latency or prove an optimal new threshold. In phases 1
and 2, hybrid had the lowest median in both runs. In phase 3 it performed no GPU
work, and the policy timing ranges overlapped substantially.

Across all 48 trials, the second run's aggregate medians were 43.398 / 42.811 /
39.862 ms for CPU / Metal / hybrid; the first measured 42.798 / 43.399 / 40.777 ms.
**The aggregate hybrid lead must not hide its slower phase 0.** This does not
establish a universally fastest selector, a production default, an application
quality gate, or a video throughput improvement. New frame offsets remain part
of the same generated pattern family, not a representative image dataset.

Resource accounting: five four-slot pools allocate 160.125 MiB of Metal
resources. Host payloads are 32 MiB of sources, 53 MiB of coordinates, and
106 MiB of oracle arrays; the largest returned batch adds 20 MiB. All policies
coexist with these comparison pools. These are not process peak memory or
separate RAM/VRAM pools. Setup took 200.350 / 203.768 ms and remains excluded from
trial timings.

Recorded runs (relative to the repository root):

- `.test-results/synthetic-metal/run-20260903T111814Z-JvS6ck/report.json`
- `.test-results/synthetic-metal/run-20260903T112006Z-i9oYn5/report.json`

An additional `--mixed-only` compatibility run passed after the shared harness
change; its checksums, reference errors, and work counters matched the earlier
schema-5 run. It is recorded at
`.test-results/synthetic-metal/run-20260903T112131Z-XwooGc/report.json`.
This was a correctness/compatibility check, not a controlled before/after speed
comparison. The original fixed-size timing suites were not rerun for these results.
Only the standalone synthetic package and wrapper changed; application code,
models, and defaults remain untouched by this experiment.

## Per-job results — schema 7, 2026-09-03

All 35 unit tests passed. Two focused release runs passed identity, moving-pattern,
resident safety, all sixteen held-out scenarios before/after timing, and all 40
profiled/plain four-output pixel-parity comparisons. Each run measured 96 pairs
per policy, with 24 samples per job/reuse phase. Recorded work, checksums,
schedules, resource accounting, and reference errors matched between runs.
Reference errors stayed at `1.1921e-7` CPU and `0.002603948` Metal. Independent
JSON checks confirmed complete sample counts and nonoverlapping host accounting.

The 640×512/eight-use cases now have direct job measurements. The second run's
**profiled whole-job median [P10–P90] milliseconds** are below. Each job returns
32 CPU-readable output arrays (four inputs × eight uses), not one video frame.
Indices are zero-based, and both jobs are kept separate:

| Job | CPU-only policy | Metal-only policy | Frozen hybrid (CPU for these jobs) |
| --- | --- | --- | --- |
| 3 | 20.074 [17.836–23.990] | 13.074 [12.189–17.689] | 20.507 [18.417–22.749] |
| 7 | 20.302 [17.180–24.098] | 12.970 [12.217–16.294] | 20.131 [18.095–23.206] |

The first run measured CPU/Metal medians of 20.731/14.408 ms for job 3 and
21.604/13.444 ms for job 7. Metal's lower median at eight uses persisted, but
these are generated-pattern measurements, not a production threshold or a
proven application speedup. No routing rule was changed.

Second-run Metal host-stage medians, **milliseconds for the entire job**:

| Job | Input upload/validation | Encode/submit/wait | CPU output-array copy | Other host | GPU subset (overlaps wait) |
| --- | --- | --- | --- | --- | --- |
| 3 | 1.723 | 7.478 | 3.806 | 0.023 | 4.131 |
| 7 | 1.718 | 7.472 | 3.793 | 0.021 | 4.056 |

Upload is not the largest stage at this reuse count. Output copying and the
combined encoding/submission/wait interval are substantial; this measurement
does not split encoding, scheduling, and waiting from one another. GPU timestamps
overlap the host interval and are **not added** to it. Stage medians need not sum
to median whole-job time. CPU compute/allocation medians were 20.064/20.292 ms;
there is no invented CPU copy stage.

Whole-trial paired differences (profiled minus plain) expose considerable noise.
For the second run their **median [P10–P90] ms** were CPU −0.440 [−3.472–2.512],
Metal +0.501 [−5.600–4.872], and hybrid −0.052 [−3.888–3.622]. First-run medians
were +0.227/+0.101/−0.089 ms. Negative values are real recorded differences,
not negative execution costs. The spread does not support claiming a precise
timer overhead or subtracting a fixed correction from stage measurements.

The same five four-slot GPU pools use 160.125 MiB; host payloads remain 32 MiB
of sources, 53 MiB of coordinates, and 106 MiB of oracles. Timed trials keep at
most one returned 20 MiB four-output batch; off-clock parity checks briefly hold
both plain and profiled outputs. Profiles retain numbers, not output-image
histories. Resource figures exclude temporary validation arrays and driver/runtime
overhead and are not process peak RSS.
Fixture/pool/oracle setup took 225.612/215.257 ms, excluded from trial timings.

Recorded profiling runs (relative to the repository root):

- `.test-results/synthetic-metal/run-20260903T155840Z-6U6AUl/report.json`
- `.test-results/synthetic-metal/run-20260903T160010Z-fwIwWP/report.json`

The existing unprofiled `--held-out-only` mode also passed after this change:
`.test-results/synthetic-metal/run-20260903T160154Z-d5dSSo/report.json`.
Its schedule, work counters, checksums, and reference errors matched the previous
schema-6 held-out run. This was a compatibility check, not a before/after speed
comparison. The fixed-size and original mixed timing sections were not rerun.

This is diagnostic instrumentation, not an application optimization. A follow-up
could compare a separately named experimental selector for large, highly reused
synthetic jobs, retaining this frozen rule as the control and fresh-input cases
as regression checks. These measurements alone do not justify generalizing to
all larger shapes, removing CPU-readable outputs, or changing application defaults.

## Large/high-reuse selector candidate — schema 8, 2026-09-04

All 40 unit tests passed. Three fresh release runs passed identity, motion,
resident-safety, pre/post full-pixel validation across all sixteen phase/bank/job
orders and four policies, and every timed workload/checksum gate. CPU/Metal
reference errors stayed at `1.1921e-7` / `0.002603948`. The original
`MixedPolicy.allCases` remains exactly CPU/Metal/frozen-hybrid; this candidate is
available only through `--selector-candidate-only`.

The focused candidate preserves the frozen 512×512-at-four-or-more route and
adds Metal only for exactly 640×512 with four or eight uses. One/two-use jobs
stay on CPU. Each run measured 64 trials/policy and 16 trials/reuse phase.
Four policy positions × four phases × two banks × two job orders are jointly
balanced once per policy. Candidate runs before and after the control equally
often. Candidate-minus-control is a same-round signed pair; negative is faster.

| 640×512 reuse | Run | Frozen control median | Candidate median | Paired difference median [P10–P90] | Candidate faster |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 uses | 1 | 62.364 ms | 52.972 ms | −9.234 [−13.389–−6.432] ms | 16/16 |
| 8 uses | 2 | 59.775 ms | 48.397 ms | −11.076 [−15.317–−5.001] ms | 16/16 |
| 8 uses | 3 | 56.544 ms | 44.921 ms | −11.200 [−18.918–−7.912] ms | 16/16 |
| 4 uses | 1 | 46.240 ms | 44.180 ms | −3.066 [−8.042–1.747] ms | 13/16 |
| 4 uses | 2 | 42.939 ms | 39.849 ms | −4.202 [−9.512–−0.731] ms | 14/16 |
| 4 uses | 3 | 43.117 ms | 38.714 ms | −4.265 [−6.761–−1.889] ms | 16/16 |

The eight-use improvement was present in every pair and its P10–P90 interval
stayed below zero in all three runs. Four uses improved 43/48 pairs; the first run's
interval crossed zero, so this boundary is promising but less decisive. In the
unchanged one/two-use phases, paired medians stayed around noise: −0.187/+0.411
ms at one use and +1.151/−0.094 ms at two uses. This is expected because control
and candidate use identical routes there; it is also a check against attributing
all timing variation to the routing change.

Across deliberately mixed phases, candidate-minus-control medians were −1.240,
−1.707, and −1.629 ms, with the candidate faster in 43/64, 46/64, and 42/64
trials. Aggregate
medians mix different pixel workloads and should not replace the phase evidence.
The useful result is concentrated where the candidate changes routing.

The candidate adds eight uploads and sixteen submissions versus control in the
eight-use phase, and eight uploads/eight submissions in the four-use phase, while
returning the same 120 CPU-readable arrays and exact output-pixel count. Setup,
fixture generation, pools, oracles, and full-pixel validation remain outside
timing. There is no CPU/GPU overlap or cross-job reuse.

Recorded valid runs (relative to the repository root):

- `.test-results/synthetic-metal/run-20260904T081247Z-GTMRN5/report.json`
- `.test-results/synthetic-metal/run-20260904T081355Z-9exC3R/report.json`
- `.test-results/synthetic-metal/run-20260904T083035Z-2XW5CY/report.json`

The existing frozen `--held-out-only` mode also passed after the shared routing
interface change at
`.test-results/synthetic-metal/run-20260904T081628Z-q1Uw26/report.json`.
Its schedules, work counters, checksums, routes, and reference errors matched the
previous schema-7 compatibility run. This was not a timing-speed comparison.

This validates the candidate only for generated bilinear-sampling workloads on
this Apple M4. It does not prove video/model throughput, projection quality, a
universal size threshold, or an application change. The next safe step would be
another independently shaped synthetic holdout near this boundary—not promotion
to production code.

## Pixel-area boundary holdout — schema 9, 2026-09-04

All 46 unit tests passed. Three release runs passed the identity, motion,
resident-safety, pre/post full-pixel, exact checksum, and complete workload
gates. The holdout used new frame offsets 56/72 and two unseen rectangular
outputs: 576×448 (258,048 pixels, below 512×512) and 704×560 (394,240 pixels,
above it). The diagnostic candidate preserves the frozen rule, keeps the smaller
shape on CPU at every reuse count, and sends the larger shape to Metal only at
four/eight uses.

Each phase returned 120 CPU-readable arrays. Four policy positions × four reuse
phases × two source banks × two job orders were jointly balanced once per policy
across 64 measured trials; candidate ran before and after control 32 times each.
The two eight-use phases and two four-use phases each had 16 paired samples/run.

| Larger-shape reuse | Run | Paired candidate−control median [P10–P90] | Candidate faster |
| --- | --- | ---: | ---: |
| 8 uses, phase 0 | 1 | −9.548 [−12.330–−3.608] ms | 16/16 |
| 8 uses, phase 2 | 1 | −8.936 [−10.825–−5.606] ms | 16/16 |
| 8 uses, phase 0 | 2 | −8.876 [−14.691–−4.965] ms | 16/16 |
| 8 uses, phase 2 | 2 | −8.308 [−13.135–−4.239] ms | 16/16 |
| 4 uses, phase 1 | 1 | −1.992 [−4.512–1.842] ms | 10/16 |
| 4 uses, phase 3 | 1 | −3.145 [−4.944–0.405] ms | 13/16 |
| 4 uses, phase 1 | 2 | −3.106 [−5.763–0.475] ms | 14/16 |
| 4 uses, phase 3 | 2 | −2.714 [−4.465–0.712] ms | 13/16 |
| 8 uses, phase 0 | 3 | −9.105 [−12.616–−6.945] ms | 16/16 |
| 8 uses, phase 2 | 3 | −8.733 [−13.712–−6.003] ms | 16/16 |
| 4 uses, phase 1 | 3 | −3.861 [−9.815–−1.445] ms | 15/16 |
| 4 uses, phase 3 | 3 | −3.975 [−8.296–−1.624] ms | 15/16 |

The eight-use route won **96/96** pairs and every P10–P90 interval stayed below
zero. The four-use route won 80/96 pairs and improved every phase median. Its
first four P90 values crossed zero, while both third-run intervals stayed below
zero. This supports the area-based Metal route at four/eight uses for the
generated workload, but not an application default: production crop extraction
is fixed at 256×256, below this candidate's area boundary.

Whole mixed-trial candidate-minus-control medians were −4.528, −4.784, and
−6.945 ms, with 55/64, 59/64, and 62/64 paired wins. These aggregates include different pixel work
across phases, so the phase evidence above is the decision-quality result.

Recorded valid runs (relative to the repository root):

- `.test-results/synthetic-metal/run-20260904T084638Z-spmn4z/report.json`
- `.test-results/synthetic-metal/run-20260904T084748Z-t8a7dK/report.json`
- `.test-results/synthetic-metal/run-20260904T085531Z-MkEVGc/report.json`

This remains generated bilinear sampling on one Apple M4. It does not establish
model/video throughput, restoration quality, a universal pixel threshold, or an
application routing change. The production selector and every standard
diagnostic mode remain unchanged.
