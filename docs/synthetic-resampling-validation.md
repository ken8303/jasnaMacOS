# Synthetic resampling check — 2026-09-02

Scope: generated gradients and a smooth moving checkerboard only. No video
files, detector, model packages, neural inference or runtime-setting changes.
Tests live in `Tests/JasnaMetalPoCTests/SyntheticResamplingTests.swift` and
`SyntheticProjectionEdgeTests.swift` in that same test directory.

## Result

The fisheye coordinate pair originally had a reproducible pixel-centre
mismatch. It was corrected on 2026-09-08 by using the same pixel-centred
conversion in both directions: `normalized * size - 0.5`. Production crop
caches were moved from `crop-v4` to `crop-v5` so misaligned cached results
cannot be reused.

| Generated fixture | Largest absolute single-axis error |
| --- | ---: |
| Raw moving ramp, 4096-pixel canvas | 0.000031 source pixels |
| Fisheye moving ramp, 512-pixel canvas | 0.000176 source pixels |
| Fisheye moving ramp, 4096-pixel canvas | 0.001415 source pixels |
| Independent pixel-centred reference, 4096-pixel canvas | 0.001392 source pixels |

The ramp check translates a crop across 12 positions, checking 63 interior
points at each position. The model grid is 256×256. Canvas and crop geometry
scale together; no 4K raster allocation is needed. Bilinear interpolation of
a linear ramp expresses the error directly in source pixels.

`FisheyeMosaicCropTransform` now places model samples at
`(index + 0.5) / size` and converts normalized coordinates back with
`normalized * size - 0.5`. The source-image and model-grid conversions are
therefore inverses under the same pixel-centre convention.

The largest frame-to-frame change in alignment error is below 0.00071 source
pixels for the 4096-pixel fixture.

## Extraction and motion control

A separate 256×256 generated BGRA raster moves three pixels horizontally and
two vertically per frame over 12 frames. Red/green ramps and a 64-pixel-period
smooth checkerboard are sampled onto a 64×64 grid through the actual CPU
extraction path, for raw and fisheye projection. Each projection now tests
top-left, centre and bottom-right crops, including 1,013 boundary-clamped
fisheye samples in each corner crop. The double-precision oracle
interpolates the quantized source bytes independently at the sampling map's
coordinates. Thus this check tests byte layout, interpolation, half precision
and successive frame contents, not the map's geometric correctness.

Both projections pass:

- Maximum normalized sample error: 0.000244171 or less.
- Maximum frame-difference error: 0.000486247 or less.
- Bounds are 0.00025 and 0.0005 respectively, allowing Float16 rounding.

The smooth pattern avoids treating expected downsampling/aliasing loss as a
bug. This does not test a neural model's temporal recurrence or benchmark
production throughput.

## Pixel-centre reference: border and motion follow-up

These are **reference-only** checks. They do not modify the running application's
transform or prove production performance/quality has improved.

- Ten closed-form equator/central-meridian anchors check absolute position,
  orientation and scale, rather than relying on two functions agreeing with
  each other. At the exact poles longitude is undefined; only latitude is
  asserted on the inverse. Ordinary first/last-row pixel centres are tested
  separately and remain invertible.
- At each of 512- and 4096-pixel canvas sizes, 780 border and near-border
  positions (including fractional coordinates and all corners) round-trip
  analytically. Maximum error is below 0.000000002 source pixels. No raster
  interpolation occurs in this particular check.
- A 60-frame ramp sequence moves by 0.25/0.125 source pixels per frame while
  the crop origin and odd/even dimensions change five times. This check **does**
  use discrete bilinear resampling. Maximum position error is 0.000891 pixels;
  maximum frame-difference error is 0.000582 pixels, both below a 0.01-pixel
  bound. All samples remain inside the patch to avoid confusing clipping with
  coordinate alignment.

The reference stays consistent on these fixtures, and the production fisheye
transform now passes the alignment checks. Resampling at a spherical pole,
out-of-hemisphere samples, arbitrary high-frequency images and neural temporal
behaviour are not certified by these checks.

## Reproduce

### One-command strict diagnostic (2026-09-03)

From the package directory, run:

```sh
./script/test_synthetic_resampling.sh
```

Or select a persistent results folder:

```sh
./script/test_synthetic_resampling.sh "/path/to/SyntheticChecks"
```

This accepts a **results directory, not a video file**. No detector, model
loading, restoration or GPU work is invoked. The default results directory is
`.test-results/synthetic-resampling` under the package. Each run has its own
timestamped `run-*` directory with `synthetic-tests.log` and `summary.txt`.
Earlier runs are not overwritten. The rebuildable compiler and module cache
now uses `/private/tmp/jasna-synthetic-build-<user-id>-<package-id>`, a stable,
private per-user/per-package directory outside the file-provider-managed
Documents tree. SwiftPM's auxiliary cache/configuration and temporary files
remain beside the results. `JASNA_SYNTHETIC_BUILD_DIR` optionally selects a
different compiler cache. The runner does not delete existing caches. If the
OS clears `/private/tmp`, only recompilation is needed: saved logs and summaries
remain in the results folder.

The runner sets test-only `JASNA_SYNTHETIC_STRICT_ALIGNMENT=1` and requires
completion markers for all 13 parameterized cases, preventing
a zero-tests-selected result from being treated as a pass.

- Exit 0 / `PASS`: all selected assertions pass and all cases are present.
- Exit 1 / `FAIL`: all cases completed, but Swift reported a test failure.
- Exit 2 / `INCOMPLETE`: setup/build/execution/logging failed or cases are missing.
- `RUNNING` left in a summary means the process did not finalize; it is not PASS.

The corrected production transform passes the 512- and 4096-pixel fisheye
alignment assertions. This runner is a synthetic correctness check, not a
readiness or performance certificate for restoration.

Fourteen isolated runner tests cover pass/fail status, empty selection, failed
builds, logging failure, strict-mode enforcement, paths with spaces,
repeat-run preservation, argument handling, external-cache reuse, symlink
rejection, targeted metadata repair, bounded retry and independent retry logs.
Run them without Swift, models or media processing:

```sh
python3 -m unittest discover -s tools -p test_synthetic_resampling_runner.py -v
```

The real strict run on 2026-09-03 completed all 13 cases with `FAIL`, Swift exit
1, logging exit 0, and zero missing cases. Only the two previously documented
fisheye alignment assertions failed. An earlier restricted run stopped in
Swift's test helper before test execution and was correctly saved as
`INCOMPLETE`; rerunning with the needed local process permission resolved that
setup problem. The runner now selects Swift Testing only, not XCTest.

### Signing repair verified on 2026-09-03

The user's output-local build failed before testing because both generated
`.xctest` bundle roots carried `com.apple.FinderInfo`. The bundles were unsigned
at that point; this was a local ad-hoc signing failure, not a certificate,
provisioning, entitlement or notarization problem. File-provider metadata was
also present. Clearing the two FinderInfo attributes allowed signing and test
execution, but subsequent inspection showed the attributes had reappeared.
That made cleanup alone unsuitable as the default fix.

The new default compiler-cache location described above built successfully on
the first attempt. Both generated test bundles subsequently passed
`codesign --verify --strict`. All 13 synthetic cases completed with zero missing
cases; only the two existing alignment assertions failed. This remains `FAIL`
for geometry, but is no longer `INCOMPLETE` from signing.

For a custom cache, the runner also supports **one** retry after this exact
signing error. It removes only FinderInfo/resource-fork attributes, if present,
from the roots of `JasnaAppSupportTests.xctest` and `JasnaMetalPoCTests.xctest`
under the selected cache's `out/Products/Debug`. It refuses symlinked paths.
It does not recursively strip attributes, remove quarantine, change entitlements,
disable signing or touch source files/models. Each attempt has its own log;
completion markers are checked against the final attempt only. Recurring
metadata inside the bundles is not automatically cleared.

The previous output-local build cache and all prior reports were left intact.

### Development-suite behaviour

The development suite verifies the corrected alignment directly:

```sh
CLANG_MODULE_CACHE_PATH=/private/tmp/jasna-codex-test-build/ModuleCache \
SWIFTPM_MODULECACHE_OVERRIDE=/private/tmp/jasna-codex-test-build/ModuleCache \
swift test --disable-sandbox \
  --scratch-path /private/tmp/jasna-codex-test-build \
  --filter 'syntheticReference|syntheticMovingRawRampKeepsPixelCentres|syntheticMovingFisheyeRampKeepsPixelCentres|syntheticMovingDetailExtractionMatchesBilinearOracle'
```

The tests build and run without Metal-compatible buffers or external media
access. The fisheye ramp passes its original 0.05-pixel bound at both canvas
sizes; the tolerance was not loosened and the previous `withKnownIssue`
exemption was removed.

Parameterized cases are additional executions within these functions;
extraction alone runs six projection/crop-position combinations.

The common pixel-centre reference has now passed the independent anchors,
border and motion fixtures above. Keep it test-only: any future generic
resampling change needs its own compatibility and image-quality review, and
must not treat this report as neural-inference validation.
