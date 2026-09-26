# Jasna VR Restoration — test build

This build is for testing on an Apple-silicon Mac running macOS 27.

The app includes the Jasna Metal ML restoration models, batch-2 models,
deformable-convolution weights, RF-DETR VR detector model, restoration engine,
workflow scripts, the prepared RF-DETR Python packages, a private Python
runtime, FFmpeg, FFprobe, and their required non-system libraries. No Homebrew
installation is required on the testing Mac.

New restorations use the measured faster BasicVSR++ batch-2 profile with two
Metal windows per process and serial direct-SBS compositing. Compatible
interrupted work keeps its previously recorded batch setting so resume remains
safe. Restore-only comparisons replay the original crop, mask, cache, and
compositor settings unless the comparison explicitly overrides one setting.

This is an ad-hoc signed development build, not a notarized public release.
After transferring it to another Mac, right-click the app and choose **Open**.
If macOS still blocks it, open System Settings → Privacy & Security and choose
**Open Anyway** for Jasna VR Restoration.

Select an 8K side-by-side source video and an MP4 output. Leave the mosaic time
field empty to scan the full video, or enter ranges such as `11:00-30:00`.

Logs and resumable work files are written beside the selected output, not into
the application bundle.

See `THIRD_PARTY_NOTICES.md` inside the app resources before redistributing this
private test package.
