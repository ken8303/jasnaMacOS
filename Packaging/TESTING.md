# Jasna VR Restoration — test build

This build is for testing on an Apple-silicon Mac running macOS 27.

The app includes the Jasna Metal ML restoration models, batch-2 models,
deformable-convolution weights, RF-DETR VR detector model, restoration engine,
workflow scripts, the prepared RF-DETR Python packages, a private Python
runtime, FFmpeg, FFprobe, and their required non-system libraries. No Homebrew
installation is required on the testing Mac.

New restorations use the measured faster BasicVSR++ batch-2 profile with eight
Metal windows per process in Fast mode and serial direct-SBS compositing. Compatible
interrupted work keeps its previously recorded batch setting so resume remains
safe. Restore-only comparisons replay the original crop, mask, cache, and
compositor settings unless the comparison explicitly overrides one setting.

This is an ad-hoc signed development build, not a notarized public release.
After transferring it to another Mac, right-click the app and choose **Open**.
If macOS still blocks it, open System Settings → Privacy & Security and choose
**Open Anyway** for Jasna VR Restoration.

Drop one or more side-by-side videos into Source Videos, or use Add Videos.
They run one at a time. Results are saved beside each source with `_restored`
added to the filename, using MOV for MOV sources and MP4 otherwise. Leave the
mosaic time field empty to scan every video, or enter ranges such as `11:00-30:00`.

If a different model or setting conflicts with saved work, choose **Start Fresh**.
The affected video's existing output and saved work are moved to a unique
`filename_restored.jasna-previous-*` folder before restarting. Other queued videos
keep their saved work. The backup path appears in the log. If this archive step
is interrupted, choose Start Fresh again to finish it safely. For an app-only
update with matching settings, **Resume Saved Work** reuses the compatible work.

Logs and resumable work files are written beside the selected output, not into
the application bundle.

See `THIRD_PARTY_NOTICES.md` inside the app resources before redistributing this
private test package.
