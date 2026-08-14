#!/usr/bin/env bash

# Shared, read-only identity helpers for resumable restoration launchers.
# The caller is expected to enable its own strict shell options.

jasna_hash_lines() {
  /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'
}

jasna_content_fingerprint() {
  local path file
  {
    for path in "$@"; do
      if [[ -f "$path" ]]; then
        /usr/bin/shasum -a 256 "$path"
      elif [[ -d "$path" ]]; then
        while IFS= read -r file; do
          /usr/bin/shasum -a 256 "$file"
        done < <(/usr/bin/find "$path" -type f -print | LC_ALL=C /usr/bin/sort)
      else
        printf 'missing  %s\n' "$path"
      fi
    done
  } | LC_ALL=C /usr/bin/sort | jasna_hash_lines
}

jasna_metadata_fingerprint() {
  local path file
  {
    for path in "$@"; do
      if [[ -f "$path" ]]; then
        /usr/bin/stat -f '%N|%z|%m' "$path"
      elif [[ -d "$path" ]]; then
        while IFS= read -r file; do
          /usr/bin/stat -f '%N|%z|%m' "$file"
        done < <(/usr/bin/find "$path" -type f -print | LC_ALL=C /usr/bin/sort)
      else
        printf 'missing|%s\n' "$path"
      fi
    done
  } | LC_ALL=C /usr/bin/sort | jasna_hash_lines
}

jasna_source_fingerprint() {
  local source_path="$1"
  {
    printf 'path=%s\n' "$source_path"
    /usr/bin/stat -f 'size=%z|modified=%m' "$source_path"
  } | jasna_hash_lines
}

jasna_implementation_fingerprint() {
  local root_dir="$1"
  jasna_content_fingerprint \
    "$root_dir/Package.swift" \
    "$root_dir/Sources/JasnaMetalPoC" \
    "$root_dir/script/build_and_run.sh" \
    "$root_dir/script/restoration_identity.sh" \
    "$root_dir/script/restore_vr_eye_segments.sh" \
    "$root_dir/script/restore_vr_eye_sparse.sh" \
    "$root_dir/script/restore_vr_sparse_sbs.sh" \
    "$root_dir/script/scan_mosaic_regions.sh" \
    "$root_dir/script/test_vr_sparse_30s.sh" \
    "$root_dir/tools/manifest_window_runs.py" \
    "$root_dir/tools/rfdetr_mps_detector.py" \
    "$root_dir/tools/scan_mosaic_regions.py" \
    "$root_dir/tools/reconcile_stereo_manifests.py"
}

jasna_model_fingerprint() {
  local root_dir="$1"
  local detector="$2"
  local detector_model
  case "$detector" in
    rfdetr-vr-v1)
      detector_model="$root_dir/Models/MosaicDetection/rfdetr-vr-v1.pt"
      ;;
    yolo-v2-fast)
      detector_model="$root_dir/Models/MosaicDetection/lada_vr_mosaic_detection_model_v2_fast.pt"
      ;;
    *)
      detector_model="$root_dir/Models/MosaicDetection/unknown-$detector"
      ;;
  esac
  local detector_hash
  detector_hash="$(jasna_content_fingerprint "$detector_model")"
  {
    printf 'detector=%s\n' "$detector_hash"
    printf 'restoration=%s\n' "$(jasna_metadata_fingerprint \
      "$root_dir/Models/MetalML" \
      "$root_dir/Models/MetalMLBatch2" \
      "$root_dir/Models/DeformConv")"
  } | jasna_hash_lines
}
