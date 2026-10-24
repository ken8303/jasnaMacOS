#!/usr/bin/env python3
"""Extract three paired SBS samples for a bounded detector-only trial."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=0, help='first sample time in seconds')
    parser.add_argument('--span', type=float, default=30, help='seconds between first and last sample')
    args = parser.parse_args()
    if not np.isfinite([args.start, args.span]).all() or args.start < 0 or args.span <= 0:
        parser.error('start must be nonnegative and span must be positive and finite')
    if not args.video.is_file():
        parser.error('video does not exist')
    # A new directory prevents comparisons against stale predictions from another clip.
    # Delay creation until decoding succeeds so an invalid time range remains retryable.
    if args.output.exists():
        parser.error('output directory already exists')
    cap = cv2.VideoCapture(str(args.video))
    images, samples = [], []
    try:
        if not cap.isOpened():
            raise ValueError('cannot open video')
        for seconds in (args.start, args.start + args.span / 2, args.start + args.span):
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'cannot decode sample at {seconds:g}s; choose a shorter span')
            width = frame.shape[1]
            if width % 2:
                raise ValueError('SBS video width must be even')
            for eye, crop in enumerate((frame[:, :width//2], frame[:, width//2:])):
                rgb = cv2.cvtColor(cv2.resize(crop, (768, 768)), cv2.COLOR_BGR2RGB)
                image = rgb.astype(np.float32) / 255
                image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
                images.append(image)
                samples.append(dict(video=str(args.video.resolve()), seconds=seconds, eye=eye))
    finally:
        cap.release()
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / 'inputs.npz', images=np.stack(images))
    (args.output / 'samples.json').write_text(json.dumps(samples, indent=2)+'\n')
    print(f'Saved {len(images)} eye images to {args.output}')


if __name__ == '__main__':
    main()
