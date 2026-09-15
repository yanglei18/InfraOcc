#!/usr/bin/env python3
"""Crop OCC gts labels from 320x320x32 to a lower z range.

Default conversion:
  input : ./data/v2xreal_nuscenes/gts
  output: ./data/v2xreal_nuscenes/gts_16

The default source range is [-64, -64, -4.8, 64, 64, 8.0].
The default target range is [-64, -64, -4.8, 64, 64, 1.6],
which keeps z bins [0, 16) for voxel_size=0.4.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def parse_range(text):
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) != 6:
        raise ValueError(f"pc range must have 6 values, got {len(values)}: {text}")
    return np.asarray(values, dtype=np.float64)


def occ_size(pc_range, voxel_size):
    size = (pc_range[3:] - pc_range[:3]) / float(voxel_size)
    rounded = np.rint(size).astype(np.int64)
    if not np.allclose(size, rounded, atol=1e-6):
        raise ValueError(
            f"pc_range {pc_range.tolist()} is not aligned to voxel_size={voxel_size}"
        )
    return tuple(int(v) for v in rounded)


def crop_slices(src_range, dst_range, voxel_size):
    if np.any(dst_range[:3] < src_range[:3] - 1e-6) or np.any(dst_range[3:] > src_range[3:] + 1e-6):
        raise ValueError("target pc range must be inside source pc range")

    start_float = (dst_range[:3] - src_range[:3]) / float(voxel_size)
    end_float = (dst_range[3:] - src_range[:3]) / float(voxel_size)
    start = np.rint(start_float).astype(np.int64)
    end = np.rint(end_float).astype(np.int64)
    if not np.allclose(start_float, start, atol=1e-6) or not np.allclose(end_float, end, atol=1e-6):
        raise ValueError("target pc range boundaries must align to voxel grid")

    return tuple(slice(int(s), int(e)) for s, e in zip(start, end))


def crop_npz(src_path, dst_path, src_shape, dst_shape, slices, overwrite=False):
    if dst_path.exists() and not overwrite:
        return "skip"

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(src_path, allow_pickle=True)
    output = {}
    cropped_keys = []
    preserved_keys = []

    for key in data.files:
        arr = data[key]
        if arr.ndim >= 3 and tuple(arr.shape[:3]) == src_shape:
            output[key] = arr[slices + tuple(slice(None) for _ in range(arr.ndim - 3))]
            if tuple(output[key].shape[:3]) != dst_shape:
                raise RuntimeError(
                    f"{src_path}: key={key} crop produced {output[key].shape[:3]}, expected {dst_shape}"
                )
            cropped_keys.append(key)
        else:
            output[key] = arr
            preserved_keys.append(key)

    tmp_path = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            dir=str(dst_path.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        np.savez_compressed(tmp_path, **output)
        os.replace(tmp_path, dst_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    return f"ok cropped={','.join(cropped_keys) or '-'} preserved={','.join(preserved_keys) or '-'}"


def main():
    parser = argparse.ArgumentParser(
        description="Crop OCC gts labels.npz from a source pc range to a smaller target pc range."
    )
    parser.add_argument("--input-root", default="./data/v2xreal_nuscenes",
                        help="Dataset root that contains the source gts directory.")
    parser.add_argument("--src-gts-name", default="gts",
                        help="Source gts directory name under input-root.")
    parser.add_argument("--dst-gts-name", default="gts_16",
                        help="Output gts directory name under input-root.")
    parser.add_argument("--src-pc-range", default="-64.0,-64.0,-4.8,64.0,64.0,8.0",
                        help="Source pc range: xmin,ymin,zmin,xmax,ymax,zmax.")
    parser.add_argument("--dst-pc-range", default="-64.0,-64.0,-4.8,64.0,64.0,1.6",
                        help="Target pc range: xmin,ymin,zmin,xmax,ymax,zmax.")
    parser.add_argument("--voxel-size", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files in dst-gts-name.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print conversion summary; do not write output files.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    src_root = input_root / args.src_gts_name
    dst_root = input_root / args.dst_gts_name

    src_range = parse_range(args.src_pc_range)
    dst_range = parse_range(args.dst_pc_range)
    src_shape = occ_size(src_range, args.voxel_size)
    dst_shape = occ_size(dst_range, args.voxel_size)
    slices = crop_slices(src_range, dst_range, args.voxel_size)

    if not src_root.exists():
        raise FileNotFoundError(f"source gts directory not found: {src_root}")

    label_paths = sorted(src_root.rglob("labels.npz"))
    if not label_paths:
        raise FileNotFoundError(f"no labels.npz found under: {src_root}")

    print(f"input_root : {input_root}")
    print(f"source     : {src_root}")
    print(f"target     : {dst_root}")
    print(f"src_shape  : {src_shape}, pc_range={src_range.tolist()}")
    print(f"dst_shape  : {dst_shape}, pc_range={dst_range.tolist()}")
    print(f"crop       : x={slices[0]}, y={slices[1]}, z={slices[2]}")
    print(f"frames     : {len(label_paths)}")

    if args.dry_run:
        sample = label_paths[0]
        data = np.load(sample, allow_pickle=True)
        print(f"sample     : {sample}")
        for key in data.files:
            arr = data[key]
            action = "crop" if arr.ndim >= 3 and tuple(arr.shape[:3]) == src_shape else "preserve"
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, action={action}")
        return

    def convert_one(src_path):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        return src_path, crop_npz(
            src_path,
            dst_path,
            src_shape,
            dst_shape,
            slices,
            overwrite=args.overwrite,
        )

    workers = max(1, int(args.num_workers))
    ok = 0
    skipped = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_one, p) for p in label_paths]
        for idx, fut in enumerate(as_completed(futures), 1):
            try:
                _, status = fut.result()
                if status == "skip":
                    skipped += 1
                else:
                    ok += 1
            except Exception as exc:
                failed += 1
                print(f"[ERROR] {exc}")
            if idx == 1 or idx % 200 == 0 or idx == len(futures):
                print(f"progress {idx}/{len(futures)} ok={ok} skipped={skipped} failed={failed}")

    if failed:
        raise SystemExit(f"finished with {failed} failed frames")
    print(f"done: ok={ok}, skipped={skipped}, output={dst_root}")


if __name__ == "__main__":
    main()
