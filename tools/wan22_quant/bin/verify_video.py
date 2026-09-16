#!/usr/bin/env python3
"""解码校验生成的视频：帧数、分辨率、fps、有限性与逐帧方差。

退出码 0 表示视频可完整解码且参数符合验收。只依赖 imageio（Omni 镜像已自带）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="解码校验生成的视频。")
    parser.add_argument("--video", required=True, type=pathlib.Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    reader = imageio.get_reader(str(args.video), format="FFMPEG")
    meta = reader.get_meta_data()
    count = 0
    frame_stds: list[float] = []
    for frame in reader:
        if frame.shape != (args.height, args.width, 3):
            raise ValueError(f"帧 {count} 尺寸不符: {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError(f"帧 {count} 含非有限值")
        frame_stds.append(float(frame.std()))
        count += 1
    reader.close()
    if count != args.frames:
        raise ValueError(f"解码帧数 {count} != 期望 {args.frames}")
    if abs(meta["fps"] - args.fps) > 1e-6:
        raise ValueError(f"fps {meta['fps']} != 期望 {args.fps}")

    result = {
        "status": "passed",
        "decoded_frames": count,
        "width": args.width,
        "height": args.height,
        "fps": meta["fps"],
        "codec": meta.get("codec"),
        "sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "bytes": args.video.stat().st_size,
        "frame_std_min": min(frame_stds),
        "frame_std_max": max(frame_stds),
        "quality_gate": "not_run",
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"视频校验失败: {error}", file=sys.stderr)
        sys.exit(2)
