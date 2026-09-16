#!/usr/bin/env python3
"""msModelSlim 单级 MXFP4 双专家导出的只读结构校验（不修改任何文件）。

只依赖 torch 与 safetensors。用于在真实导出之后、CPU 转换之前确认导出确实是
单级 W4A4_MXFP4 + Smooth 合同，而不是 DualScale、W8A8 或其它配方。

校验内容：
  * 两个专家目录、唯一 quant_model_description、唯一 quant_model_weight 分片；
  * model_quant_type 精确为 W4A4_MXFP4，tensor 标签只有 W4A4_MXFP4 / FLOAT；
  * packed weight 为 uint8[N, K/2]，scale 为 uint8[N, K/32] 且不含 E8M0 NaN(255)；
  * 不存在 dual_scale / rotation / hadamard 等不属于本合同的 tensor 或文件；
  * Smooth 覆盖全部量化层且非平凡（不是全 1），数值有限且为正；
  * 量化层只出现在被配方排除的 blocks.0-4 之外（固定配方的 exclude 名单）。

退出码 0 表示结构通过；非 0 表示失败并打印原因。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from safetensors import safe_open

QUANT_TYPE = "W4A4_MXFP4"
EXPERTS = ("high_noise_model", "low_noise_model")
EXCLUDED_BLOCK_PREFIXES = tuple(f"blocks.{index}." for index in range(5))
FORBIDDEN_TOKENS = ("quarot", "rotation", "rotate", "hadamard", "dual_scale")


def _read_json(path: pathlib.Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 重复键 {key}: {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def _tensor_headers(directory: pathlib.Path) -> dict[str, dict]:
    headers: dict[str, dict] = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as archive:
            for name in archive.keys():
                if name in headers:
                    raise ValueError(f"分片重复 tensor {name}: {path}")
                view = archive.get_slice(name)
                headers[name] = {"file": path, "shape": tuple(view.get_shape()), "dtype": view.get_dtype()}
    if not headers:
        raise ValueError(f"缺少 safetensors: {directory}")
    return headers


def check_expert(root: pathlib.Path, expert: str) -> dict:
    directory = root / expert
    if not directory.is_dir():
        raise ValueError(f"缺少专家目录: {directory}")
    for path in root.rglob("*"):
        if path.is_file() and any(token in path.name.lower() for token in FORBIDDEN_TOKENS):
            raise ValueError(f"出现不属于单级合同的文件: {path}")
    descriptions = sorted(directory.glob("quant_model_description*.json"))
    if len(descriptions) != 1:
        raise ValueError(f"需要唯一 quant_model_description: {directory}")
    description = _read_json(descriptions[0])
    if description.get("model_quant_type") != QUANT_TYPE:
        raise ValueError(f"model_quant_type 必须为 {QUANT_TYPE}: {description.get('model_quant_type')!r}")
    labels = {key: value for key, value in description.items() if isinstance(value, str)}
    unknown = {key: value for key, value in labels.items() if value not in (QUANT_TYPE, "FLOAT")}
    if unknown:
        raise ValueError(f"不支持的 quant label: {unknown}")

    headers = _tensor_headers(directory)
    missing = headers.keys() - labels.keys()
    if missing:
        raise ValueError(f"tensor 缺少量化描述: {sorted(missing)}")

    quantized: list[str] = []
    scales: dict[str, torch.Tensor] = {}
    smooth: dict[str, tuple[float, float, bool]] = {}
    for name, info in headers.items():
        if name.endswith(".weight_scale"):
            prefix = name.removesuffix(".weight_scale")
            weight = headers.get(prefix + ".weight")
            if weight is None or weight["dtype"] != "U8" or info["dtype"] != "U8":
                raise ValueError(f"weight/weight_scale 必须为 uint8: {name}")
            if len(info["shape"]) != 2 or len(weight["shape"]) != 2:
                raise ValueError(f"weight/weight_scale 必须为二维: {name}")
            if info["shape"][0] != weight["shape"][0] or info["shape"][1] * 16 != weight["shape"][1]:
                raise ValueError(f"scale 与 weight 的 group32 关系不成立: {name}")
            with safe_open(str(info["file"]), framework="pt", device="cpu") as archive:
                value = archive.get_tensor(name)
            if bool((value == 255).any()):
                raise ValueError(f"weight_scale 含 E8M0 NaN(255): {name}")
            scales[prefix] = value
            quantized.append(prefix)
        elif name.endswith(".div.mul_scale"):
            prefix = name.removesuffix(".div.mul_scale")
            with safe_open(str(info["file"]), framework="pt", device="cpu") as archive:
                value = archive.get_tensor(name).float()
            if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                raise ValueError(f"mul_scale 必须有限且为正: {name}")
            smooth[prefix] = (float(value.min()), float(value.max()), bool((value == 1).all()))

    if not quantized:
        raise ValueError(f"{expert} 没有单级量化层")
    if any(token in name for name in headers for token in FORBIDDEN_TOKENS):
        raise ValueError(f"{expert} 含 dual_scale 或旋转 tensor")

    quant_layers = {name.removesuffix(".linear") for name in quantized}
    if set(quant_layers) != set(smooth):
        raise ValueError(
            f"{expert} Smooth 未覆盖全部量化层: 量化 {len(quant_layers)} / Smooth {len(smooth)}"
        )
    trivial = sorted(name for name, value in smooth.items() if value[2])
    if trivial:
        raise ValueError(f"{expert} 存在全 1 的平凡 Smooth（未真正生效）: {trivial[:5]}")
    inside_excluded = sorted(
        name for name in quant_layers if name.startswith(EXCLUDED_BLOCK_PREFIXES)
    )
    if inside_excluded:
        raise ValueError(f"{expert} 前 5 个 block 不应被量化: {inside_excluded[:5]}")

    return {
        "tensor_count": len(headers),
        "quantized_layers": len(quant_layers),
        "smooth_layers": len(smooth),
        "smooth_min": min(value[0] for value in smooth.values()),
        "smooth_max": max(value[1] for value in smooth.values()),
        "description_sha256": __import__("hashlib").sha256(descriptions[0].read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="只读校验 msModelSlim 单级 MXFP4 双专家导出。")
    parser.add_argument("--quant-path", required=True, type=pathlib.Path, help="导出根目录（含两个专家）")
    args = parser.parse_args()
    root = args.quant_path.resolve()
    if not root.is_dir():
        parser.exit(2, f"导出目录不存在: {root}\n")
    directories = {path.name for path in root.iterdir() if path.is_dir()}
    if directories != set(EXPERTS):
        parser.exit(2, f"导出根必须只含 {sorted(EXPERTS)}，实际 {sorted(directories)}\n")
    report = {"status": "EXPORT_STRUCTURE_VALIDATED", "quant_path": str(root), "experts": {}}
    try:
        for expert in EXPERTS:
            report["experts"][expert] = check_expert(root, expert)
    except (ValueError, OSError) as error:
        parser.exit(2, f"导出结构校验失败: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
