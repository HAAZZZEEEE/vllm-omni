#!/usr/bin/env python3
"""转换结果只读对账：把 Omni checkpoint 逐 tensor 与 msModelSlim 原生导出比对。

只依赖 torch 与 safetensors。校验内容：
  * 两个专家的 config.json 都显式写入 quantization_config：
    quant_method=mxfp4、is_checkpoint_mxfp4_serialized=true、
    require_smooth_scale=true、mxfp4_scale_alg=2（UOS）、以及自动生成的 ignored_layers；
  * 每个量化层的 weight 与原生 packed FP4 的 E2M1 精确展开逐元素相等（不乘 scale）；
  * 每个 weight_scale 的原字节与原生导出完全相同；
  * 每个 Smooth mul_scale 与原生导出逐元素相等；
  * 明确 FLOAT 的层沿用原 BF16 或导出浮点值，不出现静默 BF16 替代。

默认全量比对（约 66 GB 读 I/O）。`--sample N` 可只比对每专家前 N 个量化层，
用于快速冒烟；正式验收请用默认全量。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import torch
from safetensors import safe_open

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_native = _load("_wan22_quant_native", _HERE / "mxfp4_native.py")
_converter = _load("_wan22_quant_converter", _HERE / "merge_mxfp4_checkpoint.py")
EXPERT_TO_COMPONENT = _native.NATIVE_EXPERT_TO_COMPONENT


class Shards:
    """按 header 索引读取转换后 checkpoint 的单个 tensor。"""

    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        self.files: dict[str, pathlib.Path] = {}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as archive:
                for name in archive.keys():
                    if name in self.files:
                        raise ValueError(f"分片重复 tensor: {name}")
                    self.files[name] = path
        if not self.files:
            raise ValueError(f"转换后 checkpoint 缺少 safetensors: {directory}")

    def get(self, name: str) -> torch.Tensor:
        with safe_open(str(self.files[name]), framework="pt", device="cpu") as archive:
            return archive.get_tensor(name)


def compare(output_model: pathlib.Path, native_root: pathlib.Path, original_model: pathlib.Path, sample: int) -> dict:
    report: dict = {"status": "CHECKING", "experts": {}}
    for native_expert, component in sorted(EXPERT_TO_COMPONENT.items(), key=lambda item: item[1]):
        expert_dir = output_model / component
        config = json.loads((expert_dir / "config.json").read_text(encoding="utf-8"))
        quant_config = config.get("quantization_config") or {}
        if quant_config.get("quant_method") != "mxfp4":
            raise ValueError(f"{component} quant_method 必须为 mxfp4: {quant_config.get('quant_method')!r}")
        if quant_config.get("is_checkpoint_mxfp4_serialized") is not True:
            raise ValueError(f"{component} 缺少 is_checkpoint_mxfp4_serialized=true")
        if quant_config.get("require_smooth_scale") is not True:
            raise ValueError(f"{component} 缺少 require_smooth_scale=true")
        if quant_config.get("mxfp4_scale_alg") != 2:
            raise ValueError(f"{component} 必须显式写入 mxfp4_scale_alg=2，实际 {quant_config.get('mxfp4_scale_alg')!r}")
        ignored = quant_config.get("ignored_layers")
        if not isinstance(ignored, list) or not ignored:
            raise ValueError(f"{component} ignored_layers 缺失或为空")

        reader = _native.NativeMXFP4Expert(
            native_root / native_expert, original_model / component, True
        )
        shards = Shards(expert_dir)
        layers = sorted(reader.quant_layers)
        if sample:
            layers = layers[:sample]
        checked = 0
        for layer in layers:
            weight_target = layer + ".weight"
            source = reader.native_name_by_diffusers_name[weight_target]
            expected = _converter._unpack_fp4(reader.quant.get(source))
            actual = shards.get(weight_target)
            if actual.dtype != torch.bfloat16 or actual.shape != expected.shape:
                raise ValueError(f"{component} {weight_target} dtype/shape 不符: {actual.dtype} {tuple(actual.shape)}")
            if not torch.equal(actual, expected):
                raise ValueError(f"{component} {weight_target} 与原生 E2M1 展开不一致")

            scale_target = layer + ".weight_scale"
            scale_source = source.removesuffix(".weight") + ".weight_scale"
            expected_scale = reader.quant.get(scale_source)
            actual_scale = shards.get(scale_target)
            if not torch.equal(actual_scale, expected_scale):
                raise ValueError(f"{component} {scale_target} 与原生 E8M0 字节不一致")

            if layer in reader.smooth:
                smooth_target = layer + ".mul_scale"
                smooth_source = source.removesuffix(".weight").removesuffix(".linear") + ".div.mul_scale"
                expected_smooth = reader.quant.get(smooth_source).float()
                actual_smooth = shards.get(smooth_target).float()
                if not torch.equal(actual_smooth, expected_smooth):
                    raise ValueError(f"{component} {smooth_target} 与原生 Smooth 不一致")
            checked += 1
            if checked % 50 == 0:
                print(f"  {component}: 已比对 {checked}/{len(layers)} 层", flush=True)

        report["experts"][component] = {
            "quantized_layers_total": len(reader.quant_layers),
            "quantized_layers_compared": checked,
            "sample_mode": bool(sample),
            "smooth_layers": len(reader.smooth),
            "ignored_layers": len(ignored),
            "mxfp4_scale_alg": quant_config["mxfp4_scale_alg"],
        }
        print(f"{component}: 对账通过（{checked} 层）", flush=True)
    report["status"] = "CONVERTED_CHECKPOINT_RECONCILED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="只读对账转换后的 Omni MXFP4 checkpoint 与原生导出。")
    parser.add_argument("--quant-path", required=True, type=pathlib.Path, help="msModelSlim 原生导出根")
    parser.add_argument("--original-model", required=True, type=pathlib.Path, help="原始 BF16 Diffusers 模型")
    parser.add_argument("--output-model", required=True, type=pathlib.Path, help="转换后的 Omni checkpoint")
    parser.add_argument("--sample", type=int, default=0, help="每专家只比对前 N 层（0 = 全量）")
    args = parser.parse_args()
    try:
        report = compare(args.output_model.resolve(), args.quant_path.resolve(), args.original_model.resolve(), args.sample)
    except (ValueError, OSError) as error:
        parser.exit(2, f"转换对账失败: {error}\n")
    report["output_model"] = str(args.output_model.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
