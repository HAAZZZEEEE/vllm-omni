# Wan2.2 单级 MXFP4 量化导出（UOS + Smooth）

把一个原始 Wan2.2-T2V-A14B 模型导出成可直接推理的 Omni MXFP4 checkpoint 的最小工具集。
只做一件事：**导出 → 转换 → 校验**，外加一条视频冒烟命令。

- 唯一默认路线：原始 Wan 模型 → msModelSlim `mindie_format_saver` 双专家导出 →
  CPU converter＋BF16 Diffusers 原模型 → Omni 规范 checkpoint → 推理。
- 范围限定 `Wan2.2-T2V-A14B`（双专家）。不覆盖 DualScale、W8A8/MXFP8、W4A8_MXFP、FA、
  QuaRot/rotation，也不提供“静默转 BF16”开关。
- 本工具不做发布、装包、CI 矩阵或精度重测。

## 快速开始

```bash
cd tools/wan22_quant
cp env.example.sh env.sh
vi env.sh                     # 只改这一个文件
bash run.sh   /absolute/path/env.sh    # 导出 → 转换 → 校验
bash smoke.sh /absolute/path/env.sh    # 用上一步的 model/ 生成一个视频
```

`run.sh` 与 `smoke.sh` 可以分别在不同节点执行：导出需要 NPU 与 msModelSlim 环境，
视频需要 Omni 推理镜像。两边各写一份 env.sh 即可。**但更推荐在同一台机器上跑完**，
见下文「本次实测」——跨机需要搬 66G 的 `model/`，而同机只需要带上源码树和覆盖补丁。

失败即退出；`OUTPUT_ROOT/native` 或 `OUTPUT_ROOT/model` 已存在时直接拒绝覆盖。

## env.sh 变量

完整示例见 `env.example.sh`，每个变量都有注释。必填项：

| 变量 | 含义 |
| --- | --- |
| `NATIVE_MODEL` | 原始 Wan2.2-T2V-A14B（导出输入，含 `high_noise_model`/`low_noise_model`） |
| `BF16_DIFFUSERS_MODEL` | 配套的原始 BF16 Diffusers 模型（转换输入，必须含 `transformer/` 与 `transformer_2/`） |
| `BF16_DIFFUSERS_COMPONENT_ROOT` | 可选。当上面这个目录里的 `scheduler`/`text_encoder`/`tokenizer`/`vae` 是指向 `/models/...` 的绝对软链时，把该目录挂到容器的 `/models` |
| `CALIB_DATASET_DIR` | 校准输入目录，容器内固定挂载为 `/dataset`；配方读 `/dataset/index.jsonl` |
| `MSS_MODELSLIM_LIB` | msModelSlim 及其依赖所在目录（含 `modelslim/`、`msmodelslim/`） |
| `WAN_SRC` | Wan 源码（`--trust_remote_code` 需要），同时作为容器工作目录 |
| `MINDIESD_WHEEL` | mindiesd 依赖 wheel，见下节 |
| `OUTPUT_ROOT` | 输出根，脚本在其下生成 `native/`、`model/`、`logs/` |
| `DEVICE` | 导出用物理 NPU 号 |

`smoke.sh` 另外需要 `OMNI_SOURCE`（已验证的 vllm-omni 源码树）、`SMOKE_IMAGE`、`SMOKE_DEVICE`，
以及可选的 `ADALAYERNORM_PATCH`、`JEMALLOC`、`MALLOC_CONF_VALUE`。

## 输出位置

```text
$OUTPUT_ROOT/
├── native/      # msModelSlim 双专家导出（high_noise_model/、low_noise_model/）
├── model/       # 转换后的 Omni checkpoint（含 mxfp4_conversion_report.json）
├── logs/
│   ├── in-container.sh       # 本次容器内实际执行的完整步骤
│   ├── run.log               # 完整输出
│   ├── run-result.txt        # 成功时的输出路径清单
│   ├── smoke-command.sh      # 本次生成的完整推理命令
│   ├── smoke.log             # 生成日志
│   ├── smoke.exit            # 生成进程退出码
│   └── video-validation.json # 解码校验结果
└── smoke.mp4                 # 冒烟视频
```

测试不需要手改任何中间 JSON；所有路径都由 env.sh 决定。

## 固定配方

`c7-smooth.yaml` 就是已验证的 C7/UOS 配方，脚本直接使用它，不做文本替换：

- 权重 `ceil_x_value=7.25`、`enable_search=false`；激活 `method=minmax`；
- 前 5 个 block 与根部浮点层保持原策略（`exclude: blocks.0-4`）；
- IterSmooth `alpha=0.25`；`mindie_format_saver`，`part_file_size=0`；
- 校准 1280×720、81 帧、40 步、shift 12、seed 0、unipc。

运行时转换工具会把 `mxfp4_scale_alg=2`（A4 显式 UOS）写入**两个**专家的
`quantization_config`，同时保留自动生成的 `ignored_layers` 与
`is_checkpoint_mxfp4_serialized`，并要求 Smooth。该取值是固定配方的显式选择，
由入口显式传入 `--mxfp4-scale-alg 2`（独立转换 CLI 同样默认 `2`），并记入 `mxfp4_conversion_report.json` 的
`runtime_scale_alg` 字段，不是无来源的全局默认。

## 成功标准

`run.sh` 退出码 0，且：

1. `logs/run.log` 出现 `==== [2/4] 校验原生导出结构 ====` 的结构校验通过输出；
2. `native/` 下包含 `high_noise_model/`、`low_noise_model/` 两个专家目录；生产工具也可能保存配方和校准缓存；
3. `model/` 下两个专家的 `config.json` 的 `quantization_config` 含
   `quant_method=mxfp4`、`is_checkpoint_mxfp4_serialized=true`、
   `require_smooth_scale=true`、`mxfp4_scale_alg=2` 和非空 `ignored_layers`；
4. `logs/run.log` 出现 `CONVERTED_CHECKPOINT_RECONCILED`：转换结果与原生导出
   逐 tensor 对账通过（weight 是 E2M1 精确展开、scale 字节不变、Smooth 原值）。

`smoke.sh` 退出码 0，且 `logs/video-validation.json` 为 `"status": "passed"`，
`decoded_frames` 等于设定帧数。质量与性能评估不在本工具范围。

## 本次实测（2026-09-16）

在同一台 A5 机器上完成导出、转换、校验和出片。首次入口执行在最终对账阶段遇到校验器参数名不匹配；修正后复用本次导出和转换产物，全量对账及视频冒烟通过。交付代码已包含该修正，未重新执行整条导出入口。

| 项 | 结果 |
| --- | --- |
| 导出 | msModelSlim 双专家，NPU2，单级 W4A4_MXFP4；`native/` 22G，每专家 350 量化层 / 350 Smooth 层 |
| 导出描述哈希 | `eb6a64dbc0ef11a7191fe55581290166a29f6cbe2eeb70b8b835baa5a48e8f31`（与历史 C7 导出一致） |
| 结构校验 | `EXPORT_STRUCTURE_VALIDATED` |
| 转换 | 66G；两专家 `quant_method=mxfp4`、`is_checkpoint_mxfp4_serialized=true`、`require_smooth_scale=true`、`mxfp4_scale_alg=2`、`ignored_layers=247` |
| 对账 | `CONVERTED_CHECKPOINT_RECONCILED`；两专家各 350/350 层全量逐 tensor 一致 |
| 视频 | 退出码 0，33 帧全解码，832×480、16fps、h264，sha256 `ee46b729a498f85b16253b062da32b3e01d005c846391f546500437f79a4cb6e` |
| 移出测试 | 78 passed，退出码 0 |

该配置下视频与此前验证产物逐字节相同；这只证明本次视频一致，不证明两个检查点的全部权重数值等价。转换保真由本次原生导出与转换产物的逐张量对账验证。

> **建议在同一台机器上跑完两条入口。** `model/` 约 66G，把它搬到另一台机器只为出片并不划算；
> 换机器真正需要携带的只有已验证的 `OMNI_SOURCE` 源码树（约 75MB）和
> `ADALAYERNORM_PATCH`（约 5KB）。本次视频就是在转换节点本机出片的。

导出镜像与关键版本（实测）：

| 项 | 值 |
| --- | --- |
| 镜像 | `quay.io/ascend/vllm-omni:v0.28.0-a5` |
| Python | 3.12.13 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| pytest（跑测试用） | 9.1.1 |

### mindiesd 必须单独提供

Wan2.2 源码在 import 期就会拉 mindiesd（`wan/utils/rainfusion.py`），而上面这个镜像
本身不含它，直接跑 msModelSlim 会以 `ModuleNotFoundError: No module named 'mindiesd'`
失败。`run.sh` 会把 `MINDIESD_WHEEL` 只读挂进容器并 `pip install --no-index --no-deps`
装到容器内（不写宿主环境），随后先验证 `import mindiesd, wan` 再开始导出。

必须是与该 Python/torch 匹配的 **cp312 x86_64** 版本：

```text
mindiesd-3.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
sha256 5e4831331c2b9ab76d42195bec2fd4352625fdd87815bf0e4d7a13c344add7ee
```

该 wheel 内含 `mindiesd/plugin/torch210/libPTAExtensionOPS.so`，与 torch 2.10 对应。
填了 `MINDIESD_WHEEL_SHA256` 时 `run.sh` 会先校验哈希再启动。

> 注意：cp311 的 mindiesd wheel（`...-cp311-cp311-manylinux_2_34_x86_64.whl`）在本环境
> 不可用，Python 版本不匹配。

### 测试环境条件（不是代码缺陷）

最新 E2E 环境上有两个必须写明的条件，`smoke.sh` 通过 env.sh 支持：

- **AdaLayerNorm 兼容覆盖**：裸候选在该 A5 环境会因旧 LayerNorm 算子没有 ACLNN 实现
  而失败。`ADALAYERNORM_PATCH` 把兼容实现只读覆盖到 `OMNI_SOURCE` 的
  `vllm_omni/diffusion/layers/adalayernorm.py` 上，**不改动 OMNI_SOURCE 本身**。
  留空表示不覆盖。
- **jemalloc**：该环境下进程退出时会在 `libruntime_v200.so` 的析构路径崩溃。
  `JEMALLOC`（默认 `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`）配合
  `MALLOC_CONF_VALUE=abort:true,junk:true` 可让进程正常退出。
  这是规避配置，不表示底层问题已修复。

### msModelSlim 来源

msModelSlim 用的是当时实际可用的代码树（含 `lib/` 下的 site 目录），不是只靠一个
Git SHA 就能复现的安装。把 `MSS_MODELSLIM_LIB` 指向那份实际可用的目录即可；
`run.sh` 只读挂载它。

## 转换工具说明

`bin/merge_mxfp4_checkpoint.py` 在 CPU 上把 msModelSlim 的双专家 MindIE 导出转换成
Diffusers/Omni 目录，依赖只有 `torch` 与 `safetensors`，不需要 vLLM、msModelSlim、
torch_npu 或 vllm-ascend。

它把 packed FP4 展开成 BF16 容器中的 E2M1 数值，**保留 E8M0 scale 字节和 Smooth，
不反量化再量化**，也不算 C7、不跑校准。展开后的 weight 约为 packed 体积的四倍，
默认按约 4096 MiB 分片，单 tensor 不拆分。

单独跑转换（复用已有原生导出时）：

```bash
python3 bin/merge_mxfp4_checkpoint.py \
  --original-model /path/to/Wan2.2-T2V-A14B-Diffusers-BF16 \
  --quant-path     /path/to/native \
  --output-path    /path/to/model \
  --require-smooth-scale --mxfp4-scale-alg 2
```

## 测试

移出的转换测试独立于推理栈，使用真实 safetensors/JSON fixture 与真实 CLI 子进程：

```bash
python3 -m pytest --confcutdir=tests tests/test_mxfp4_single_checkpoint_conversion.py \
  -q -o addopts="" -p no:cacheprovider
```

`-o addopts=""` 用来避免仓库根 pyproject 的 addopts 影响独立运行；此时会出现
`core_model`/`diffusion`/`cpu` 三个未知 mark 的警告，不影响结果。

## 目录内容

| 路径 | 作用 |
| --- | --- |
| `run.sh` / `smoke.sh` | 导出→转换→校验；单片推理 |
| `env.example.sh` | 用户唯一需要改的文件 |
| `c7-smooth.yaml` | 固定 C7/UOS+Smooth 配方（`dataset` 固定为 `/dataset/index.jsonl`） |
| `bin/merge_mxfp4_checkpoint.py` | CPU 格式转换（已去掉 Omni 包类型引用） |
| `bin/mxfp4_native.py` | 原生导出只读格式解析与严格校验 |
| `bin/validate_export.py` | 导出结构只读校验（标签/ packing / group32 / Smooth 非平凡） |
| `bin/validate_converted.py` | 转换结果与原生导出逐 tensor 对账 |
| `bin/verify_video.py` | 视频解码校验 |
| `tests/test_mxfp4_single_checkpoint_conversion.py` | 移出的转换测试 |
