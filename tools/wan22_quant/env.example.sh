# Wan2.2 单级 MXFP4 量化导出 —— 用户唯一需要修改的文件
#
# 用法：
#   cp env.example.sh env.sh && vi env.sh
#   bash run.sh   /absolute/path/env.sh     # 导出 → 转换 → 校验
#   bash smoke.sh /absolute/path/env.sh     # 用转换结果生成一个视频
#
# run.sh 与 smoke.sh 可以分别在不同节点执行：导出需要 NPU 与 msModelSlim 环境，
# 视频需要 Omni 推理镜像。两边的 env.sh 可以不同，按节点实际情况填写。

# ---------------------------------------------------------------------------
# 1) 输入模型（两者必须分别指定，且都必须存在）
# ---------------------------------------------------------------------------

# 原始 Wan2.2-T2V-A14B（msModelSlim 的量化输入，含 high_noise_model/low_noise_model）
NATIVE_MODEL=/mnt/weight/Wan2.2-T2V-A14B

# 配套的原始 BF16 Diffusers 模型（转换输入；必须含 transformer/ 与 transformer_2/）
BF16_DIFFUSERS_MODEL=/mnt/omni-w00665062/20260906/models/Wan2.2-T2V-A14B-Diffusers-BF16

# 可选。当 BF16_DIFFUSERS_MODEL 里的 scheduler/text_encoder/tokenizer/vae 是指向
# /models/... 的绝对软链时，把该目录挂到容器的 /models 才能解析。
# 若 BF16_DIFFUSERS_MODEL 本身是完整实体目录，把这一行留空即可。
BF16_DIFFUSERS_COMPONENT_ROOT=/mnt/share/weights

# 校准输入目录，容器内固定挂载为 /dataset；配方读取 /dataset/index.jsonl
CALIB_DATASET_DIR=/mnt/omni-w00665062/20260906/stage4/c7-preparation-20260909/lib/msmodelslim/lab_calib/wan2_2_t2v

# ---------------------------------------------------------------------------
# 2) msModelSlim 与 Wan 代码来源（导出时使用）
# ---------------------------------------------------------------------------

# msModelSlim 及其依赖所在的 site 目录（含 modelslim/、msmodelslim/）
MSS_MODELSLIM_LIB=/mnt/omni-w00665062/20260906/stage4/c7-preparation-20260909/lib

# Wan 源码（--trust_remote_code 需要），同时作为容器工作目录
WAN_SRC=/mnt/omni-w00665062/20260906/stage4/c7-preparation-20260909/wan-38fb8eb

# mindiesd 依赖 wheel。Wan 源码 import 期会拉 mindiesd，而镜像本身不含它。
# 必须是与本容器 Python/torch 匹配的 cp312 x86_64 版本：
#   mindiesd-3.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
# 验证过的副本（SHA256 见下一行）：
#   /mnt/omni-w00665062/20260906/stage4/wan22-quant-task8/deps/mindiesd-3.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
MINDIESD_WHEEL=/mnt/omni-w00665062/20260906/stage4/wan22-quant-task8/deps/mindiesd-3.1.0-cp312-cp312-manylinux_2_34_x86_64.whl

# 可选。填写后 run.sh 会在执行前校验 wheel 哈希，不匹配即拒绝启动。
MINDIESD_WHEEL_SHA256=5e4831331c2b9ab76d42195bec2fd4352625fdd87815bf0e4d7a13c344add7ee

# ---------------------------------------------------------------------------
# 3) 输出、设备与镜像
# ---------------------------------------------------------------------------

# 输出根。脚本会在其下生成 native/、model/、logs/；缺失即创建，已存在则拒绝覆盖。
OUTPUT_ROOT=/mnt/omni-w00665062/20260906/stage4/wan22-quant-task8

# 导出使用的物理 NPU 号（用执行前 npu-smi info 确认该卡空闲）
DEVICE=2

# 导出/转换使用的镜像
IMAGE=quay.io/ascend/vllm-omni:v0.28.0-a5

# ---------------------------------------------------------------------------
# 4) 视频冒烟（smoke.sh 使用）
# ---------------------------------------------------------------------------

# 已验证的 vllm-omni 源码树（smoke.sh 只读挂载它）
OMNI_SOURCE=/data1/omni-w00665062/task7-e2e-4329-20260916/source-verified

# AdaLayerNorm 兼容覆盖文件。留空表示不覆盖（裸候选在该 A5 环境会因缺少 ACLNN
# LayerNorm 实现而失败）。该覆盖只用于测试环境，不改动 OMNI_SOURCE。
ADALAYERNORM_PATCH=/data1/omni-w00665062/task7-e2e-4329-20260916/run/adalayernorm.py

# 视频使用的镜像与物理卡
SMOKE_IMAGE=vllm-omni:nightly-20260915-92715f3-a5
SMOKE_DEVICE=4

# CANN 运行时已知的析构崩溃规避（测试环境条件，不表示已修复底层问题）
JEMALLOC=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
MALLOC_CONF_VALUE=abort:true,junk:true

# 视频参数（验收固定值，通常不需要改）
SMOKE_PROMPT='A cat walks slowly across a sunlit garden.'
SMOKE_SEED=42
SMOKE_WIDTH=832
SMOKE_HEIGHT=480
SMOKE_FRAMES=33
SMOKE_STEPS=40
SMOKE_FPS=16
SMOKE_FALLBACK_STEPS='[0,1,38,39]'
