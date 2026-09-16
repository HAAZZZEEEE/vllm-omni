#!/usr/bin/env bash
# 用 run.sh 产出的 model/ 生成一个视频并解码校验。
# 用法：bash smoke.sh /absolute/path/env.sh
#
# 固定验收参数：832x480、33 帧、40 步、seed 42、16fps、step fallback=[0,1,38,39]，
# prompt 为 "A cat walks slowly across a sunlit garden."。
set -euo pipefail

ENV_FILE="${1:-}"
if [ -z "$ENV_FILE" ]; then
  echo "用法: bash smoke.sh /absolute/path/env.sh" >&2
  exit 2
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "找不到 env 文件: $ENV_FILE" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in OUTPUT_ROOT OMNI_SOURCE; do
  if [ -z "${!name:-}" ]; then
    echo "env.sh 缺少必填变量: $name" >&2
    exit 2
  fi
done
MODEL_DIR="${MODEL_DIR:-$OUTPUT_ROOT/model}"
LOG_DIR="$OUTPUT_ROOT/logs"
SMOKE_IMAGE="${SMOKE_IMAGE:-${IMAGE:-quay.io/ascend/vllm-omni:v0.28.0-a5}}"
SMOKE_DEVICE="${SMOKE_DEVICE:-${DEVICE:-0}}"
ADALAYERNORM_PATCH="${ADALAYERNORM_PATCH:-}"
JEMALLOC="${JEMALLOC:-}"
MALLOC_CONF_VALUE="${MALLOC_CONF_VALUE:-abort:true,junk:true}"
SMOKE_PROMPT="${SMOKE_PROMPT:-A cat walks slowly across a sunlit garden.}"
SMOKE_SEED="${SMOKE_SEED:-42}"
SMOKE_WIDTH="${SMOKE_WIDTH:-832}"
SMOKE_HEIGHT="${SMOKE_HEIGHT:-480}"
SMOKE_FRAMES="${SMOKE_FRAMES:-33}"
SMOKE_STEPS="${SMOKE_STEPS:-40}"
SMOKE_FPS="${SMOKE_FPS:-16}"
SMOKE_FALLBACK_STEPS="${SMOKE_FALLBACK_STEPS:-[0,1,38,39]}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "找不到转换后的模型: $MODEL_DIR（先运行 run.sh）" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"

INNER="$LOG_DIR/smoke-command.sh"
cat > "$INNER" <<'INNER_SCRIPT'
#!/usr/bin/env bash
# 由 smoke.sh 生成并保存，记录本次容器内实际执行的生成命令。
set -euo pipefail
export PYTHONPATH=/workspace/omni
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

python3 /workspace/omni/examples/offline_inference/text_to_video/text_to_video.py \
  --model /workspace/model \
  --quantization-config "{\"method\":\"mxfp4\",\"is_checkpoint_mxfp4_serialized\":true,\"require_smooth_scale\":true,\"mxfp4_scale_alg\":2,\"w4a8_fallback_steps\":$SMOKE_FALLBACK_STEPS}" \
  --prompt "$SMOKE_PROMPT" \
  --seed "$SMOKE_SEED" \
  --height "$SMOKE_HEIGHT" \
  --width "$SMOKE_WIDTH" \
  --num-frames "$SMOKE_FRAMES" \
  --num-inference-steps "$SMOKE_STEPS" \
  --guidance-scale 4.0 \
  --guidance-scale-high 3.0 \
  --flow-shift 12.0 \
  --fps "$SMOKE_FPS" \
  --vae-use-tiling \
  --enforce-eager \
  --output /out/smoke.mp4
INNER_SCRIPT
chmod +x "$INNER"

MOUNTS=(
  -v "$OMNI_SOURCE:/workspace/omni:ro"
  -v "$MODEL_DIR:/workspace/model:ro"
  -v "$TOOL_DIR:/tool:ro"
  -v "$OUTPUT_ROOT:/out"
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro
)
ENVS=(
  -e "ASCEND_RT_VISIBLE_DEVICES=$SMOKE_DEVICE"
  -e "SMOKE_PROMPT=$SMOKE_PROMPT"
  -e "SMOKE_SEED=$SMOKE_SEED"
  -e "SMOKE_WIDTH=$SMOKE_WIDTH"
  -e "SMOKE_HEIGHT=$SMOKE_HEIGHT"
  -e "SMOKE_FRAMES=$SMOKE_FRAMES"
  -e "SMOKE_STEPS=$SMOKE_STEPS"
  -e "SMOKE_FPS=$SMOKE_FPS"
  -e "SMOKE_FALLBACK_STEPS=$SMOKE_FALLBACK_STEPS"
)
if [ -n "$ADALAYERNORM_PATCH" ]; then
  MOUNTS+=( -v "$ADALAYERNORM_PATCH:/workspace/omni/vllm_omni/diffusion/layers/adalayernorm.py:ro" )
  echo "使用 AdaLayerNorm 兼容覆盖: $ADALAYERNORM_PATCH"
fi
if [ -n "$JEMALLOC" ]; then
  ENVS+=( -e "LD_PRELOAD=$JEMALLOC" -e "MALLOC_CONF=$MALLOC_CONF_VALUE" )
  echo "使用 jemalloc: $JEMALLOC ($MALLOC_CONF_VALUE)"
fi

CONTAINER_NAME="wan22-quant-smoke-$(date +%Y%m%d-%H%M%S)"
echo "容器: $CONTAINER_NAME   镜像: $SMOKE_IMAGE   卡: $SMOKE_DEVICE"
echo "模型: $MODEL_DIR"

VERIFY="python3 /tool/bin/verify_video.py --video /out/smoke.mp4 \
  --frames $SMOKE_FRAMES --width $SMOKE_WIDTH --height $SMOKE_HEIGHT --fps $SMOKE_FPS \
  --out /out/logs/video-validation.json"

set +e
docker run --rm --name "$CONTAINER_NAME" \
  --privileged --ipc=host --network=host --log-driver=none \
  "${ENVS[@]}" "${MOUNTS[@]}" \
  "$SMOKE_IMAGE" bash -lc "
    set -o pipefail
    bash /out/logs/smoke-command.sh 2>&1 | tee /out/logs/smoke.log
    rc=\${PIPESTATUS[0]}
    echo \$rc > /out/logs/smoke.exit
    if [ \$rc -ne 0 ]; then exit \$rc; fi
    $VERIFY 2>&1 | tee -a /out/logs/smoke.log
  "
RC=$?
set -e

echo "退出码: $RC"
if [ "$RC" -ne 0 ]; then
  echo "失败。日志: $LOG_DIR/smoke.log" >&2
  exit "$RC"
fi
echo "视频: $OUTPUT_ROOT/smoke.mp4"
echo "校验: $LOG_DIR/video-validation.json"
