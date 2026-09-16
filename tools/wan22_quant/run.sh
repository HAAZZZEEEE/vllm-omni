#!/usr/bin/env bash
# 导出 → 转换 → 校验。用法：bash run.sh /absolute/path/env.sh
#
# 出错立即退出；输出已存在时拒绝覆盖。所有产物、日志与本次实际执行的容器内脚本
# 都保存在 OUTPUT_ROOT 下，便于测试直接引用。
set -euo pipefail

ENV_FILE="${1:-}"
if [ -z "$ENV_FILE" ]; then
  echo "用法: bash run.sh /absolute/path/env.sh" >&2
  exit 2
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "找不到 env 文件: $ENV_FILE" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in OUTPUT_ROOT NATIVE_MODEL BF16_DIFFUSERS_MODEL CALIB_DATASET_DIR MSS_MODELSLIM_LIB WAN_SRC MINDIESD_WHEEL; do
  if [ -z "${!name:-}" ]; then
    echo "env.sh 缺少必填变量: $name" >&2
    exit 2
  fi
done
if [ ! -f "$MINDIESD_WHEEL" ]; then
  echo "找不到 mindiesd wheel: $MINDIESD_WHEEL" >&2
  exit 2
fi
MINDIESD_WHEEL_SHA256="${MINDIESD_WHEEL_SHA256:-}"
if [ -n "$MINDIESD_WHEEL_SHA256" ]; then
  ACTUAL_SHA="$(sha256sum "$MINDIESD_WHEEL" | awk '{print $1}')"
  if [ "$ACTUAL_SHA" != "$MINDIESD_WHEEL_SHA256" ]; then
    echo "mindiesd wheel 哈希不符：期望 $MINDIESD_WHEEL_SHA256，实际 $ACTUAL_SHA" >&2
    exit 2
  fi
fi
DEVICE="${DEVICE:-0}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-omni:v0.28.0-a5}"
BF16_DIFFUSERS_COMPONENT_ROOT="${BF16_DIFFUSERS_COMPONENT_ROOT:-}"
WHEEL_BASENAME="$(basename "$MINDIESD_WHEEL")"

NATIVE_OUT="$OUTPUT_ROOT/native"
MODEL_OUT="$OUTPUT_ROOT/model"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

if [ -e "$NATIVE_OUT" ] || [ -e "$MODEL_OUT" ]; then
  echo "输出已存在，拒绝覆盖。先确认并自行移走: $NATIVE_OUT 或 $MODEL_OUT" >&2
  exit 2
fi

CONTAINER_NAME="wan22-quant-run-$(date +%Y%m%d-%H%M%S)"
INNER="$LOG_DIR/in-container.sh"

cat > "$INNER" <<'INNER_SCRIPT'
#!/usr/bin/env bash
# 由 run.sh 生成并保存，记录本次容器内实际执行的完整步骤。
set -euo pipefail
export PYTHONPATH=/opt/mslib:/opt/wan
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd /opt/wan

# Wan2.2 源码在 import 期会拉 mindiesd（wan/utils/rainfusion.py）。镜像本身不含它，
# 必须在容器内装入与 py3.12/torch2.10 匹配的 cp312 x86_64 wheel。只装进本容器，
# 不写宿主环境。
echo "==== [0/4] 安装 mindiesd 依赖 ===="
python3 -m pip install --no-index --no-deps --quiet "/deps/$MINDIESD_WHEEL_NAME"
python3 -c "import mindiesd, wan; print('mindiesd + wan import OK')"

echo "==== [1/4] 导出：msModelSlim mindie_format_saver 双专家 ===="
python3 -u -m msmodelslim.cli quant \
  --model_path "$NATIVE_MODEL_IN" \
  --save_path /out/native \
  --device npu \
  --model_type Wan2.2-T2V-A14B \
  --config /tool/c7-smooth.yaml \
  --trust_remote_code True

echo "==== [2/4] 校验原生导出结构 ===="
python3 /tool/bin/validate_export.py --quant-path /out/native

echo "==== [3/4] CPU 转换：原生双专家 + BF16 Diffusers → Omni checkpoint ===="
python3 /tool/bin/merge_mxfp4_checkpoint.py \
  --original-model "$BF16_MODEL_IN" \
  --quant-path /out/native \
  --output-path /out/model \
  --require-smooth-scale \
  --mxfp4-scale-alg 2

echo "==== [4/4] 对账转换结果（原生 vs checkpoint，全量） ===="
python3 /tool/bin/validate_converted.py \
  --quant-path /out/native \
  --original-model "$BF16_MODEL_IN" \
  --output-model /out/model

echo "==== 全部通过 ===="
INNER_SCRIPT
chmod +x "$INNER"

MOUNTS=(
  -v "$NATIVE_MODEL:/in/native-model:ro"
  -v "$BF16_DIFFUSERS_MODEL:/in/bf16:ro"
  -v "$CALIB_DATASET_DIR:/dataset:ro"
  -v "$MSS_MODELSLIM_LIB:/opt/mslib:ro"
  -v "$WAN_SRC:/opt/wan:ro"
  -v "$MINDIESD_WHEEL:/deps/$WHEEL_BASENAME:ro"
  -v "$TOOL_DIR:/tool:ro"
  -v "$OUTPUT_ROOT:/out"
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro
)
if [ -n "$BF16_DIFFUSERS_COMPONENT_ROOT" ]; then
  MOUNTS+=( -v "$BF16_DIFFUSERS_COMPONENT_ROOT:/models:ro" )
fi

echo "容器: $CONTAINER_NAME   镜像: $IMAGE   卡: $DEVICE"
echo "输出根: $OUTPUT_ROOT"

set +e
docker run --rm --name "$CONTAINER_NAME" \
  --privileged --ipc=host --network=host --log-driver=none \
  -e "ASCEND_RT_VISIBLE_DEVICES=$DEVICE" \
  -e "NATIVE_MODEL_IN=/in/native-model" \
  -e "BF16_MODEL_IN=/in/bf16" \
  -e "MINDIESD_WHEEL_NAME=$WHEEL_BASENAME" \
  "${MOUNTS[@]}" \
  "$IMAGE" bash -lc "bash /out/logs/in-container.sh 2>&1 | tee /out/logs/run.log; exit \${PIPESTATUS[0]}"
RC=$?
set -e

echo "退出码: $RC"
if [ "$RC" -ne 0 ]; then
  echo "失败。完整日志: $LOG_DIR/run.log" >&2
  exit "$RC"
fi

{
  echo "status=passed"
  echo "finished_at=$(date -Is)"
  echo "output_root=$OUTPUT_ROOT"
  echo "native=$NATIVE_OUT"
  echo "model=$MODEL_OUT"
} > "$LOG_DIR/run-result.txt"
cat "$LOG_DIR/run-result.txt"
