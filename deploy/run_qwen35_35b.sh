#!/bin/bash
# ============================================================
# Qwen3.5-35B-A3B on vllm-ascend Docker
# TP=2 across 2x Ascend 910B3 (910_9362)
# ============================================================

MODEL_DIR="/models/Qwen"
MODEL_NAME="Qwen3.5-35B-A3B"   # ModelScope symlink target name
IMAGE="quay.io/ascend/vllm-ascend:v0.20.2rc1-a3"
PORT=8000

# NPU devices (check with: ls /dev/davinci*)
DEVICES=(
    /dev/davinci4
    /dev/davinci5
    /dev/davinci_manager
    /dev/hisi_hdc
)

# Build --device args
DEVICE_ARGS=""
for dev in "${DEVICES[@]}"; do
    DEVICE_ARGS="$DEVICE_ARGS --device=$dev"
done

echo "============================================================"
echo " Qwen3.5-35B-A3B  |  TP=2  |  Ascend 910B3 x2"
echo " Model: $MODEL_NAME"
echo " Port:  $PORT"
echo "============================================================"
echo ""

sudo docker run --rm --privileged --net=host \
    $DEVICE_ARGS \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /mnt/workspace/models:/models \
    $IMAGE \
    vllm serve "$MODEL_DIR/$MODEL_NAME" \
        --served-model-name qwen35-35b \
        --host 0.0.0.0 \
        --port "$PORT" \
        --trust-remote-code \
        --dtype bfloat16 \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --enforce-eager
