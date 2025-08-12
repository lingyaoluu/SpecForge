SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export CUDA_VISIBLE_DEVICES=4
# train eagle3 for qwen3-coder
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path /home/qjsys/llm_models/Qwen3-32B \
    --draft-model-config $ROOT_DIR/configs/qwen3-32b-eagle3.json \
    --train-data-path /home/qjsys/lly/SpecForge/warn_dataset/gc/processed/alarm_clean.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states \
    --output-dir $ROOT_DIR/outputs/Qwen3-32B_Eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --seed 42 \
    --tp-size $NUM_GPUS 
    # --wandb \
    # --wandb-key 58091b7ab7ea822a0f6898052f6f415a605d694d \
    # --wandb-project Qwen3-32b_eagle3_offline \
    # --wandb-name "train" \
    # --resume
