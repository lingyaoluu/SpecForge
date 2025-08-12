export CUDA_VISIBLE_DEVICES=2
torchrun --nproc_per_node=1 \
    --master_port 29507 \
    /home/qjsys/lly/SpecForge/scripts/prepare_hidden_states.py \
    --model-path /home/qjsys/llm_models/Qwen3-32B \
    --enable-aux-hidden-states \
    --data-path /home/qjsys/lly/SpecForge/warn_dataset/gc/processed/alarm_eval_clean.jsonl \
    --output-path /home/qjsys/lly/SpecForge/cache/eval_hidden_states \
    --chat-template qwen \
    --max-length 2048 \
    --tp-size 1 \
    --batch-size 2 \
    --mem-frac=0.8 \
    --dist-timeout 1800 \
    # --num-samples 4 \