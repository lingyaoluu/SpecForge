export CUDA_VISIBLE_DEVICES=4,5
torchrun --nproc_per_node=2 \
    /home/qjsys/lly/SpecForge/scripts/prepare_hidden_states.py \
    --model-path /home/qjsys/llm_models/Qwen3-32B \
    --enable-aux-hidden-states \
    --data-path /home/qjsys/lly/SpecForge/data/sharegpt/processed/sharegpt.jsonl \
    --output-path /home/qjsys/lly/SpecForge/cache/sharegpt \
    --chat-template qwen \
    --max-length 2048 \
    --tp-size 2 \
    --batch-size 2 \
    --mem-frac=0.8 \
    --dist-timeout 1800
    # --num-samples 4 \