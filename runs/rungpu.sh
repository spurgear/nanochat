#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# This script was last updated/tuned on Jan 17, 2026.

# Run as:
# bash runs/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

# all the setup stuff
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~2B characters (~34 seconds on my MacBook Pro M3 Max)
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# Train a d8 model (125M params, n_embd=512) tuned for ~30 min on an RTX 4060 (8 GB VRAM).
# Compared to the original d6 (73M params, n_embd=384):
#   - 1.7× more parameters → meaningfully smarter
#   - device_batch_size=8 → halves grad_accum kernel launches for better GPU utilisation
#   - num_iterations=4000 → fewer steps than d6's 5000, offset by the larger model capacity
# Estimated time: ~19 min (vs ~15 min for d6/5000).
python -m scripts.base_train \
    --depth=8 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=8 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=4000 \
    --device-type=cuda \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# SFT (~10 minutes on my MacBook Pro M3 Max)
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# Stability notes for the d8 / max_seq_len=512 config on an RTX 4060:
# - The base model was pretrained on raw text and has never seen the chat special tokens,
#   so SFT starts with very large gradients on the new token embeddings. With the upstream
#   defaults (no LR warmup, init_lr_frac=0.8, warm-started optimizer momentum) and no
#   gradient clipping, the loss explodes to NaN within ~3 steps and the model becomes
#   gibberish. The flags below stabilize the start:
#     --warmup-ratio=0.2   : ramp LR up over ~33 steps; 0.05 only gives ~8 steps (too few)
#     --init-lr-frac=0.1   : smaller SFT peak LR than the 0.8 default
#     --load-optimizer=0   : start with a fresh optimizer (no stale pretraining momentum)
# - num-iterations=2800 gives ~700 optimizer steps (4 data iters per step with batch=8).
#   This is ~3.7× more SFT than the d6 default, fitting in ~3.5 min.
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=8 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=2800 \
    --warmup-ratio=0.2 \
    --init-lr-frac=0.1 \
    --load-optimizer=0 \
    --device-type=cuda \
    --run=$WANDB_RUN

# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
# python -m scripts.chat_cli -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
