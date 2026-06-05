# RTX 4060 SFT Fixes

Three bugs prevented SFT from training on the RTX 4060. All three are now fixed. This document
records what each bug was, how it was diagnosed, and how it was fixed.

---

## Bug 1 — Crash: `optimizer.step()` called on None gradients

**File:** `scripts/chat_sft.py`

**Symptom:** Training crashed at the first step where all microbatches had non-finite loss:

```
TypeError: expected Tensor as element 0 in argument 0, but got NoneType
  File "nanochat/optim.py", line 259, in _step_muon
    stacked_grads = torch.stack([p.grad for p in params])
```

**Root cause:** The original training loop called `model.zero_grad(set_to_none=True)` at the
end of each step, setting all gradients to `None`. On a step where every microbatch had
non-finite loss, no `backward()` call was made, so gradients stayed `None`. The subsequent
`clip_grad_norm_()` returned 0.0 (treating None as zero — finite), so `optimizer.step()` was
called with `None` tensors, crashing the Muon optimizer's `torch.stack` call.

**Fix:** Track whether any microbatch in the step ran a backward pass. If none did, skip the
optimizer step entirely.

```python
any_finite_microbatch = False
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    if torch.isfinite(loss):
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        any_finite_microbatch = True
    else:
        num_skipped_microbatches += 1
    ...

if not any_finite_microbatch:
    num_skipped_steps += 1
    print0(f"step {step:05d} | all microbatches non-finite; skipping optimizer step")
else:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        num_skipped_steps += 1
        print0(f"step {step:05d} | non-finite grad norm; skipping optimizer step")
```

---

## Bug 2 — Root cause: packer buffer deadlock on oversized conversations (most impactful)

**File:** `scripts/chat_sft.py` — `sft_data_generator_bos_bestfit` / `refill_buffer`

**Symptom:** After 2–3 real training steps, every subsequent step ran in ~48ms (vs ~180ms for
real steps), all reported the same stale loss value, and `num_skipped_steps` climbed to 1500.
The model did no real training.

**Root cause:** The best-fit packer only packs conversations that fit entirely within
`row_capacity = max_seq_len + 1 = 513` tokens. MMLU conversations formatted with the question,
four labeled answer choices, and chat template tokens can exceed 513 tokens. These oversized
conversations are never selected by the packer, but they are still added to the 100-item
`conv_buffer`. Once the buffer filled with only oversized conversations:

1. Every row found no fitting conversation → `best_idx == -1`
2. The row was padded entirely with BOS tokens (`mask=0`)
3. All target positions were masked to `-1` (ignore_index)
4. `cross_entropy(reduction='mean')` over zero unmasked tokens returned NaN
5. Every microbatch was skipped → no backward → no optimizer step → model never updated

This happened consistently because the packer consumed the short MMLU questions (which fit)
first, then the buffer refilled with only the long ones.

Confirmed with a diagnostic: `valid_targets=0` for the first microbatch of the failing step,
while `weights_inf=False` — the model weights were completely healthy. The NaN came from the
data pipeline, not from model divergence.

**Fix:** Filter oversized conversations in `refill_buffer` before adding them to the buffer.
This ensures the buffer always contains packable conversations.

```python
def refill_buffer():
    nonlocal cursor, epoch
    while len(conv_buffer) < buffer_size:
        conversation = dataset[cursor]
        ids, mask = tokenizer.render_conversation(conversation)
        # Only add conversations that can fit in a row. Conversations longer than
        # row_capacity would never be selected by the best-fit packer and would
        # accumulate in the buffer, eventually causing all rows to be padded with
        # mask=0 BOS tokens, producing all-masked batches and NaN loss.
        if len(ids) <= row_capacity:
            conv_buffer.append((ids, mask))
        cursor += ddp_world_size
        ...
```

Long MMLU questions are excluded from SFT training under the 512-token limit. They remain
available for evaluation.

---

## Bug 3 — LR instability at SFT start

**File:** `runs/rungpu.sh`

**Symptom:** With upstream defaults, the SFT loss jumped from ~1.8 to NaN within 3 steps.
The pre-clip gradient norm at step 1 was 15.6× above the clip threshold.

**Root cause:** Three compounding issues:

1. **No LR warmup.** The `progress` counter advances ~0.6% per optimizer step (9 data
   iterations per step, num_iterations=1500), so `--warmup-ratio=0.05` completed in only
   ~8 steps — far too few for the model to adapt to new chat special tokens.
2. **High initial LR fraction.** The default `--init-lr-frac=0.8` started at 80% of the
   pretraining peak LR immediately.
3. **Warm optimizer momentum.** Loading the pretrained optimizer state carried over stale
   momentum from pretraining, amplifying the first few SFT updates.

**Fix:** Three flags added to the SFT command in `rungpu.sh`:

```bash
--warmup-ratio=0.2   # ramp LR over ~33 steps instead of ~8
--init-lr-frac=0.1   # peak SFT LR is 10% of pretraining LR
--load-optimizer=0   # fresh optimizer, no stale pretraining momentum
```

Gradient clipping (`max_norm=1.0`) was also added to the training loop as a safety net for
rare loss spikes.

---

## Results

| Metric | Before fixes | After fixes |
|--------|-------------|-------------|
| Steps with real gradients | 2 / 188 | **188 / 188** |
| Validation bpb | 1.063 (unchanged from base) | **0.779** |
| ChatCORE | 0.004 | **0.053** (12.5×) |
| SpellingBee | 0% | **29.2%** |
| Training time (SFT) | 0.16 min (all skipped) | **0.55 min** |
| Peak VRAM | 2.5 GB | **2.2 GB** |


## validation
1. Chat with it in the CLI — fastest sanity check:
source .venv/bin/activate
python -m scripts.chat_cli -p "What is the capital of France?"
Try a few prompts. The model should be able to identify itself and answer basic factual questions.

2. Run the full ChatCORE eval — same benchmark used during training:
source .venv/bin/activate
python -m scripts.chat_eval

3. Chat in the browser UI:
source .venv/bin/activate
python -m scripts.chat_web

4. Run with a specific checkpoint (if you want to compare steps):
python -m scripts.chat_cli --model-tag chatsft --model-step 188 -p "Hi"

The --source flag is required. For the SFT model you just trained:

source .venv/bin/activate

### Run all eval tasks on the SFT model
python -m scripts.chat_eval -i sft

### Run a specific task only (faster)
python -m scripts.chat_eval -i sft -a ARC-Easy
python -m scripts.chat_eval -i sft -a MMLU
python -m scripts.chat_eval -i sft -a "ARC-Easy|ARC-Challenge|MMLU"

### Limit problems per task for a quick sanity check
python -m scripts.chat_eval -i sft -x 50

### Load a specific checkpoint step
python -m scripts.chat_eval -i sft -s 188

The -i sft tells it to load from chatsft_checkpoints/ (vs -i ut -s, it picks the latest checkpoint automatically.

Note: training is still running (rungpu3.log), so if you run heckpoint from the last run — the d8 model isn't ready yet.
