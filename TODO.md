# nanochat TODO

## Glossary

**depth** — Number of transformer layers. More layers = more sequential reasoning steps, but slower and more memory-intensive. The single complexity dial in nanochat; most other hyperparameters are derived from it.

**aspect_ratio** — Controls model width relative to depth. `n_embd = depth × aspect_ratio` (rounded up to a multiple of `head_dim`). Higher aspect ratio = wider, shallower model. Wider models tend to store more knowledge per layer; deeper models tend to compose reasoning over more steps.

**n_embd** — Embedding dimension (model width). The size of the vector used to represent each token at every layer. All internal computations happen in this space. Larger = more expressive but more memory.

**n_head** — Number of attention heads. Each head attends to different parts of the context independently. Derived as `n_embd / head_dim`. More heads = finer-grained attention patterns.

**head_dim** — Size of each attention head's key/query/value vectors (default 128). Fixed; `n_head` is derived from `n_embd / head_dim`.

**n_kv_head** — Number of key/value heads (Group-Query Attention). Fewer KV heads than query heads reduces memory and compute during inference while keeping most quality. Currently set equal to `n_head` (no GQA reduction).

**seq_len** — Maximum context length (tokens). Longer = the model can read and generate longer conversations, but memory scales with seq_len² in attention and training time roughly doubles per 2× increase.

**param_data_ratio** — Number of training tokens = `ratio × num_params`. Higher ratio = each parameter sees more data, improving generalization. Chinchilla optimum is ~20 for general LLMs; small models benefit from higher ratios.

**bf16_mfu** — BFloat16 Model FLOPs Utilization. Percentage of the GPU's theoretical peak FLOP/s actually used. Higher = more efficient. 50–60% is excellent for a single GPU; gaps are due to memory bandwidth, kernel overhead, etc.

**bpb** — Bits per byte. Language model loss normalized by token byte length, making it comparable across different tokenizers and vocab sizes. Lower = better. Human-level English is ~1.0 bpb.

---

## Model Configuration Comparisons

### Runs completed

| Config | depth | ar | n_embd | seq_len | ratio | params (est.) | pretrain bpb | MFU | pretrain time |
|---|---|---|---|---|---|---|---|---|---|
| d12/ar64 | 12 | 64 | 768 | 512 | 4 | 286M | 0.979 | 51% | ~5.5 hrs |
| d8/ar96 | 8 | 96 | 768 | 1024 | 8 | ~150M | in progress | 60% | ~5.3 hrs |

### Candidate next runs

| Config | depth | ar | n_embd | n_head | seq_len | Notes |
|---|---|---|---|---|---|---|
| d10/ar80 | 10 | 80 | 896 | 7 | 1024 | Mid-point between d8 and d12; fits 8GB comfortably |
| d10/ar96 | 10 | 96 | 1024 | 8 | 1024 | Likely tight on 8GB at batch=4; may OOM |
| d12/ar96 | 12 | 96 | 1152 | 9 | 1024 | Larger, probably OOM |

### Key observations
- d8/ar96 reached bpb 1.090 at step 5000; d12/ar64 needed 10000 steps — same quality in half the steps
- d8/ar96 runs at 60% MFU vs 51% for d12/ar64 — shallower model is more compute-efficient on RTX 4060
- Wider/shallower (higher aspect_ratio) is better for knowledge-heavy tasks at small scale

---

## Pending improvements

### Architecture / training
- [ ] Try d10/ar80 after d8/ar96 completes — natural next step if d8 underfits
- [ ] Evaluate whether RL (chat_rl.py) helps after SFT — needs non-zero GSM8K pass rate to get gradient signal
- [ ] Consider GQA (n_kv_head < n_head) to reduce inference memory

### SFT
- [ ] Current mixture: SmolTalk ×2, identity ×2, GSM8K ×4, SimpleSpelling, SpellingBee (MMLU dropped)
- [ ] Evaluate whether dropping GSM8K also makes sense if RL is not run
- [ ] Consider distillation: generate SFT data with a large model (Claude/GPT-4) for better signal density

### Sudoku reasoning task
- [ ] Add `tasks/sudoku.py` — generate puzzle/solution pairs programmatically (unlimited free data, binary eval signal)
- [ ] Train with step-by-step chain-of-thought solutions, not just puzzle→answer (critical for constraint propagation)
- [ ] Curriculum: start with easy puzzles (35+ givens), gradually increase difficulty
- [ ] Use RL after SFT (adapt `chat_rl.py`) with binary correct/wrong reward — same structure as GSM8K
- [ ] Realistic targets at d8 scale: easy puzzles reliably, medium 30–60% pass rate, hard unlikely without search

### Multimodal / Vision (LLaVA-style)

Architecture: CLIP ViT-B/32 vision encoder (~86M params, frozen) → linear projection (vision_dim→n_embd) → nanochat LLM. Each image becomes 196 patch tokens prepended to the text sequence. This is the LLaVA-1 design: cheap to train, no architectural surgery on the LLM.

Feasibility on RTX 4060:
- ViT-B/32 + 768-wide LLM fits in 8GB with batch=1–2 at seq_len=1024+196
- Training: freeze vision encoder, train projection + LLM end-to-end on image-text pairs
- Inference: ~same speed as text-only (196 extra tokens per image)

What to build:
- [ ] Add `nanochat/vision.py` — load CLIP ViT-B/32 via `open_clip`, extract patch embeddings, linear projection to n_embd
- [ ] Modify `nanochat/gpt.py` — accept optional `image_embeddings` tensor, prepend to token embeddings before transformer layers
- [ ] Add `tasks/vqa.py` — wrap LLaVA-Instruct or similar VQA dataset into the Task interface
- [ ] Add `scripts/vision_sft.py` — SFT script that feeds image+text batches; requires DataLoader that returns (pixel_values, input_ids, labels)
- [ ] Realistic targets at d8 scale: image captioning, basic VQA (yes/no, color, count), object identification — not fine-grained reasoning

Training data options (all freely available):
- LLaVA-Instruct-150K (GPT-4V generated, 150K image-text pairs)
- COCO Captions (330K images, 5 captions each)
- TextVQA (documents/signs — tests OCR + reasoning)

### Infrastructure
- [ ] Re-run base_eval on d12 pretrained model with max_seq_len fix (was crashing before fix)
- [ ] Clean up intermediate d12 SFT checkpoints (steps 0–1600) once d8 run is validated
