# Training Explained

A plain-language walkthrough of every step in `runs/speedrun.sh`.

---

## 1. Tokenizer Training

Builds the vocabulary that maps text into numbers the model can process.

**Algorithm:** Byte-Pair Encoding (BPE). Reads ~2B characters of text and repeatedly finds the most frequent character pair, merging it into a single token. After 32,503 merges you have a vocabulary of 2¹⁵ = 32,768 tokens covering common words, subwords, and individual characters.

**Example progression:** `t`+`h` → `th`, then `th`+`e` → `the`, then `the`+` ` → `the `, etc.

**Why it matters:** A good tokenizer compresses text efficiently. Fewer tokens per sentence means the model fits more context in its fixed 512-token window and trains faster. The compression ratio is reported by `tok_eval` after training.

**In this speedrun:** Trained on 8 data shards (~2B chars), producing the vocab used by all subsequent steps.

---

## 2. Pretraining

Teaches the model to predict the next token, over and over, on a large corpus of text.

**The core loop:**
1. Take a 512-token chunk of text
2. For every position, ask: "given all tokens so far, what comes next?"
3. Measure how wrong it was (cross-entropy loss)
4. Nudge the weights to be less wrong (backprop + optimizer step)
5. Repeat ~70,000 times across ~1B tokens of text

No labels, no human feedback — the text itself is the supervision signal.

**What the model learns:** By predicting the next token across billions of examples, the model implicitly learns grammar, facts, reasoning patterns, and code syntax — because knowing those things helps it predict better. None of it is taught explicitly.

**What it produces:** A "base model" that is very good at continuing text. If you give it "The capital of France is", it will say "Paris." But it does not know how to have a conversation or follow instructions yet.

**Quality metric:** Bits-per-byte (BPB) on held-out text. Lower is better.

**In this speedrun:** 12-layer transformer (d12) trained from random weights on ~1B tokens of ClimbMix web text. The longest phase by far (~10+ hours on an RTX 4060).

---

## 3. SFT (Supervised Fine-Tuning)

Teaches the base model to be a useful assistant rather than just a text predictor.

**The problem with base models:** If you ask "What is the capital of France?", a base model might respond "What is the capital of Germany? What is the capital of Spain?" — because on the internet, questions are often followed by more questions. It is completing text, not answering you.

**What SFT does:** Same training loop as pretraining, but the data is conversations in a structured format:

```
[USER] Why is the sky blue?
[ASSISTANT] Because of Rayleigh scattering...
```

Loss is computed only on the assistant turns. After thousands of these examples, the model learns: when a human asks something, produce a helpful answer.

**What it teaches specifically:**
- Conversation structure — special tokens marking turn boundaries
- Personality — from `identity_conversations.jsonl` (2.3MB of synthetic conversations)
- Format following — tool use syntax, multiple choice, etc.

| | Pretraining | SFT |
|--|--|--|
| Data | Raw web text (~1B tokens) | Curated conversations |
| Duration | ~10 hours | ~30 minutes |
| Goal | Learn world knowledge | Learn to be an assistant |
| Loss computed on | Every token | Assistant turns only |

---

## 4. Evaluation

After each major phase, the model is evaluated against benchmarks.

- **Base eval** (after pretraining): BPB on train/val splits, plus text samples
- **Chat eval** (after SFT): ARC, MMLU, GSM8K, HumanEval, SpellingBee, and a ChatCORE composite score

Results are collected into a final Markdown report written to `~/.cache/nanochat/report/`.

---

## What comes after (not in this speedrun)

### RLHF / Preference Training
Show the model's outputs to humans or another model, collect rankings of which response is better, then train to produce the preferred ones.
- **PPO** — train a separate reward model, optimize against it with RL. Powerful but complex and unstable.
- **DPO** (Direct Preference Optimization) — skip the reward model, optimize directly on preference pairs. Simpler, now more common.

### Constitutional AI / RLAIF
Use another AI as the judge instead of humans. Cheaper and faster than human feedback.

### Safety / Alignment Tuning
Targeted fine-tuning to reduce harmful outputs. Usually applied after preference training.

---

## Fine-Tuning an Established Model

Instead of training from scratch (like this speedrun does), you can start from a model that already knows a lot and teach it something specific. This is called fine-tuning, and it is usually far cheaper than pretraining.

**The idea:** A large pretrained model like Llama 3 has already spent millions of dollars of compute learning general language understanding. Fine-tuning borrows all of that and adds a narrow skill on top, using a fraction of the data and time.

**Example — fine-tuning Llama 3 for medical Q&A:**

Suppose you want a model that answers clinical questions accurately and refuses to speculate without evidence. You would:

1. Start from `meta-llama/Meta-Llama-3-8B-Instruct` (already SFT'd)
2. Assemble a dataset of ~10,000 high-quality medical Q&A pairs written or reviewed by clinicians
3. Run SFT on that dataset for 1–3 epochs — just enough to shift the model's style and knowledge, not so much that it forgets general language ability
4. Evaluate on a held-out medical benchmark (e.g. MedQA)

Total compute: a few hours on a single GPU, versus months for pretraining from scratch.

**The risk — catastrophic forgetting:** If you fine-tune too aggressively (too many epochs, too high a learning rate), the model overwrites the general weights and gets worse at everything outside your narrow domain. The usual mitigations:

- **Low learning rate** — typically 10–100× smaller than pretraining
- **Few epochs** — often just 1–3 passes over the fine-tuning data
- **LoRA (Low-Rank Adaptation)** — instead of updating all weights, inject small trainable rank-decomposition matrices alongside the frozen original weights. The base model is untouched; only the adapters are trained. Memory-efficient and easy to swap out.

**LoRA in practice:**

```
Original weight matrix W  (frozen)
        +
Low-rank adapter: W' = W + A·B   (A and B are tiny, trained)
```

A and B together might have 0.1% of the parameters of W. You get most of the fine-tuning benefit at a fraction of the cost, and you can distribute just the adapter (a few MB) rather than the full model.

**When to fine-tune vs. other options:**

| Approach | When to use |
|--|--|
| Fine-tuning (full) | You have enough data and want deep behavior change |
| LoRA / QLoRA | Limited GPU memory, or want to keep the base model intact |
| Few-shot prompting | Very little data; just show examples in the prompt |
| RAG | Knowledge is external and changes frequently |

---

## RAG vs LoRA

Both solve the same surface problem — "the model doesn't know enough about my domain" — but they attack it at completely different layers.

**LoRA** changes the model's weights. After training, the knowledge is baked in; the model answers from memory.

**RAG (Retrieval-Augmented Generation)** leaves the model's weights alone. At inference time, it fetches relevant documents from an external store and stuffs them into the prompt. The model reads the documents and answers from them, like an open-book exam.

```
LoRA:  [question] → model (knowledge baked in) → answer

RAG:   [question] → retriever → relevant docs
                                      ↓
                   [question + docs] → model → answer
```

### When each wins

**Use LoRA when:**
- Your knowledge is stable and won't change often (a legal style guide, a company's internal coding conventions)
- You want a change in *behavior or tone*, not just facts (the model should always respond formally, or always output JSON)
- Latency matters — RAG adds a retrieval round-trip before every response
- You can't fit the relevant documents in a prompt (highly specialized knowledge spread across thousands of sources)

**Use RAG when:**
- Your knowledge changes frequently (a support bot over a product docs site that ships weekly)
- You need citations — RAG can return the source documents alongside the answer, making it auditable
- Your dataset is too large or sensitive to train on (a company's private document store)
- You want to swap knowledge without retraining — just update the index

### A concrete example

Suppose you're building a customer support bot for a software product.

- **LoRA approach:** Fine-tune on past support tickets and resolved issues. The model learns the product's terminology and common fixes. Works well until the product changes significantly — then you retrain.
- **RAG approach:** Index the product docs, changelog, and known issues into a vector database. On each query, retrieve the top-5 most relevant chunks and pass them to the model. When docs are updated, re-index — no retraining.

Most production systems end up using **both**: LoRA (or full fine-tuning) to teach the model the right *format, tone, and behavior*, and RAG to supply *current factual knowledge*. The fine-tuned model knows how to talk; RAG tells it what to say.

### The failure modes

| | LoRA | RAG |
|--|--|--|
| Knowledge goes stale | Yes — requires retraining | No — update the index |
| Hallucination | Still possible (baked-in wrong answers) | Reduced, but model can still misread retrieved docs |
| Latency | None at inference | Adds retrieval round-trip |
| Hard to audit | Yes — why did it say that? | No — you can inspect what was retrieved |
| Needs training data | Yes — labeled examples required | No — works with raw documents |

---

## A note on image tokenization

Text tokenization is a one-time preprocessing step. Images work differently:

- **Patch tokenization (ViT-style):** Divide the image into 16×16 pixel patches, project each into a vector. No discrete tokens — the model sees continuous embeddings.
- **VQ-VAE (discrete tokens):** A learned encoder maps each patch to the nearest entry in a fixed codebook (e.g. 8192 entries). That index is the token — an integer, just like text. Used in the original DALL-E and ImageGPT.
- **Modern multimodal models (LLaVA, Gemini):** Use continuous patch embeddings projected into the LLM's embedding space via a small adapter. No integer image tokens at all.

Unlike BPE (a deterministic counting algorithm), VQ-VAE codebook learning is gradient descent on a reconstruction objective — the "tokenizer" is trained jointly with the model.
