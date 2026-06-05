"""Diagnostic: reproduce SFT dataloader + model forward to locate the NaN."""
import os
os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from tasks.common import TaskMixture
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K

base_dir = os.environ["NANOCHAT_BASE_DIR"]
device_type = autodetect_device_type()
ddp, rank, lrank, world, device = compute_init(device_type)

model, tokenizer, meta = load_model("base", device, phase="eval")
model.train()
bos = tokenizer.get_bos_token_id()
max_seq_len = 512
row_capacity = max_seq_len + 1

idf = os.path.join(base_dir, "identity_conversations.jsonl")
tasks = [SmolTalk(split="train"), CustomJSON(filepath=idf), CustomJSON(filepath=idf),
         *[MMLU(subset="all", split="auxiliary_train") for _ in range(3)],
         *[GSM8K(subset="main", split="train") for _ in range(4)]]
ds = TaskMixture(tasks)

# length distribution over first 2000 convs (the region the first steps actually touch)
lens = []
overlong = 0
zero_mask = 0
for i in range(2000):
    ids, mask = tokenizer.render_conversation(ds[i])
    lens.append(len(ids))
    if len(ids) > row_capacity:
        overlong += 1
    if sum(mask) == 0:
        zero_mask += 1
import statistics as st
print(f"[lens over first 2000 convs] min={min(lens)} med={int(st.median(lens))} "
      f"mean={int(st.mean(lens))} max={max(lens)} p95={sorted(lens)[int(0.95*len(lens))]}")
print(f"[overlong > {row_capacity}] {overlong}/2000 = {overlong/20:.1f}%   "
      f"[zero assistant-mask convs] {zero_mask}/2000")

# best-fit packing replica
def gen(dbs=4, buffer_size=100):
    buf = []
    cursor = 0
    def refill():
        nonlocal cursor
        while len(buf) < buffer_size:
            ids, mask = tokenizer.render_conversation(ds[cursor % len(ds)])
            buf.append((ids, mask)); cursor += 1
    while True:
        rows, mrows = [], []
        for _ in range(dbs):
            row, mrow = [], []
            while len(row) < row_capacity:
                while len(buf) < buffer_size: refill()
                rem = row_capacity - len(row)
                bi, bl = -1, 0
                for j,(c,_) in enumerate(buf):
                    if len(c) <= rem and len(c) > bl: bi, bl = j, len(c)
                if bi >= 0:
                    c, cm = buf.pop(bi); row += c; mrow += cm
                else:
                    row += [bos]*rem; mrow += [0]*rem; break
            rows.append(row[:row_capacity]); mrows.append(mrow[:row_capacity])
        yield rows, mrows

g = gen()
for step in range(1, 9):
    rows, mrows = next(g)
    bt = torch.tensor(rows, dtype=torch.long)
    x = bt[:, :-1].to(device=device, dtype=torch.int32).contiguous()
    y = bt[:, 1:].to(device=device, dtype=torch.int64).contiguous()
    mt = torch.tensor(mrows, dtype=torch.int8)[:, 1:].to(device=device)
    y[mt == 0] = -1
    valid_per_row = [(y[r] != -1).sum().item() for r in range(y.size(0))]
    total_valid = sum(valid_per_row)
    with torch.no_grad():
        loss = model(x, y)
    print(f"step {step}: valid_targets/row={valid_per_row} total={total_valid} "
          f"loss={loss.item():.4f} nan={torch.isnan(loss).item()}")
