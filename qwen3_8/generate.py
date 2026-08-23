"""Generate Turkish names from a trained Qwen3.8 checkpoint.

Run:  python generate.py [count] [temperature]
"""

import sys
import torch

from model import Qwen38Text
from tokenizer import CharTokenizer

CHECKPOINT = "tiny_qwen38.pt"


def load():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    tokenizer = CharTokenizer(ckpt["chars"])
    model = Qwen38Text(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_names(model, tokenizer, n, temperature, top_k=None):
    start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long)
    out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                         temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    names = [tokenizer.decode(row[1:]).split("\n")[0] for row in out.tolist()]
    return [n for n in names if n]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    model, tokenizer = load()
    for name in generate_names(model, tokenizer, n, temperature):
        print(name)


if __name__ == "__main__":
    main()
