from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokens", type=int, default=10_000_000)
    parser.add_argument("--out", default="data/fineweb_gpt2_10m_u16.bin")
    parser.add_argument("--meta-out", default="data/fineweb_gpt2_10m_meta.json")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--eos", action="store_true", default=True)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    meta_out = pathlib.Path(args.meta_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    arr = np.memmap(out, dtype=np.uint16, mode="w+", shape=(args.tokens,))
    dataset = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)

    count = 0
    docs = 0
    start = time.perf_counter()
    for row in dataset:
        text = row.get(args.text_field)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if args.eos and eos_id is not None:
            ids.append(eos_id)
        if not ids:
            continue
        room = args.tokens - count
        if len(ids) > room:
            ids = ids[:room]
        arr[count : count + len(ids)] = np.asarray(ids, dtype=np.uint16)
        count += len(ids)
        docs += 1
        if count >= args.tokens:
            break
        if docs % 1000 == 0:
            elapsed = max(time.perf_counter() - start, 1e-9)
            print(f"docs={docs} tokens={count} tok/sec={count / elapsed:.0f}", flush=True)

    arr.flush()
    elapsed = time.perf_counter() - start
    meta = {
        "dataset": args.dataset,
        "name": args.name,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "tokens_requested": args.tokens,
        "tokens_written": count,
        "docs": docs,
        "path": str(out),
        "dtype": "uint16",
        "elapsed_sec": elapsed,
        "tok_per_sec": count / elapsed if elapsed > 0 else None,
    }
    with meta_out.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))
    if count != args.tokens:
        raise SystemExit(f"only wrote {count} of {args.tokens} requested tokens")


if __name__ == "__main__":
    main()
