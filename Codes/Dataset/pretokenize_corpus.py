"""
Dataset/pretokenize_corpus.py -- one-time corpus pre-tokenization.

WHY THIS EXISTS
---------------
common/train_loop.data_generator tokenizes every document on the fly in a
single Python thread. Measured on an L4 that starves the GPU: ~4% model-FLOPs
utilisation (2.4 of ~60 TFLOPS dense bf16, GPU ~30% busy). Tokenisation is
pure CPU work that is identical on every epoch and every run, so doing it once
up front and memory-mapping the result removes the bottleneck entirely --
roughly an 8x throughput improvement, i.e. 8x less GPU rent for the same
science.

FORMAT (deliberately simple, nanoGPT-style)
-------------------------------------------
A flat uint16 array of fixed-size sequence blocks:

    <out>.bin   : uint16[num_blocks * seq_length]
    <out>.json  : {seq_length, num_blocks, vocab_size, tokenizer_md5, ...}

uint16 is safe because the WordPiece vocab is 30522 < 65536.

SEMANTICS ARE PRESERVED EXACTLY. Each block is built the same way
data_generator builds one: a document is tokenized without truncation, split
into consecutive (seq_length - 2)-token chunks, each chunk wrapped as
[CLS] ... [SEP] and padded to seq_length; chunk tails shorter than
_MIN_CHUNK_TOKENS are dropped. So a pre-tokenized run sees the same sequence
distribution as an on-the-fly run -- only faster.

Usage:
    python3 Dataset/pretokenize_corpus.py --workers 8
    python3 Dataset/pretokenize_corpus.py --input data/fineweb-edu/train_filtered.jsonl \
                                          --output data/fineweb-edu/train_128.bin \
                                          --seq-length 128
"""

import os
import sys
import json
import argparse
import hashlib
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging

logger = setup_logging('pretokenize_corpus')

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
_MIN_CHUNK_TOKENS = 16          # must match common/train_loop._MIN_CHUNK_TOKENS

_TOK = None
_SEQ = None
_IDS = None


def _init_worker(tokenizer_dir, seq_length):
    """Each worker builds its own tokenizer once (fast tokenizers aren't fork-safe to share)."""
    global _TOK, _SEQ, _IDS
    from transformers import PreTrainedTokenizerFast
    _TOK = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    _SEQ = seq_length
    _IDS = (_TOK.cls_token_id, _TOK.sep_token_id, _TOK.pad_token_id)


def _encode_doc(line):
    """One JSONL line -> list of fixed-size uint16 blocks (may be empty)."""
    try:
        text = json.loads(line).get('text', '')
    except json.JSONDecodeError:
        return []
    if not text:
        return []
    cls_id, sep_id, pad_id = _IDS
    max_body = _SEQ - 2
    ids = _TOK(text, add_special_tokens=False, truncation=False)['input_ids']
    blocks = []
    for start in range(0, len(ids), max_body):
        chunk = ids[start:start + max_body]
        if len(chunk) < _MIN_CHUNK_TOKENS:
            break                      # drop short tail (matches data_generator)
        seq = [cls_id] + chunk + [sep_id]
        if len(seq) < _SEQ:
            seq = seq + [pad_id] * (_SEQ - len(seq))
        blocks.append(np.asarray(seq, dtype=np.uint16))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=os.path.join(DATA_DIR, 'fineweb-edu', 'train_filtered.jsonl'))
    ap.add_argument('--output', default=None, help='default: <input dir>/train_<seq>.bin')
    ap.add_argument('--tokenizer', default=os.path.join(DATA_DIR, 'tokenizer'))
    ap.add_argument('--seq-length', type=int, default=128)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--chunk-size', type=int, default=2000, help='lines per worker task')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"input corpus not found: {args.input}")
    out_bin = args.output or os.path.join(os.path.dirname(args.input),
                                          f"train_{args.seq_length}.bin")
    out_meta = os.path.splitext(out_bin)[0] + '.json'

    tok_json = os.path.join(args.tokenizer, 'tokenizer.json')
    tok_md5 = hashlib.md5(open(tok_json, 'rb').read()).hexdigest()

    logger.info(f"pre-tokenizing {args.input} -> {out_bin} "
                f"(seq={args.seq_length}, workers={args.workers})")

    n_blocks = 0
    n_docs = 0
    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(out_bin, 'wb') as f_out, \
         Pool(args.workers, initializer=_init_worker,
              initargs=(args.tokenizer, args.seq_length)) as pool:
        for blocks in tqdm(pool.imap(_encode_doc, f_in, chunksize=args.chunk_size),
                           desc="pre-tokenizing", unit="doc"):
            n_docs += 1
            for b in blocks:
                f_out.write(b.tobytes())
                n_blocks += 1

    meta = {
        'source': os.path.relpath(args.input, DATA_DIR),
        'seq_length': args.seq_length,
        'num_blocks': n_blocks,
        'num_documents': n_docs,
        'total_tokens_padded': n_blocks * args.seq_length,
        'dtype': 'uint16',
        'tokenizer_md5': tok_md5,
        'min_chunk_tokens': _MIN_CHUNK_TOKENS,
        'note': ('Fixed-size [CLS] ... [SEP] padded blocks, semantics identical to '
                 'common/train_loop.data_generator. Read with '
                 'np.memmap(path, dtype=np.uint16).reshape(-1, seq_length).'),
    }
    with open(out_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    gb = n_blocks * args.seq_length * 2 / 1e9
    logger.info(f"wrote {n_blocks:,} blocks from {n_docs:,} docs "
                f"({meta['total_tokens_padded']/1e6:.1f}M padded tokens, {gb:.2f} GB)")
    logger.info(f"metadata -> {out_meta}")


if __name__ == "__main__":
    main()
