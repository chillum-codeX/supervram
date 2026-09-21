#!/usr/bin/env python3
"""Compare two svram-verify outputs (JSON + optional raw logits dumps).

Reports whether greedy token ids match (the hard gate), whether logits hashes are bit-identical,
and, when dumps are given, max-abs / max-rel differences and argmax agreement per step.
Exit status 0 when token ids match for the compared prefix, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a_json", type=Path)
    parser.add_argument("b_json", type=Path)
    parser.add_argument("--a-logits", type=Path)
    parser.add_argument("--b-logits", type=Path)
    parser.add_argument("--n-vocab", type=int, default=151936)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    a = json.loads(args.a_json.read_text())
    b = json.loads(args.b_json.read_text())
    n = min(len(a["tokens"]), len(b["tokens"]))
    first_div = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
    hash_equal = a["logits_hashes"][:n] == b["logits_hashes"][:n]

    report = {
        "a": {"mode": a["mode"], "ngl": a["ngl"], "cpu_moe": a.get("cpu_moe"), "cache_mib": a.get("cache_mib"), "decode_tps": a.get("decode_tps")},
        "b": {"mode": b["mode"], "ngl": b["ngl"], "cpu_moe": b.get("cpu_moe"), "cache_mib": b.get("cache_mib"), "decode_tps": b.get("decode_tps")},
        "compared_steps": n,
        "tokens_identical": first_div is None,
        "first_divergence_step": first_div,
        "logits_hashes_identical": hash_equal,
    }

    if args.a_logits and args.b_logits and args.a_logits.exists() and args.b_logits.exists():
        la = np.fromfile(args.a_logits, dtype=np.float32)
        lb = np.fromfile(args.b_logits, dtype=np.float32)
        steps = min(la.size, lb.size) // args.n_vocab
        la = la[: steps * args.n_vocab].reshape(steps, args.n_vocab)
        lb = lb[: steps * args.n_vocab].reshape(steps, args.n_vocab)
        diff = np.abs(la - lb)
        report["logits"] = {
            "steps": steps,
            "max_abs_diff": float(diff.max()) if steps else None,
            "mean_abs_diff": float(diff.mean()) if steps else None,
            "max_rel_diff": float((diff / (np.abs(la) + 1e-6)).max()) if steps else None,
            "argmax_agreement": float((la.argmax(1) == lb.argmax(1)).mean()) if steps else None,
            "within_atol": bool(diff.max() <= args.atol) if steps else None,
            "atol": args.atol,
        }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    sys.exit(0 if report["tokens_identical"] else 1)


if __name__ == "__main__":
    main()
