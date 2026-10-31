#!/usr/bin/env python3
"""Step 2: 2,000 elicited prompts from the 40 books + 500 UltraChat."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_common import (  # noqa: E402
    DATA_DIR,
    MODEL_ID,
    State,
    log,
    prompt_id_for,
    write_coverage,
    write_jsonl,
)

ELICITED_N = 2000
ULTRACHAT_N = 500
TOTAL_N = 2500


def load_raw() -> list[dict]:
    path = DATA_DIR / "raw_principles.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rebalance_elicited(raw: list[dict], target: int = ELICITED_N) -> list[dict]:
    per_book: list[tuple[str, list[str]]] = []
    for rec in raw:
        bid = rec.get("book_id") or ""
        prompts = [p.strip() for p in (rec.get("elicited_prompts") or []) if isinstance(p, str) and p.strip()]
        # unique within book
        seen, uniq = set(), []
        for p in prompts:
            k = " ".join(p.lower().split())
            if k in seen:
                continue
            seen.add(k)
            uniq.append(p)
        if uniq:
            per_book.append((bid, uniq))
    if not per_book:
        raise SystemExit("step2: no elicited prompts")
    # round-robin so longer books fill the remainder after short books exhaust
    out: list[dict] = []
    seen_global: set[str] = set()
    idx = [0] * len(per_book)
    active = True
    while len(out) < target and active:
        active = False
        for b, (bid, prompts) in enumerate(per_book):
            if len(out) >= target:
                break
            if idx[b] >= len(prompts):
                continue
            p = prompts[idx[b]]
            idx[b] += 1
            k = " ".join(p.lower().split())
            if k in seen_global:
                continue
            seen_global.add(k)
            out.append(
                {
                    "prompt": p,
                    "source": "elicited",
                    "book_id": bid,
                    "id": prompt_id_for(p),
                    "model": MODEL_ID,
                }
            )
            active = True
    # oversample from books that still have unused prompts, then wrap
    if len(out) < target:
        pool = []
        for bid, prompts in per_book:
            pool.extend((bid, p) for p in prompts)
        i = 0
        while len(out) < target and pool:
            bid, p = pool[i % len(pool)]
            # allow duplicates only after unique pool is exhausted, tagged
            rec = {
                "prompt": p,
                "source": "elicited",
                "book_id": bid,
                "id": prompt_id_for(p + f"#{len(out)}"),
                "model": MODEL_ID,
            }
            out.append(rec)
            i += 1
    return out[:target]


def fetch_ultrachat(n: int = ULTRACHAT_N) -> list[dict]:
    cache = DATA_DIR / "ultrachat_500.jsonl"
    if cache.exists():
        rows = []
        with cache.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if len(rows) >= n:
            return rows[:n]
    rows: list[dict] = []
    seen: set[str] = set()
    session = requests.Session()
    session.headers.update({"User-Agent": "CAI-pipeline/1.0"})
    offset = 0
    splits = ["train_sft", "train", "train_gen"]
    split = splits[0]
    while len(rows) < n and offset < 5000:
        url = "https://datasets-server.huggingface.co/rows"
        params = {
            "dataset": "HuggingFaceH4/ultrachat_200k",
            "config": "default",
            "split": split,
            "offset": offset,
            "length": 100,
        }
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 400 and split != splits[-1]:
                # try next split name
                si = splits.index(split)
                split = splits[min(si + 1, len(splits) - 1)]
                offset = 0
                continue
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            log(f"step2 ultrachat fetch offset={offset} split={split} err={e}")
            if split != splits[-1]:
                split = splits[splits.index(split) + 1]
                offset = 0
                continue
            break
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row") or item
            text = None
            if isinstance(row.get("prompt"), str) and row["prompt"].strip():
                text = row["prompt"].strip()
            elif isinstance(row.get("messages"), list) and row["messages"]:
                m0 = row["messages"][0]
                if isinstance(m0, dict):
                    text = (m0.get("content") or "").strip()
            elif isinstance(row.get("conversations"), list) and row["conversations"]:
                c0 = row["conversations"][0]
                if isinstance(c0, dict):
                    text = (c0.get("value") or c0.get("content") or "").strip()
            if not text or len(text) < 12:
                continue
            k = " ".join(text.lower().split())
            if k in seen:
                continue
            seen.add(k)
            rec = {
                "prompt": text,
                "source": "ultrachat",
                "id": prompt_id_for(text),
                "model": "ultrachat",
            }
            rows.append(rec)
            if len(rows) >= n:
                break
        offset += 100
    if len(rows) < n:
        raise SystemExit(f"step2: only got {len(rows)} UltraChat prompts (need {n})")
    write_jsonl(cache, rows)
    return rows[:n]


def main() -> None:
    state = State()
    raw = load_raw()
    elicited = rebalance_elicited(raw, ELICITED_N)
    ultrachat = fetch_ultrachat(ULTRACHAT_N)
    combined = elicited + ultrachat
    if len(combined) != TOTAL_N:
        raise SystemExit(f"step2: expected {TOTAL_N} prompts, got {len(combined)}")
    out_path = DATA_DIR / "prompts_dataset.jsonl"
    write_jsonl(out_path, combined)
    # seed sl_cai rows for resume (one transaction; avoid per-row fsync)
    state.executemany(
        """
        INSERT OR IGNORE INTO sl_cai(prompt_id, prompt, source, book_id, status)
        VALUES(?,?,?,?,?)
        """,
        [
            (rec["id"], rec["prompt"], rec.get("source"), rec.get("book_id"), "pending")
            for rec in combined
        ],
    )
    state.set_meta("prompts_total", str(len(combined)))
    state.set_meta("prompts_elicited", str(len(elicited)))
    state.set_meta("prompts_ultrachat", str(len(ultrachat)))
    log(f"step2 wrote {len(combined)} prompts elicited={len(elicited)} ultrachat={len(ultrachat)}")
    write_coverage(
        state,
        extra={
            "constitution_n": state.get_meta("constitution_n"),
            "prompts_total": len(combined),
            "prompts_elicited": len(elicited),
            "prompts_ultrachat": len(ultrachat),
            "notes": ["Step 2 assembled 2,000 elicited + 500 UltraChat = 2,500."],
        },
    )


if __name__ == "__main__":
    main()
