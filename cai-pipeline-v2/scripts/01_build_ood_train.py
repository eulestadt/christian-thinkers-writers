#!/usr/bin/env python3
"""Build ~1000 domain-balanced OOD-train prompts, disjoint from ood_eval and v1."""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (  # noqa: E402
    DOMAINS,
    EMBED_MODEL,
    EVAL_DIR,
    MIX,
    NEAR_DUP,
    REPO,
    V1_CONTROL_DATA,
    V1_DATA,
    V1_SCRIPTS,
    V2_ROOT,
    assert_no_leakage,
    load_jsonl,
    norm,
    write_jsonl,
)

os.environ.setdefault("CAI_STATE_DB", str(V2_ROOT / "data" / "eval" / "ood_train_gen_state.db"))
os.environ.setdefault("CAI_LOG_PATH", str(V2_ROOT / "data" / "eval" / "ood_train_gen.log"))
sys.path.insert(0, str(V1_SCRIPTS))
from llm_common import LLMClient, State, log  # noqa: E402

PER_DOMAIN = MIX["ood_train"] // len(DOMAINS)  # 200
GEN_PER_CALL = 40
CALLS_PER_DOMAIN = 8  # 320 candidates / domain before filter
EXCLUDE_COS = NEAR_DUP
INTERNAL_COS = 0.90

PROMPT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "domain": {"type": "string"},
                },
                "required": ["prompt"],
            },
        }
    },
    "required": ["prompts"],
}


def load_blocklist() -> list[str]:
    paths = [
        EVAL_DIR / "ood_contemporary_dilemmas.jsonl",
        EVAL_DIR / "open_ended_dilemmas.jsonl",
        V1_DATA / "prompts_dataset.jsonl",
        V1_CONTROL_DATA / "prompts_dataset.jsonl",
        V1_DATA / "sl_cai_train.jsonl",
        V1_CONTROL_DATA / "sl_cai_train.jsonl",
    ]
    out = []
    for p in paths:
        for row in load_jsonl(p):
            t = row.get("prompt") or ""
            if t.strip():
                out.append(t)
    return out


def gen_instruction(domain: str, n: int, avoid: list[str], batch_id: int) -> str:
    avoid_txt = "\n".join(f"- {a[:180]}" for a in avoid[:40])
    return f"""Write {n} distinct contemporary first-person human dilemmas in the domain "{domain}".
Each dilemma is a situation a person in the 2020s might actually face (workplace software, civic platforms, interpersonal apps, medical-adjacent portals, or online life).
Requirements:
- First person ("I ..."). One to three sentences, then a question about what to do.
- Using ONLY ordinary language, without importing any outside terminology or external frameworks.
- Do not quote or paraphrase published books, sermons, or novels.
- Do not repeat or lightly rewrite any of these already-used dilemmas:
{avoid_txt}
- Batch {batch_id}: invent new concrete facts (apps, jobs, institutions) not used above.
Return JSON with a `prompts` array of objects {{prompt, domain}}. domain must be "{domain}".
"""


def encode(texts: list[str], model):
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64))


def greedy_select(cands: list[dict], cand_emb, block_emb, n: int, rng_offset=0) -> list[dict]:
    """Max-min selection vs blocklist and already chosen."""
    chosen = []
    chosen_idx = []
    if cand_emb.shape[0] == 0:
        return []
    # distance to blocklist: 1 - max cosine
    if block_emb is not None and len(block_emb):
        max_sim_block = cand_emb @ block_emb.T
        block_max = max_sim_block.max(axis=1)
    else:
        block_max = np.zeros(len(cands))
    available = [i for i, c in enumerate(cands) if block_max[i] < EXCLUDE_COS]
    if not available:
        return []
    # start with lowest block similarity
    first = min(available, key=lambda i: float(block_max[i]))
    chosen_idx.append(first)
    while len(chosen_idx) < n and len(chosen_idx) < len(available):
        best_i, best_d = None, -1.0
        ch_emb = cand_emb[chosen_idx]
        for i in available:
            if i in chosen_idx:
                continue
            sim_ch = float((cand_emb[i] @ ch_emb.T).max())
            if sim_ch >= INTERNAL_COS:
                continue
            # maximize distance to chosen and to block
            d = min(1.0 - sim_ch, 1.0 - float(block_max[i]))
            if d > best_d:
                best_d, best_i = d, i
        if best_i is None:
            break
        chosen_idx.append(best_i)
    for i in chosen_idx:
        chosen.append(cands[i])
    return chosen


def main() -> None:
    V2_ROOT.joinpath("data/eval").mkdir(parents=True, exist_ok=True)
    block = load_blocklist()
    block_norm = {norm(t) for t in block}
    log(f"ood-train blocklist n={len(block)} unique_norm={len(block_norm)}")

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(EMBED_MODEL)
    log(f"encoding blocklist with {EMBED_MODEL}")
    block_emb = encode(block, enc) if block else None

    state = State()
    pool = LLMClient(state)
    avoid_eval = [r["prompt"] for r in load_jsonl(EVAL_DIR / "ood_contemporary_dilemmas.jsonl")]
    avoid_held = [r["prompt"] for r in load_jsonl(EVAL_DIR / "open_ended_dilemmas.jsonl")]
    avoid_sample = avoid_eval + avoid_held[:40]

    raw_path = V2_ROOT / "data" / "eval" / "ood_train_raw.jsonl"
    raw_rows: list[dict] = load_jsonl(raw_path)
    seen_raw = {norm(r["prompt"]) for r in raw_rows if r.get("prompt")}

    def one_call(domain: str, batch_id: int):
        inst = gen_instruction(domain, GEN_PER_CALL, avoid_sample, batch_id)
        assert_no_leakage(inst, f"ood-train gen instruction {domain}")
        data = pool.generate_json(
            inst,
            PROMPT_SCHEMA,
            step="ood_train_prompts",
            ref=f"{domain}-{batch_id}",
            max_output_tokens=8192,
            temperature=0.9,
        )
        items = data.get("prompts") or []
        out = []
        for it in items:
            p = (it.get("prompt") or "").strip()
            if not p:
                continue
            try:
                assert_no_leakage(p, "ood-train candidate")
            except RuntimeError:
                continue
            if norm(p) in block_norm or norm(p) in seen_raw:
                continue
            out.append({"prompt": p, "domain": domain, "batch": batch_id})
        return out

    jobs = [(d, b) for d in DOMAINS for b in range(CALLS_PER_DOMAIN)]
    # skip completed domain-batch if raw already has enough from that batch
    have_batch = {(r.get("domain"), r.get("batch")) for r in raw_rows}
    pending = [(d, b) for d, b in jobs if (d, b) not in have_batch]
    log(f"ood-train generate pending={len(pending)} already_raw={len(raw_rows)}")
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(one_call, d, b) for d, b in pending]
        for fut in as_completed(futs):
            got = fut.result()
            for rec in got:
                if norm(rec["prompt"]) in seen_raw:
                    continue
                seen_raw.add(norm(rec["prompt"]))
                raw_rows.append(rec)
            write_jsonl(raw_path, raw_rows)
            log(f"ood-train raw n={len(raw_rows)}")

    # greedy select per domain
    selected: list[dict] = []
    for domain in DOMAINS:
        cands = [r for r in raw_rows if r.get("domain") == domain]
        if not cands:
            log(f"WARN no candidates for {domain}")
            continue
        texts = [c["prompt"] for c in cands]
        log(f"encode {domain} candidates n={len(texts)}")
        emb = encode(texts, enc)
        picked = greedy_select(cands, emb, block_emb, PER_DOMAIN)
        if len(picked) < PER_DOMAIN:
            log(f"WARN {domain} only picked {len(picked)}/{PER_DOMAIN}")
        for i, rec in enumerate(picked[:PER_DOMAIN]):
            selected.append({"id": len(selected), "prompt": rec["prompt"], "domain": domain, "split": "train"})
        log(f"{domain} selected {min(len(picked), PER_DOMAIN)}")

    # final exact overlap check
    sel_norm = [norm(r["prompt"]) for r in selected]
    if len(set(sel_norm)) != len(sel_norm):
        raise SystemExit("internal exact dup after select")
    overlap = [n for n in sel_norm if n in block_norm]
    if overlap:
        raise SystemExit(f"exact overlap with blocklist n={len(overlap)}")

    out_store = V2_ROOT / "data" / "eval" / "ood_train_contemporary_dilemmas.jsonl"
    out_repo = EVAL_DIR / "ood_train_contemporary_dilemmas.jsonl"
    write_jsonl(out_store, selected)
    write_jsonl(out_repo, selected)
    by = {}
    for r in selected:
        by[r["domain"]] = by.get(r["domain"], 0) + 1
    log(f"wrote n={len(selected)} by_domain={by} store={out_store} repo={out_repo}")
    if len(selected) < MIX["ood_train"] * 0.9:
        raise SystemExit(f"too few OOD-train prompts: {len(selected)}")


if __name__ == "__main__":
    main()
