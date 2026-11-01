#!/usr/bin/env python3
"""SL-CAI on OOD-train prompts. Treatment: p ∝ sim * s_i^alpha. Control: p ∝ sim."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (  # noqa: E402
    ALPHA_DEFAULT,
    EMBED_MODEL,
    K_PRINCIPLES,
    V1_SCRIPTS,
    V2_ROOT,
    attach_s,
    example_weight,
    load_constitution,
    load_ier_by_writer,
    load_jsonl,
    sample_principles,
    slim_principle,
    write_jsonl,
)

ARM_ROOT = None  # set in main


def _configure_arm(arm: str) -> Path:
    data = V2_ROOT / "data" / arm
    data.mkdir(parents=True, exist_ok=True)
    os.environ["CAI_STATE_DB"] = str(data / "state.db")
    os.environ["CAI_LOG_PATH"] = str(data / "pipeline.log")
    return data


sys.path.insert(0, str(V1_SCRIPTS))
import llm_common as lc  # noqa: E402
from llm_common import (  # noqa: E402
    LLMClient,
    MODEL_ID,
    SL_BATCH_SCHEMA,
    SL_CAI_TEMPLATE,
    State,
    log,
    prompt_id_for,
)

BATCH = 5
RNG_SEED = 3407


def build_item_prompt(prompt: str, ps: list[dict]) -> str:
    s = [slim_principle(p) for p in ps]
    text = SL_CAI_TEMPLATE.format(
        prompt=prompt,
        p1_critique=s[0]["critique_request"],
        p1_revision=s[0]["revision_request"],
        p2_critique=s[1]["critique_request"],
        p2_revision=s[1]["revision_request"],
        p3_critique=s[2]["critique_request"],
        p3_revision=s[2]["revision_request"],
    )
    from v2_common import assert_no_leakage

    assert_no_leakage(text, "SL-CAI serving template fill")
    return text


def persist_ok(state: State, rec: dict, item: dict, ps: list[dict], mean_s: float, use_ier: bool) -> None:
    draft = (item.get("draft_0") or "").strip()
    r1 = (item.get("revision_1") or "").strip()
    r2 = (item.get("revision_2") or "").strip()
    r3 = (item.get("revision_3_final") or item.get("revision_3") or "").strip()
    if not draft or not r3:
        raise ValueError("missing draft_0 or revision_3_final")
    meta = example_weight(ps, mean_s)
    if not use_ier:
        meta["weight"] = 1.0
    slims = [slim_principle(p) for p in ps]
    state.execute(
        """
        UPDATE sl_cai SET draft_0=?, revision_1=?, revision_2=?, revision_3=?,
          p1_json=?, p2_json=?, p3_json=?, status='ok', error=NULL, model=?
        WHERE prompt_id=?
        """,
        (
            draft, r1, r2, r3,
            json.dumps(slims[0], ensure_ascii=False),
            json.dumps(slims[1], ensure_ascii=False),
            json.dumps(slims[2], ensure_ascii=False),
            MODEL_ID,
            rec["id"],
        ),
    )
    # trainer metadata only — not a prompt
    state.execute(
        "INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)",
        (f"weight:{rec['id']}", json.dumps(meta, ensure_ascii=False)),
    )


def parse_items(data: dict) -> list[dict]:
    items = []
    if isinstance(data, dict):
        items = data.get("items") or []
    elif isinstance(data, list):
        items = data
    return [it for it in items if isinstance(it, dict)]


def process_batch(pool, state, packed, mean_s, use_ier) -> None:
    body_parts = []
    for i, (rec, ps) in enumerate(packed, 1):
        body_parts.append(f"----- ITEM {i} of {len(packed)} -----\n{build_item_prompt(rec['prompt'], ps)}")
    instruction = (
        "For each ITEM below, independently perform the sequential Critique-and-Revision steps. "
        "Keep Draft_0 a genuine baseline answer (about 80–160 words). Keep critiques concise. "
        "Revision_3_Final is the full quality answer (about 120–220 words). "
        "Return JSON with an `items` array of the same length and order, each object containing "
        "prompt, draft_0, critique_1, revision_1, critique_2, revision_2, critique_3, revision_3_final.\n\n"
        + "\n\n".join(body_parts)
    )
    ref = "+".join(rec["id"] for rec, _ in packed)
    try:
        data = pool.generate_json(
            instruction, SL_BATCH_SCHEMA, step="v2_step3", ref=ref[:80],
            max_output_tokens=16384, temperature=0.7,
        )
        items = parse_items(data)
        if len(items) != len(packed):
            raise ValueError(f"batch_len {len(items)} != {len(packed)}")
        for (rec, ps), item in zip(packed, items):
            persist_ok(state, rec, item, ps, mean_s, use_ier)
    except Exception as e:
        log(f"v2 step3 batch fail ({len(packed)}): {e}; singles")
        for rec, ps in packed:
            try:
                one = (
                    "Perform the sequential Critique-and-Revision steps. "
                    "Keep Draft_0 a genuine baseline (80–160 words) and Revision_3_Final the full answer (120–220 words). "
                    "Return JSON with items: a one-element array containing prompt, draft_0, critique_1, revision_1, "
                    "critique_2, revision_2, critique_3, revision_3_final.\n\n"
                    + build_item_prompt(rec["prompt"], ps)
                )
                data = pool.generate_json(
                    one, SL_BATCH_SCHEMA, step="v2_step3", ref=rec["id"],
                    max_output_tokens=8192, temperature=0.7,
                )
                items = parse_items(data)
                if not items:
                    raise ValueError("no items")
                persist_ok(state, rec, items[0], ps, mean_s, use_ier)
            except Exception as e2:
                state.execute(
                    "UPDATE sl_cai SET status='error', error=? WHERE prompt_id=?",
                    (str(e2)[:800], rec["id"]),
                )
                state.log_error("v2_step3", MODEL_ID, rec["id"], type(e2).__name__, str(e2))


def seed_prompts(state: State, rows: list[dict]) -> None:
    for r in rows:
        pid = r.get("id")
        if isinstance(pid, int):
            pid = f"ood-{pid:04d}"
        elif not pid:
            pid = prompt_id_for(r["prompt"])
        r["id"] = str(pid)
        state.execute(
            """
            INSERT OR IGNORE INTO sl_cai(prompt_id, prompt, source, book_id, status)
            VALUES(?,?,?,?,?)
            """,
            (r["id"], r["prompt"], "ood", None, "pending"),
        )


def export_train(state: State, data_dir: Path) -> int:
    rows = state.execute(
        """
        SELECT prompt_id, prompt, revision_3 FROM sl_cai
        WHERE status='ok' AND revision_3 IS NOT NULL AND revision_3 != ''
          AND COALESCE(model,'')=?
        """,
        (MODEL_ID,),
    ).fetchall()
    out = []
    for pid, prompt, y in rows:
        wrow = state.execute("SELECT v FROM meta WHERE k=?", (f"weight:{pid}",)).fetchone()
        meta = json.loads(wrow[0]) if wrow else {"weight": 1.0}
        out.append({
            "prompt": prompt,
            "completion": y,
            "model": MODEL_ID,
            "source": "ood",
            "prompt_id": pid,
            "weight": meta.get("weight", 1.0),
            "principle_ids": meta.get("principle_ids"),
            "book_ids": meta.get("book_ids"),
        })
    write_jsonl(data_dir / "sl_cai_ood.jsonl", out)
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["treatment", "control"], required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    args = ap.parse_args()
    data_dir = _configure_arm(args.arm)
    lc.LOG_PATH = data_dir / "pipeline.log"
    lc.STATE_DB = data_dir / "state.db"
    lc.DATA_DIR = data_dir
    use_ier = args.arm == "treatment"
    ood_path = V2_ROOT / "data" / "eval" / "ood_train_contemporary_dilemmas.jsonl"
    if not ood_path.is_file():
        ood_path = Path("/workspace/colab/data/eval/ood_train_contemporary_dilemmas.jsonl")
    rows = load_jsonl(ood_path)
    if not rows:
        raise SystemExit(f"no OOD-train prompts at {ood_path}")

    state = State(path=data_dir / "state.db")
    seed_prompts(state, rows)
    principles = load_constitution(args.arm)
    ier = load_ier_by_writer() if use_ier else {}
    mean_s = attach_s(principles, ier) if use_ier else 1.0
    log(f"v2 sl-cai arm={args.arm} n_prompts={len(rows)} n_principles={len(principles)} use_ier={use_ier} alpha={args.alpha} mean_s={mean_s:.6f}")

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(EMBED_MODEL)
    texts = [p.get("source_thesis") or p.get("critique_request") or "" for p in principles]
    emb = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    emb = np.asarray(emb)

    pending = state.execute(
        "SELECT prompt_id, prompt FROM sl_cai WHERE status IS NULL OR status='pending' OR status='error' OR COALESCE(model,'') != ?",
        (MODEL_ID,),
    ).fetchall()
    recs = [{"id": pid, "prompt": prompt} for pid, prompt in pending]
    log(f"v2 sl-cai pending={len(recs)}")
    rng = random.Random(RNG_SEED + (0 if use_ier else 99))
    if recs:
        qemb = enc.encode([r["prompt"] for r in recs], normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        packed_all = [
            (rec, sample_principles(rec["prompt"], principles, emb, qemb[i], rng, k=K_PRINCIPLES, alpha=args.alpha, use_ier=use_ier))
            for i, rec in enumerate(recs)
        ]
    else:
        packed_all = []
    batches = [packed_all[i : i + BATCH] for i in range(0, len(packed_all), BATCH)]
    pool = LLMClient(state)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_batch, pool, state, b, mean_s, use_ier) for b in batches]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 5 == 0 or done == len(batches):
                n_ok = state.execute("SELECT COUNT(*) FROM sl_cai WHERE status='ok'").fetchone()[0]
                n_exp = export_train(state, data_dir)
                log(f"v2 sl-cai {args.arm} batches {done}/{len(batches)} sl_ok={n_ok} exported={n_exp}")
    n = export_train(state, data_dir)
    log(f"v2 sl-cai {args.arm} done n={n}")


if __name__ == "__main__":
    main()
