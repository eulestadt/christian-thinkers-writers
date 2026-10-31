#!/usr/bin/env python3
"""Step 3: SL-CAI 3-step critique/revision. Resume from state.db."""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_common import (  # noqa: E402
    DATA_DIR,
    LLMClient,
    MODEL_ID,
    SL_BATCH_SCHEMA,
    SL_CAI_TEMPLATE,
    State,
    log,
    write_coverage,
    write_jsonl,
)

BATCH = 5
RNG = random.Random(42)


def load_constitution() -> list[dict]:
    data = json.loads((DATA_DIR / "constitution.json").read_text(encoding="utf-8"))
    if data.get("model") != MODEL_ID:
        raise SystemExit(f"step3: constitution.json model is {data.get('model')!r}, expected {MODEL_ID}")
    pr = data.get("principles") or []
    if not isinstance(pr, list) or not pr:
        raise SystemExit("step3: empty constitution")
    return pr


def embed_principles(principles: list[dict]):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [p.get("source_thesis") or p.get("critique_request") or "" for p in principles]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return model, np.asarray(emb)


def sample_principles(model, emb, principles: list[dict], prompt: str) -> list[dict]:
    n = len(principles)
    if n == 0:
        raise RuntimeError("no principles")
    if n <= 3:
        return [principles[i % n] for i in range(3)]
    q = model.encode([prompt], normalize_embeddings=True, show_progress_bar=False)[0]
    return sample_from_query(emb, principles, q)


def sample_from_query(emb, principles: list[dict], q) -> list[dict]:
    n = len(principles)
    if n <= 3:
        return [principles[i % n] for i in range(3)]
    sims = emb @ q
    top = np.argsort(-sims)[:8]
    # take top-1 plus two more from the top-8 without replacement
    chosen_idx = [int(top[0])]
    rest = [int(i) for i in top[1:]]
    RNG.shuffle(rest)
    for i in rest:
        if i not in chosen_idx:
            chosen_idx.append(i)
        if len(chosen_idx) == 3:
            break
    while len(chosen_idx) < 3:
        j = int(RNG.randrange(n))
        if j not in chosen_idx:
            chosen_idx.append(j)
    return [principles[i] for i in chosen_idx]


def slim(p: dict) -> dict:
    return {
        "source_thesis": p.get("source_thesis") or "",
        "critique_request": p.get("critique_request") or "",
        "revision_request": p.get("revision_request") or "",
    }


def build_item_prompt(prompt: str, ps: list[dict]) -> str:
    return SL_CAI_TEMPLATE.format(
        prompt=prompt,
        p1_critique=ps[0]["critique_request"],
        p1_revision=ps[0]["revision_request"],
        p2_critique=ps[1]["critique_request"],
        p2_revision=ps[1]["revision_request"],
        p3_critique=ps[2]["critique_request"],
        p3_revision=ps[2]["revision_request"],
    )


def parse_items(data: dict, n_expected: int) -> list[dict]:
    items = []
    if isinstance(data, dict):
        items = data.get("items") or []
    elif isinstance(data, list):
        items = data
    if not isinstance(items, list):
        items = []
    return [it for it in items if isinstance(it, dict)]


def persist_ok(state: State, pid: str, rec: dict, item: dict, ps: list[dict]) -> None:
    draft = (item.get("draft_0") or "").strip()
    r1 = (item.get("revision_1") or "").strip()
    r2 = (item.get("revision_2") or "").strip()
    r3 = (item.get("revision_3_final") or item.get("revision_3") or "").strip()
    if not draft or not r3:
        raise ValueError("missing draft_0 or revision_3_final")
    state.execute(
        """
        UPDATE sl_cai SET draft_0=?, revision_1=?, revision_2=?, revision_3=?,
          p1_json=?, p2_json=?, p3_json=?, status='ok', error=NULL, model=?
        WHERE prompt_id=?
        """,
        (
            draft,
            r1,
            r2,
            r3,
            json.dumps(slim(ps[0]), ensure_ascii=False),
            json.dumps(slim(ps[1]), ensure_ascii=False),
            json.dumps(slim(ps[2]), ensure_ascii=False),
            MODEL_ID,
            pid,
        ),
    )


def process_batch(pool: LLMClient, state: State, packed: list) -> None:
    # packed: list of (rec, ps) with principles already sampled on the main thread
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
    pids = ",".join(r["prompt"][:12] for r, _ in packed)
    ref = packed[0][0]["prompt_id"] if "prompt_id" in packed[0][0] else packed[0][0].get("id", "")
    # prompt_id is on the sqlite row; we pass it as id
    ref = "+".join(rec.get("id") or rec.get("prompt_id") for rec, _ in packed)
    try:
        data = pool.generate_json(
            instruction,
            SL_BATCH_SCHEMA,
            step="step3",
            ref=ref[:80],
            max_output_tokens=16384,
            temperature=0.7,
        )
        items = parse_items(data, len(packed))
        if len(items) != len(packed):
            raise ValueError(f"batch_len {len(items)} != {len(packed)}")
        for (rec, ps), item in zip(packed, items):
            persist_ok(state, rec["id"], rec, item, ps)
    except Exception as e:
        log(f"step3 batch fail ({len(packed)}): {e}; falling back to singles")
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
                    one,
                    SL_BATCH_SCHEMA,
                    step="step3",
                    ref=rec["id"],
                    max_output_tokens=8192,
                    temperature=0.7,
                )
                items = parse_items(data, 1)
                if not items:
                    raise ValueError("no items")
                persist_ok(state, rec["id"], rec, items[0], ps)
            except Exception as e2:
                state.execute(
                    "UPDATE sl_cai SET status='error', error=? WHERE prompt_id=?",
                    (str(e2)[:800], rec["id"]),
                )
                state.log_error("step3", MODEL_ID, rec["id"], type(e2).__name__, str(e2))


def export_train(state: State) -> int:
    rows = state.execute(
        """
        SELECT prompt, revision_3 FROM sl_cai
        WHERE status='ok' AND revision_3 IS NOT NULL AND revision_3 != ''
          AND COALESCE(model,'')=?
        """,
        (MODEL_ID,),
    ).fetchall()
    write_jsonl(
        DATA_DIR / "sl_cai_train.jsonl",
        [{"prompt": p, "completion": y, "model": MODEL_ID} for p, y in rows],
    )
    return len(rows)


def main() -> None:
    state = State()
    pool = LLMClient(state)
    workers = 8
    principles = load_constitution()
    log(f"step3 constitution n={len(principles)}")
    model, emb = embed_principles(principles)
    pending = state.execute(
        "SELECT prompt_id, prompt, source, book_id FROM sl_cai WHERE status IS NULL OR status='pending' OR status='error' OR COALESCE(model,'') != ?",
        (MODEL_ID,),
    ).fetchall()
    recs = [
        {"id": pid, "prompt": prompt, "source": src, "book_id": bid}
        for pid, prompt, src, bid in pending
    ]
    log(f"step3 pending={len(recs)}; sampling principles on main thread")
    if recs:
        qemb = model.encode(
            [rec["prompt"] for rec in recs],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        packed_all = [
            (rec, sample_from_query(emb, principles, qemb[i])) for i, rec in enumerate(recs)
        ]
    else:
        packed_all = []
    batches = [packed_all[i : i + BATCH] for i in range(0, len(packed_all), BATCH)]
    # modest concurrency: RPM limiter is the real gate
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_batch, pool, state, b) for b in batches]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 10 == 0 or done == len(batches):
                n_ok = state.execute("SELECT COUNT(*) FROM sl_cai WHERE status='ok'").fetchone()[0]
                log(f"step3 batches {done}/{len(batches)} sl_ok={n_ok}")
                n_ok_export = export_train(state)
                write_coverage(
                    state,
                    extra={
                        "constitution_n": state.get_meta("constitution_n"),
                        "prompts_total": state.get_meta("prompts_total"),
                        "prompts_elicited": state.get_meta("prompts_elicited"),
                        "prompts_ultrachat": state.get_meta("prompts_ultrachat"),
                        "notes": [f"Step 3 in progress: {n_ok_export} SL-CAI rows exported."],
                    },
                )
    n = export_train(state)
    log(f"step3 done sl_cai_train={n}")
    write_coverage(
        state,
        extra={
            "constitution_n": state.get_meta("constitution_n"),
            "prompts_total": state.get_meta("prompts_total"),
            "prompts_elicited": state.get_meta("prompts_elicited"),
            "prompts_ultrachat": state.get_meta("prompts_ultrachat"),
            "notes": [f"Step 3 wrote {n} SL-CAI SFT pairs."],
        },
    )


if __name__ == "__main__":
    main()
