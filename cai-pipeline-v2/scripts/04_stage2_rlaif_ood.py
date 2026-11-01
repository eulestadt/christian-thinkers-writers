#!/usr/bin/env python3
"""RLAIF on OOD-train SL-CAI rows. Judge sees source_thesis only (no scores)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import V1_SCRIPTS, V2_ROOT, load_jsonl, write_jsonl  # noqa: E402

sys.path.insert(0, str(V1_SCRIPTS))
import llm_common as lc  # noqa: E402
from llm_common import LLMClient, MODEL_ID, RLAIF_SCHEMA, RLAIF_TEMPLATE, State, log  # noqa: E402

RNG = random.Random(7)


SIDECAR_LOCK = threading.Lock()


def judge_one(pool: LLMClient, state: State, row: dict, sink: list, sink_path: Path) -> None:
    pid = row["prompt_id"]
    existing = state.execute("SELECT status FROM rlaif WHERE prompt_id=?", (pid,)).fetchone()
    if existing and existing[0] == "ok":
        return
    y0 = row["draft_0"]
    y3 = row["revision_3"]
    if RNG.random() < 0.5:
        ya, yb, order = y0, y3, "draft,final"
        map_choice = {"A": y0, "B": y3}
    else:
        ya, yb, order = y3, y0, "final,draft"
        map_choice = {"A": y3, "B": y0}
    p1 = json.loads(row["p1_json"] or "{}")
    p2 = json.loads(row["p2_json"] or "{}")
    p3 = json.loads(row["p3_json"] or "{}")
    prompt = RLAIF_TEMPLATE.format(
        X=row["prompt"],
        Y_A=ya,
        Y_B=yb,
        P1=p1.get("source_thesis") or "",
        P2=p2.get("source_thesis") or "",
        P3=p3.get("source_thesis") or "",
    )
    prompt += (
        "\n\nReturn JSON with keys thought (the chain-of-thought that would go inside <thought> tags) "
        "and choice (exactly \"A\" or \"B\")."
    )
    from v2_common import assert_no_leakage

    assert_no_leakage(prompt, f"rlaif {pid}")
    try:
        data = pool.generate_json(
            prompt, RLAIF_SCHEMA, step="v2_step4", ref=pid,
            max_output_tokens=2048, temperature=0.3,
        )
        choice = str((data or {}).get("choice") or "").strip().upper()
        if choice not in ("A", "B"):
            raise ValueError(f"bad choice {choice!r}")
        chosen_text = map_choice[choice]
        rejected_text = y0 if chosen_text == y3 else y3
        state.execute(
            """
            INSERT INTO rlaif(prompt_id, ya, yb, order_ab, thought, choice, chosen, rejected, status, error, model)
            VALUES(?,?,?,?,?,?,?,?, 'ok', NULL, ?)
            ON CONFLICT(prompt_id) DO UPDATE SET
              ya=excluded.ya, yb=excluded.yb, order_ab=excluded.order_ab,
              thought=excluded.thought, choice=excluded.choice,
              chosen=excluded.chosen, rejected=excluded.rejected,
              status='ok', error=NULL, model=excluded.model
            """,
            (pid, ya, yb, order, (data or {}).get("thought") or "", choice, chosen_text, rejected_text, MODEL_ID),
        )
        rec = {
            "prompt_id": pid,
            "prompt": row["prompt"],
            "chosen": chosen_text,
            "rejected": rejected_text,
            "model": MODEL_ID,
            "source": "ood",
            "weight": row.get("weight", 1.0),
            "principle_ids": row.get("principle_ids"),
            "book_ids": row.get("book_ids"),
        }
        with SIDECAR_LOCK:
            sink.append(rec)
            with sink_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        state.execute(
            """
            INSERT INTO rlaif(prompt_id, status, error) VALUES(?,?,?)
            ON CONFLICT(prompt_id) DO UPDATE SET status='error', error=excluded.error
            """,
            (pid, "error", str(e)[:800]),
        )
        state.log_error("v2_step4", MODEL_ID, pid, type(e).__name__, str(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["treatment", "control"], required=True)
    args = ap.parse_args()
    data_dir = V2_ROOT / "data" / args.arm
    os.environ["CAI_STATE_DB"] = str(data_dir / "state.db")
    os.environ["CAI_LOG_PATH"] = str(data_dir / "pipeline.log")
    lc.LOG_PATH = data_dir / "pipeline.log"
    lc.STATE_DB = data_dir / "state.db"
    lc.DATA_DIR = data_dir
    state = State(path=data_dir / "state.db")
    pool = LLMClient(state)
    sl = load_jsonl(data_dir / "sl_cai_ood.jsonl")
    weight_by_id = {r["prompt_id"]: r for r in sl}
    pending = state.execute(
        """
        SELECT s.prompt_id, s.prompt, s.draft_0, s.revision_3, s.p1_json, s.p2_json, s.p3_json
        FROM sl_cai s
        LEFT JOIN rlaif r ON r.prompt_id = s.prompt_id
        WHERE s.status='ok' AND COALESCE(s.model,'')=?
          AND (r.status IS NULL OR r.status!='ok' OR COALESCE(r.model,'') != ?)
        """,
        (MODEL_ID, MODEL_ID),
    ).fetchall()
    recs = []
    for pid, prompt, d0, r3, p1, p2, p3 in pending:
        meta = weight_by_id.get(pid) or {}
        recs.append({
            "prompt_id": pid, "prompt": prompt, "draft_0": d0, "revision_3": r3,
            "p1_json": p1, "p2_json": p2, "p3_json": p3,
            "weight": meta.get("weight", 1.0),
            "principle_ids": meta.get("principle_ids"),
            "book_ids": meta.get("book_ids"),
        })
    log(f"v2 rlaif {args.arm} pending={len(recs)}")
    sidecar = data_dir / "rlaif_ood_checkpoint.jsonl"
    sink: list[dict] = load_jsonl(sidecar)
    known = {r.get("prompt_id") for r in sink}
    recs = [r for r in recs if r["prompt_id"] not in known]
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(judge_one, pool, state, r, sink, sidecar) for r in recs]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 20 == 0 or done == len(futs):
                log(f"v2 rlaif {args.arm} {done}/{len(futs)}")
    # rewrite clean train jsonl from sqlite + weights
    rows = state.execute(
        """
        SELECT s.prompt_id, s.prompt, r.chosen, r.rejected
        FROM rlaif r JOIN sl_cai s ON s.prompt_id = r.prompt_id
        WHERE r.status='ok' AND r.chosen IS NOT NULL AND r.rejected IS NOT NULL
        """
    ).fetchall()
    out = []
    for pid, prompt, chosen, rejected in rows:
        meta = weight_by_id.get(pid) or {}
        out.append({
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
            "model": MODEL_ID, "source": "ood", "prompt_id": pid,
            "weight": meta.get("weight", 1.0),
            "principle_ids": meta.get("principle_ids"),
            "book_ids": meta.get("book_ids"),
        })
    write_jsonl(data_dir / "rlaif_ood.jsonl", out)
    log(f"v2 rlaif {args.arm} wrote {len(out)}")


if __name__ == "__main__":
    main()
