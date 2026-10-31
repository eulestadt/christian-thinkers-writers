#!/usr/bin/env python3
"""Step 4: RLAIF CoT preference labeling. Resume from state.db + store sidecar."""
from __future__ import annotations

import json
import random
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_common import (  # noqa: E402
    DATA_DIR,
    DPO_JSONL,
    DPO_SIDECAR,
    LLMClient,
    MODEL_ID,
    RLAIF_SCHEMA,
    RLAIF_TEMPLATE,
    STATE_DB,
    State,
    append_jsonl_line,
    log,
    try_sync_state_db,
    utcnow,
    write_coverage,
    write_jsonl,
)

RNG = random.Random(7)
BATCH_EXPORT = 10


class DpoSink:
    """Durable DPO sink: store jsonl + sidecar always win if sqlite I/O fails."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.by_id: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if DPO_SIDECAR.exists():
            with DPO_SIDECAR.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    pid = rec.get("prompt_id")
                    if pid and rec.get("model") == MODEL_ID:
                        self.by_id[pid] = rec
        if DPO_JSONL.exists() and not self.by_id:
            # jsonl has no prompt_id; keep rows to rewrite later after sqlite join
            pass
        log(f"dpo sink loaded sidecar={len(self.by_id)} path={DPO_SIDECAR}")

    def known_ids(self) -> set[str]:
        with self.lock:
            return set(self.by_id)

    def ingest(self, prompt_id: str, prompt: str, chosen: str, rejected: str) -> None:
        rec = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "model": MODEL_ID,
            "ts": utcnow(),
        }
        with self.lock:
            if prompt_id in self.by_id:
                return
            self.by_id[prompt_id] = rec
            try:
                append_jsonl_line(DPO_SIDECAR, rec)
            except Exception as e:
                log(f"sidecar append failed: {type(e).__name__}: {e}")

    def record(self, prompt_id: str, prompt: str, chosen: str, rejected: str) -> int:
        rec = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "model": MODEL_ID,
            "ts": utcnow(),
        }
        train = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "model": MODEL_ID,
        }
        with self.lock:
            is_new = prompt_id not in self.by_id
            self.by_id[prompt_id] = rec
            n = len(self.by_id)
            if is_new:
                try:
                    append_jsonl_line(DPO_SIDECAR, rec)
                except Exception as e:
                    log(f"sidecar append failed: {type(e).__name__}: {e}")
                try:
                    append_jsonl_line(DPO_JSONL, train)
                except Exception as e:
                    log(f"dpo jsonl append failed: {type(e).__name__}: {e}")
                    # last resort: rewrite from memory onto the store path
                    try:
                        write_jsonl(DPO_JSONL, [self._train(v) for v in self.by_id.values()])
                    except Exception as e2:
                        log(f"dpo jsonl rewrite failed: {type(e2).__name__}: {e2}")
            if n % BATCH_EXPORT == 0 or n >= 2500:
                self._flush_locked()
        return n

    def _train(self, rec: dict) -> dict:
        return {
            "prompt": rec["prompt"],
            "chosen": rec["chosen"],
            "rejected": rec["rejected"],
            "model": MODEL_ID,
        }

    def _flush_locked(self) -> None:
        rows = [self._train(v) for v in self.by_id.values()]
        try:
            write_jsonl(DPO_JSONL, rows)
        except Exception as e:
            log(f"dpo jsonl flush failed: {type(e).__name__}: {e}")
        try_sync_state_db()

    def flush(self) -> int:
        with self.lock:
            self._flush_locked()
            return len(self.by_id)

    def n(self) -> int:
        with self.lock:
            return len(self.by_id)


SINK = DpoSink()


def sqlite_ok(state: State, sql: str, params: tuple = ()) -> bool:
    try:
        state.execute(sql, params)
        return True
    except sqlite3.Error as e:
        log(f"sqlite persist failed: {type(e).__name__}: {e}")
        return False


def judge_one(pool: LLMClient, state: State, row: dict) -> None:
    pid = row["prompt_id"]
    if pid in SINK.known_ids():
        return
    try:
        existing = state.execute("SELECT status FROM rlaif WHERE prompt_id=?", (pid,)).fetchone()
        if existing and existing[0] == "ok":
            return
    except sqlite3.Error:
        existing = None
    y0 = row["draft_0"]
    y3 = row["revision_3"]
    if RNG.random() < 0.5:
        ya, yb, order = y0, y3, "draft,final"
        map_choice = {"A": ("draft", y0), "B": ("final", y3)}
    else:
        ya, yb, order = y3, y0, "final,draft"
        map_choice = {"A": ("final", y3), "B": ("draft", y0)}
    p1 = json.loads(row["p1_json"] or "{}")
    p2 = json.loads(row["p2_json"] or "{}")
    p3 = json.loads(row["p3_json"] or "{}")
    prompt = RLAIF_TEMPLATE.format(
        X=row["prompt"],
        Y_A=ya,
        Y_B=yb,
        P1=p1.get("source_thesis") or p1.get("critique_request") or "",
        P2=p2.get("source_thesis") or p2.get("critique_request") or "",
        P3=p3.get("source_thesis") or p3.get("critique_request") or "",
    )
    prompt += (
        "\n\nReturn JSON with keys thought (the chain-of-thought that would go inside <thought> tags) "
        "and choice (exactly \"A\" or \"B\")."
    )
    try:
        data = pool.generate_json(
            prompt,
            RLAIF_SCHEMA,
            step="step4",
            ref=pid,
            max_output_tokens=2048,
            temperature=0.3,
        )
        choice = str((data or {}).get("choice") or "").strip().upper()
        if choice not in ("A", "B"):
            th = str((data or {}).get("thought") or "")
            if "choice is a" in th.lower() or th.strip().endswith("A"):
                choice = "A"
            elif "choice is b" in th.lower() or th.strip().endswith("B"):
                choice = "B"
            else:
                raise ValueError(f"bad choice {choice!r}")
        tag, winner = map_choice[choice]
        chosen_text = winner
        rejected_text = y0 if chosen_text == y3 else y3
        if chosen_text == rejected_text:
            rejected_text = yb if chosen_text == ya else ya
        sqlite_ok(
            state,
            """
            INSERT INTO rlaif(prompt_id, ya, yb, order_ab, thought, choice, chosen, rejected, status, error, model)
            VALUES(?,?,?,?,?,?,?,?, 'ok', NULL, ?)
            ON CONFLICT(prompt_id) DO UPDATE SET
              ya=excluded.ya, yb=excluded.yb, order_ab=excluded.order_ab,
              thought=excluded.thought, choice=excluded.choice,
              chosen=excluded.chosen, rejected=excluded.rejected,
              status='ok', error=NULL, model=excluded.model
            """,
            (
                pid,
                ya,
                yb,
                order,
                (data or {}).get("thought") or "",
                choice,
                chosen_text,
                rejected_text,
                MODEL_ID,
            ),
        )
        n = SINK.record(pid, row["prompt"], chosen_text, rejected_text)
        if n % BATCH_EXPORT == 0:
            log(f"step4 store jsonl={n} sidecar={DPO_SIDECAR.name} state={STATE_DB}")
    except Exception as e:
        sqlite_ok(
            state,
            """
            INSERT INTO rlaif(prompt_id, status, error) VALUES(?,?,?)
            ON CONFLICT(prompt_id) DO UPDATE SET status='error', error=excluded.error
            """,
            (pid, "error", str(e)[:800]),
        )
        state.log_error("step4", MODEL_ID, pid, type(e).__name__, str(e))


def seed_sink_from_sqlite(state: State) -> None:
    """Backfill sidecar/jsonl from sqlite so resume skips completed ids."""
    try:
        rows = state.execute(
            """
            SELECT s.prompt_id, s.prompt, r.chosen, r.rejected
            FROM rlaif r JOIN sl_cai s ON s.prompt_id = r.prompt_id
            WHERE r.status='ok' AND r.chosen IS NOT NULL AND r.rejected IS NOT NULL
              AND r.chosen != '' AND r.rejected != ''
              AND COALESCE(r.model,'')=? AND COALESCE(s.model,'')=?
            """,
            (MODEL_ID, MODEL_ID),
        ).fetchall()
    except sqlite3.Error as e:
        log(f"seed from sqlite failed: {e}")
        return
    for pid, prompt, chosen, rejected in rows:
        SINK.ingest(pid, prompt, chosen, rejected)
    n = SINK.flush()
    log(f"seeded store dpo jsonl n={n}")


def main() -> None:
    state = State()
    seed_sink_from_sqlite(state)
    already = SINK.known_ids()
    pool = LLMClient(state)
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
    recs = [
        {
            "prompt_id": pid,
            "prompt": prompt,
            "draft_0": d0,
            "revision_3": r3,
            "p1_json": p1,
            "p2_json": p2,
            "p3_json": p3,
        }
        for pid, prompt, d0, r3, p1, p2, p3 in pending
        if pid not in already
    ]
    log(f"step4 pending={len(recs)} already_store={len(already)}")
    workers = 8
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(judge_one, pool, state, rec) for rec in recs]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % BATCH_EXPORT == 0 or done == len(recs):
                n = SINK.n()
                log(f"step4 {done}/{len(recs)} rlaif_ok_store={n}")
                write_coverage(
                    state,
                    extra={
                        "constitution_n": state.get_meta("constitution_n"),
                        "prompts_total": state.get_meta("prompts_total"),
                        "prompts_elicited": state.get_meta("prompts_elicited"),
                        "prompts_ultrachat": state.get_meta("prompts_ultrachat"),
                        "notes": [f"Step 4 in progress: {n} DPO pairs on store jsonl."],
                    },
                )
    n = SINK.flush()
    log(f"step4 done rlaif_dpo_train={n} path={DPO_JSONL}")
    write_coverage(
        state,
        extra={
            "constitution_n": state.get_meta("constitution_n"),
            "prompts_total": state.get_meta("prompts_total"),
            "prompts_elicited": state.get_meta("prompts_elicited"),
            "prompts_ultrachat": state.get_meta("prompts_ultrachat"),
            "notes": [
                f"Step 4 wrote {n} RLAIF DPO pairs to {DPO_JSONL}.",
                "DeepSeek-only Steps 0–4 complete. Step 5 (Colab/GPU/Unsloth/Drive) was not run.",
            ],
        },
    )


if __name__ == "__main__":
    main()
