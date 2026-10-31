#!/usr/bin/env python3
"""Shared DeepSeek client, rate limits, SQLite checkpoints, path helpers.

One model: deepseek-flash (DeepSeek-V4.1-Flash-1017). Thinking off on every call.
Never log or write API keys. Instruction wrappers must not contain external
alignment-taxonomy terms (those may appear only inside source-book text).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

ROOT = Path("/cursor/stores/self/cai-pipeline")
BOOKS_DIR = ROOT / "books_raw"
DATA_DIR = ROOT / "data"
SCRIPTS_DIR = ROOT / "scripts"
SCORES_DB = Path("/cursor/stores/self/docs/intrinsic-motivation-scores.sqlite")
COVERAGE_MD = Path("/cursor/stores/self/docs/cai-deepseek-coverage.md")
COVERAGE_MD_LEGACY = Path("/cursor/stores/self/docs/cai-gemini-coverage.md")
STATE_DB = Path(os.environ.get("CAI_STATE_DB") or (DATA_DIR / "state.db"))
LOG_PATH = Path(os.environ.get("CAI_LOG_PATH") or (DATA_DIR / "pipeline.log"))

MODEL_ID = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
RPM = 40
TPM_CAP = 2_500_000
MAX_IN_FLIGHT = 6
# DeepSeek V4.1 Flash context is large; keep chunks under ~100k tokens.
CHUNK_MAX_CHARS = 400_000
PEAK_STOP_BUFFER_S = 20

# Peak: 01:00–04:00 and 06:00–10:00 UTC Monday–Friday excluding Chinese public holidays.
# Mainland China official holidays 2025–2026 (State Council rest days).
CHINESE_PUBLIC_HOLIDAYS = {
    "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",
    "2025-04-04", "2025-04-05", "2025-04-06",
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
    "2025-05-31", "2025-06-01", "2025-06-02",
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05",
    "2025-10-06", "2025-10-07", "2025-10-08",
    "2026-01-01", "2026-01-02", "2026-01-03",
    "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19",
    "2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23",
    "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    "2026-06-19", "2026-06-20", "2026-06-21",
    "2026-09-25", "2026-09-26", "2026-09-27",
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05",
    "2026-10-06", "2026-10-07",
}

LEAKAGE_TERMS = (
    "vanderweele",
    "vander weele",
    "flourishing",
    "fai benchmark",
    "fai taxonomy",
    "human flourishing",
    "global flourishing",
)

EXTRACT_PROMPT = """Read the provided text in its entirety. Using ONLY the arguments, worldview, and normative claims explicitly present in this text (without importing any outside terminology or external frameworks), extract:
1. The core normative principles argued by the author, formatted as Constitutional AI pairs:
   - "critique_request": An instruction asking a reviewer to identify how a response fails to uphold or contradicts this specific argument from the text.
   - "revision_request": An instruction asking a writer to rewrite the response so it faithfully embodies this specific argument and reasoning from the text.
2. 50 realistic, first-person human questions or dilemmas that the arguments in this text directly address. If the text cannot support 50 distinct dilemmas, produce as many as the text actually supports; do not invent dilemmas the author does not address, and do not import topics from other books or from any external framework.
"""

SL_CAI_TEMPLATE = """Prompt (X): {prompt}

Perform the following sequential Critique-and-Revision steps:
- Step 0 [Draft_0]: Write an initial baseline response to Prompt (X).
- Step 1 [Critique_1]: {p1_critique}
- Step 2 [Revision_1]: {p1_revision} (Rewrite Draft_0 based on Critique_1).
- Step 3 [Critique_2]: {p2_critique}
- Step 4 [Revision_2]: {p2_revision} (Rewrite Revision_1 based on Critique_2, retaining prior improvements).
- Step 5 [Critique_3]: {p3_critique}
- Step 6 [Revision_3_Final]: {p3_revision} (Write the final, cohesive response integrating all revisions).
"""

RLAIF_TEMPLATE = """Human: Consider the following conversation between a human and an assistant:
Human Prompt: {X}

Options:
(A) {Y_A}
(B) {Y_B}

Evaluate which option better adheres to the following principles:
1. {P1}
2. {P2}
3. {P3}

Assistant: Let's think step by step inside <thought> tags about how each option aligns with or violates the provided principles. Then output the final choice as "A" or "B".
"""

JSON_SYSTEM = (
    "You return JSON objects only. Output a single json object that matches the "
    "requested keys. Do not wrap the json in markdown fences. Keep every string "
    "value concise (one to three sentences). Prefer a complete valid json object "
    "over extra verbosity."
)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "book_id": {"type": "string"},
        "author": {"type": "string"},
        "principles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_thesis": {"type": "string"},
                    "critique_request": {"type": "string"},
                    "revision_request": {"type": "string"},
                },
                "required": ["source_thesis", "critique_request", "revision_request"],
            },
        },
        "elicited_prompts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["book_id", "author", "principles", "elicited_prompts"],
}

SL_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "draft_0": {"type": "string"},
        "critique_1": {"type": "string"},
        "revision_1": {"type": "string"},
        "critique_2": {"type": "string"},
        "revision_2": {"type": "string"},
        "critique_3": {"type": "string"},
        "revision_3_final": {"type": "string"},
    },
    "required": ["draft_0", "revision_1", "revision_3_final"],
}

SL_BATCH_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": SL_ITEM_SCHEMA}},
    "required": ["items"],
}

RLAIF_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "choice": {"type": "string", "enum": ["A", "B"]},
    },
    "required": ["thought", "choice"],
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repair_truncated_json(text: str) -> str:
    """Close an unterminated string and unclosed braces/brackets so truncated JSON can parse."""
    s = text.strip()
    in_str = False
    esc = False
    stack: list[str] = []
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s)
    while stack:
        s = re.sub(r",\s*$", "", s)
        s += stack.pop()
    return s


def parse_json_lenient(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    try:
        data, _end = json.JSONDecoder().raw_decode(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    repaired = repair_truncated_json(raw)
    data = json.loads(repaired)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("repaired json is not an object", repaired, 0)
    return data


def _minutes_of_day(now: datetime) -> int:
    return now.hour * 60 + now.minute


def is_peak_utc(now: Optional[datetime] = None) -> bool:
    """True during DeepSeek peak: 01:00–04:00 and 06:00–10:00 UTC weekdays, not CN holidays."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    if now.weekday() >= 5:
        return False
    if now.date().isoformat() in CHINESE_PUBLIC_HOLIDAYS:
        return False
    m = _minutes_of_day(now)
    return (1 * 60 <= m < 4 * 60) or (6 * 60 <= m < 10 * 60)


def next_offpeak_utc(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    if not is_peak_utc(now):
        return now
    m = _minutes_of_day(now)
    if 1 * 60 <= m < 4 * 60:
        return now.replace(hour=4, minute=0, second=0, microsecond=0)
    if 6 * 60 <= m < 10 * 60:
        return now.replace(hour=10, minute=0, second=0, microsecond=0)
    return now


def next_peak_utc(now: Optional[datetime] = None) -> Optional[datetime]:
    """Next weekday peak start, or None if currently in peak."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    if is_peak_utc(now):
        return None
    cursor = now
    for _ in range(10):
        day = cursor.date()
        weekday = cursor.weekday()
        holiday = day.isoformat() in CHINESE_PUBLIC_HOLIDAYS
        if weekday < 5 and not holiday:
            for hh in (1, 6):
                cand = datetime(day.year, day.month, day.day, hh, 0, 0, tzinfo=timezone.utc)
                if cand > now:
                    return cand
        cursor = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc) + timedelta(days=1)
    return None


def wait_offpeak(min_remaining_s: int = PEAK_STOP_BUFFER_S) -> None:
    """Block until we can start an LLM request in off-peak with a small buffer."""
    while True:
        now = datetime.now(timezone.utc)
        if is_peak_utc(now):
            nxt = next_offpeak_utc(now)
            wait = max(1.0, (nxt - now).total_seconds())
            global _last_peak_log
            with _last_peak_log_lock:
                if time.time() - _last_peak_log >= 120:
                    log(f"peak window: sleeping {wait:.0f}s until off-peak {nxt.isoformat()}")
                    _last_peak_log = time.time()
            time.sleep(min(wait, 60.0))
            continue
        peak = next_peak_utc(now)
        if peak is not None:
            remaining = (peak - now).total_seconds()
            if remaining < min_remaining_s:
                # too close to peak start — wait out the upcoming peak
                nxt = next_offpeak_utc(peak + timedelta(seconds=1))
                wait = max(1.0, (nxt - now).total_seconds())
                log(f"peak imminent in {remaining:.0f}s; sleeping {wait:.0f}s until {nxt.isoformat()}")
                time.sleep(min(wait, 60.0))
                continue
        return


def ensure_dirs() -> None:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "extracts").mkdir(parents=True, exist_ok=True)


def resolve_api_key() -> str:
    """Read Cursor secret `DeepSeek`. Never print or write the value."""
    for name in ("DeepSeek", "DEEPSEEK_API_KEY", "DEEPSEEK"):
        val = os.environ.get(name)
        if val and val.strip():
            os.environ["DEEPSEEK_API_KEY"] = val.strip()
            return val.strip()
    seen = sorted(
        n
        for n in os.environ
        if re.search(r"DEEPSEEK|DeepSeek|GEMINI|Gemini|GOOGLE_API|API_KEY|SECRET", n, re.I)
    )
    raise SystemExit(
        "Missing DeepSeek API key. Checked DeepSeek, DEEPSEEK_API_KEY, DEEPSEEK. "
        f"Env names matching DEEPSEEK/GEMINI/API_KEY/SECRET: {seen or '(none)'}. "
        f"CLOUD_AGENT_ALL_SECRET_NAMES={os.environ.get('CLOUD_AGENT_ALL_SECRET_NAMES', '(unset)')}. "
        f"CLOUD_AGENT_INJECTED_SECRET_NAMES={os.environ.get('CLOUD_AGENT_INJECTED_SECRET_NAMES', '(unset)')}."
    )


def assert_no_leakage(text: str, where: str) -> None:
    low = text.lower()
    for term in LEAKAGE_TERMS:
        if term in low:
            raise RuntimeError(f"leakage term {term!r} in {where}")


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


_log_lock = threading.Lock()


_last_peak_log = 0.0
_last_peak_log_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{utcnow()}] {msg}"
    with _log_lock:
        try:
            print(line, flush=True)
        except Exception:
            pass
        try:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


class State:
    def __init__(self, path: Path = STATE_DB):
        ensure_dirs()
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA busy_timeout=8000")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._init()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT
            );
            CREATE TABLE IF NOT EXISTS api_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                step TEXT,
                model TEXT,
                ref TEXT,
                error_type TEXT,
                error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                step TEXT,
                model TEXT,
                ref TEXT,
                status TEXT,
                tokens_in INTEGER,
                tokens_out INTEGER,
                latency_ms INTEGER
            );
            CREATE TABLE IF NOT EXISTS books (
                writer_id TEXT PRIMARY KEY,
                author TEXT,
                work_title TEXT,
                pd_url TEXT,
                text_source_url TEXT,
                ier_intrinsic REAL,
                n_chars INTEGER,
                fetch_status TEXT,
                extract_status TEXT,
                n_principles INTEGER,
                n_prompts INTEGER,
                n_chunks INTEGER,
                error TEXT,
                extract_model TEXT
            );
            CREATE TABLE IF NOT EXISTS sl_cai (
                prompt_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                source TEXT,
                book_id TEXT,
                draft_0 TEXT,
                revision_1 TEXT,
                revision_2 TEXT,
                revision_3 TEXT,
                p1_json TEXT,
                p2_json TEXT,
                p3_json TEXT,
                status TEXT,
                error TEXT,
                model TEXT
            );
            CREATE TABLE IF NOT EXISTS rlaif (
                prompt_id TEXT PRIMARY KEY,
                ya TEXT,
                yb TEXT,
                order_ab TEXT,
                thought TEXT,
                choice TEXT,
                chosen TEXT,
                rejected TEXT,
                status TEXT,
                error TEXT,
                model TEXT
            );
            """
        )
        for sql in (
            "ALTER TABLE books ADD COLUMN extract_model TEXT",
            "ALTER TABLE sl_cai ADD COLUMN model TEXT",
            "ALTER TABLE rlaif ADD COLUMN model TEXT",
        ):
            try:
                self.conn.execute(sql)
            except sqlite3.OperationalError:
                pass

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq) -> None:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.executemany(sql, seq)
                self.conn.execute("COMMIT")
            except Exception:
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def get_meta(self, k: str, default: Optional[str] = None) -> Optional[str]:
        row = self.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return row[0] if row else default

    def set_meta(self, k: str, v: str) -> None:
        self.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, v))

    def log_error(self, step: str, model: str, ref: str, etype: str, msg: str) -> None:
        msg = (msg or "")[:1500]
        self.execute(
            "INSERT INTO api_errors(ts,step,model,ref,error_type,error_message) VALUES(?,?,?,?,?,?)",
            (utcnow(), step, model or MODEL_ID, ref, etype, msg),
        )
        log(f"API_ERROR step={step} model={model or MODEL_ID} ref={ref} type={etype} msg={msg[:200]}")

    def log_call(
        self,
        step: str,
        model: str,
        ref: str,
        status: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: int = 0,
    ) -> None:
        self.execute(
            "INSERT INTO api_calls(ts,step,model,ref,status,tokens_in,tokens_out,latency_ms) VALUES(?,?,?,?,?,?,?,?)",
            (utcnow(), step, model or MODEL_ID, ref, status, tokens_in, tokens_out, latency_ms),
        )

    def wipe_llm_rows(self) -> None:
        """Drop Gemini/prior LLM rows. Keep fetched books and schema."""
        self.execute("DELETE FROM api_calls")
        self.execute("DELETE FROM api_errors")
        self.execute("DELETE FROM sl_cai")
        self.execute("DELETE FROM rlaif")
        self.execute(
            """
            UPDATE books SET extract_status=NULL, n_principles=NULL, n_prompts=NULL,
              n_chunks=NULL, error=NULL, extract_model=NULL
            """
        )
        for k in (
            "gemini_model",
            "model_rotation",
            "constitution_n",
            "prompts_total",
            "prompts_elicited",
            "prompts_ultrachat",
        ):
            self.execute("DELETE FROM meta WHERE k=?", (k,))
        self.set_meta("llm_model", MODEL_ID)
        self.set_meta("llm_provider", "deepseek")
        self.set_meta("thinking", "disabled")

    def counts(self) -> dict[str, Any]:
        def one(sql: str, params: tuple = ()) -> int:
            row = self.execute(sql, params).fetchone()
            return int(row[0] or 0)

        err_by = dict(
            self.execute(
                "SELECT COALESCE(error_type,'unknown'), COUNT(*) FROM api_errors GROUP BY 1"
            ).fetchall()
        )
        models = dict(
            self.execute(
                "SELECT model, COUNT(*) FROM api_calls WHERE status='ok' GROUP BY 1"
            ).fetchall()
        )
        return {
            "books_fetched": one("SELECT COUNT(*) FROM books WHERE fetch_status='ok'"),
            "books_extracted": one(
                "SELECT COUNT(*) FROM books WHERE extract_status='ok' AND COALESCE(extract_model,'')=?",
                (MODEL_ID,),
            ),
            "principles_raw": one(
                "SELECT COALESCE(SUM(n_principles),0) FROM books WHERE COALESCE(extract_model,'')=?",
                (MODEL_ID,),
            ),
            "elicited_raw": one(
                "SELECT COALESCE(SUM(n_prompts),0) FROM books WHERE COALESCE(extract_model,'')=?",
                (MODEL_ID,),
            ),
            "sl_ok": one(
                "SELECT COUNT(*) FROM sl_cai WHERE status='ok' AND COALESCE(model,'')=?",
                (MODEL_ID,),
            ),
            "sl_err": one("SELECT COUNT(*) FROM sl_cai WHERE status='error'"),
            "rlaif_ok": one(
                "SELECT COUNT(*) FROM rlaif WHERE status='ok' AND COALESCE(model,'')=?",
                (MODEL_ID,),
            ),
            "rlaif_err": one("SELECT COUNT(*) FROM rlaif WHERE status='error'"),
            "api_ok": one("SELECT COUNT(*) FROM api_calls WHERE status='ok'"),
            "api_fail": one("SELECT COUNT(*) FROM api_calls WHERE status!='ok'"),
            "api_errors": one("SELECT COUNT(*) FROM api_errors"),
            "errors_by_type": err_by,
            "ok_by_model": models,
        }


class RateLimiter:
    def __init__(self, rpm: int = RPM, tpm: int = TPM_CAP):
        self.rpm = rpm
        self.tpm = tpm
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.starts: list[float] = []
        self.token_events: list[tuple[float, int]] = []

    def acquire(self, est_tokens: int) -> None:
        est_tokens = min(int(est_tokens), self.tpm)
        while True:
            with self.cond:
                now = time.time()
                self.starts = [t for t in self.starts if now - t < 60.0]
                self.token_events = [(t, n) for t, n in self.token_events if now - t < 60.0]
                used_tpm = sum(n for _, n in self.token_events)
                if len(self.starts) < self.rpm and used_tpm + est_tokens <= self.tpm:
                    self.starts.append(now)
                    self.token_events.append((now, est_tokens))
                    return
                wait_rpm = 0.0 if len(self.starts) < self.rpm else 60.0 - (now - self.starts[0]) + 0.05
                wait_tpm = 0.0
                if used_tpm + est_tokens > self.tpm and self.token_events:
                    wait_tpm = 60.0 - (now - self.token_events[0][0]) + 0.05
                wait = max(wait_rpm, wait_tpm, 0.2)
            time.sleep(min(wait, 5.0))

    def record_actual(self, actual_tokens: int, est_tokens: int) -> None:
        delta = int(actual_tokens) - int(est_tokens)
        if delta == 0:
            return
        with self.lock:
            now = time.time()
            self.token_events.append((now, max(0, delta)))


class LLMClient:
    """OpenAI-compatible DeepSeek client. Single model, thinking disabled."""

    def __init__(self, state: State):
        key = resolve_api_key()
        self.state = state
        self.model = MODEL_ID
        self.client = OpenAI(
            api_key=key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=300.0,
            max_retries=0,
        )
        self.limiter = RateLimiter()
        self.sem = threading.Semaphore(MAX_IN_FLIGHT)
        state.set_meta("llm_model", self.model)
        state.set_meta("llm_provider", "deepseek")
        state.set_meta("thinking", "disabled")
        now = datetime.now(timezone.utc)
        peak = next_peak_utc(now)
        log(
            f"single LLM model: {self.model} base={DEEPSEEK_BASE_URL} thinking=disabled "
            f"peak_now={is_peak_utc(now)} next_peak={peak.isoformat() if peak else 'in-peak'}"
        )
        for tmpl_name, tmpl in (
            ("EXTRACT_PROMPT", EXTRACT_PROMPT),
            ("SL_CAI_TEMPLATE", SL_CAI_TEMPLATE),
            ("RLAIF_TEMPLATE", RLAIF_TEMPLATE),
            ("JSON_SYSTEM", JSON_SYSTEM),
        ):
            assert_no_leakage(tmpl, tmpl_name)

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        step: str,
        ref: str,
        max_output_tokens: int = 8192,
        temperature: float = 0.6,
        retries: int = 6,
    ) -> dict[str, Any]:
        user = (
            prompt
            + "\n\nReturn a json object matching this shape:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        est = estimate_tokens(user) + max_output_tokens
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            wait_offpeak()
            self.limiter.acquire(min(est, 200_000))
            t0 = time.time()
            try:
                with self.sem:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": JSON_SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        response_format={"type": "json_object"},
                        extra_body={"thinking": {"type": "disabled"}},
                        max_tokens=max_output_tokens,
                        temperature=temperature,
                    )
                latency = int((time.time() - t0) * 1000)
                choice = resp.choices[0] if resp.choices else None
                msg = choice.message if choice else None
                text = ((msg.content if msg else None) or "").strip()
                reasoning = (getattr(msg, "reasoning_content", None) or "") if msg else ""
                if reasoning and not text:
                    raise ValueError("thinking_not_disabled: empty content with reasoning_content")
                finish = str(getattr(choice, "finish_reason", "") or "")
                usage = getattr(resp, "usage", None)
                tin = int(getattr(usage, "prompt_tokens", 0) or 0)
                tout = int(getattr(usage, "completion_tokens", 0) or 0)
                self.limiter.record_actual(tin + tout, min(est, 200_000))
                if not text:
                    raise ValueError(f"empty_response finish={finish}")
                try:
                    data = parse_json_lenient(text)
                    status = "ok_truncated" if finish == "length" else "ok"
                    self.state.log_call(step, self.model, ref, status, tin, tout, latency)
                    return data
                except json.JSONDecodeError as e:
                    last_err = e
                    self.state.log_call(
                        step, self.model, ref, "json_error", tin, tout, latency
                    )
                    self.state.log_error(step, self.model, ref, "JSONDecodeError", str(e))
                    time.sleep(min(2 ** attempt, 20))
                    continue
            except Exception as e:
                last_err = e
                msg = str(e)
                et = type(e).__name__
                self.state.log_call(step, self.model, ref, "error", 0, 0, int((time.time() - t0) * 1000))
                self.state.log_error(step, self.model, ref, et, msg)
                low = msg.lower()
                if "429" in msg or "rate" in low or "quota" in low:
                    sleep_s = 30 * (attempt + 1)
                    log(f"quota/backoff {sleep_s}s model={self.model} ref={ref}")
                    time.sleep(sleep_s)
                elif "503" in msg or "unavailable" in low or "overloaded" in low:
                    time.sleep(min(8 * (attempt + 1), 45))
                else:
                    time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"generate_json failed after retries step={step} ref={ref}: {last_err}")


# Backward-compatible name used by older script drafts
GeminiPool = LLMClient


def strip_gutenberg(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    start = 0
    for m in start_markers:
        i = text.find(m)
        if i != -1:
            j = text.find("\n", i)
            start = (j + 1) if j != -1 else i
            break
    end = len(text)
    for m in end_markers:
        i = text.find(m, start)
        if i != -1:
            end = i
            break
    body = text[start:end]
    body = re.sub(r"\r\n?", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p>", "\n\n", html)
    html = re.sub(r"(?i)</div>", "\n", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"&lt;", "<", html)
    html = re.sub(r"&gt;", ">", html)
    html = re.sub(r"&#\d+;", " ", html)
    return re.sub(r"[ \t]+\n", "\n", html)


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        if j < n:
            k = text.rfind("\n\n", i + max_chars // 2, j)
            if k != -1:
                j = k
        piece = text[i:j].strip()
        if piece:
            chunks.append(piece)
        i = j if j > i else i + max_chars
    return chunks


def prompt_id_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    staging = Path("/tmp") / f"{path.name}.{os.getpid()}.jsonl"
    with staging.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        staging.replace(path)
    except Exception:
        import shutil

        shutil.copy2(staging, path)
        staging.unlink(missing_ok=True)


STORE_STATE_DB = DATA_DIR / "state.db"
DPO_JSONL = DATA_DIR / "rlaif_dpo_train.jsonl"
DPO_SIDECAR = DATA_DIR / "rlaif_dpo_checkpoint.jsonl"


def append_jsonl_line(path: Path, rec: dict) -> None:
    """Append one JSONL record on the store path and fsync. Raises on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def try_sync_state_db() -> bool:
    """Copy CAI_STATE_DB (/tmp) back onto the store when writable."""
    src = Path(STATE_DB)
    dest = STORE_STATE_DB
    try:
        if src.resolve() == dest.resolve():
            return True
        if not src.exists():
            return False
        import shutil

        # Stage on local disk so the store never needs a second 18MB copy.
        tmp = Path("/tmp/cai-state.db.sync")
        shutil.copy2(src, tmp)
        try:
            shutil.copy2(tmp, dest)
            return True
        except OSError as e:
            if getattr(e, "errno", None) == 28:
                try:
                    dest.unlink()
                except Exception:
                    pass
                shutil.copy2(tmp, dest)
                return True
            raise
    except Exception as e:
        log(f"store state.db sync failed: {type(e).__name__}: {e}")
        return False


def write_coverage(state: State, extra: Optional[dict] = None) -> None:
    extra = extra or {}
    counts = state.counts()
    model = state.get_meta("llm_model") or MODEL_ID
    lines = [
        "# CAI coverage (Steps 0–4) — DeepSeek",
        "",
        f"Generated: {utcnow()}",
        "",
        "DeepSeek-only constitution / SL-CAI / RLAIF data build over the I/E-R intrinsic top 40.",
        "No Colab, GPU, Unsloth, or Drive. Ranked by I/E-R intrinsic only (no I−E filter).",
        "Extraction, critique, revision, and judge prompts use author-text vocabulary only.",
        "Gemini-generated principles, prompts, SL-CAI, and RLAIF labels were discarded and regenerated.",
        "",
        "## Models",
        "",
        f"- Single model (Steps 1–4): `{model}`",
        "- Thinking: disabled (`thinking: {type: disabled}`; no reasoning_effort)",
        f"- Client: OpenAI-compatible `{DEEPSEEK_BASE_URL}`",
        f"- Rate limit: {RPM} RPM, TPM cap {TPM_CAP}, max in-flight {MAX_IN_FLIGHT}",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Books fetched | {counts['books_fetched']} |",
        f"| Books extracted (`{MODEL_ID}`) | {counts['books_extracted']} |",
        f"| Raw principles (pre-cluster) | {counts['principles_raw']} |",
        f"| Elicited prompts (raw sum) | {counts['elicited_raw']} |",
        f"| Constitution principles | {extra.get('constitution_n', '')} |",
        f"| Prompt set total | {extra.get('prompts_total', '')} |",
        f"| Prompt set elicited | {extra.get('prompts_elicited', '')} |",
        f"| Prompt set UltraChat | {extra.get('prompts_ultrachat', '')} |",
        f"| SL-CAI rows ok | {counts['sl_ok']} |",
        f"| SL-CAI rows error | {counts['sl_err']} |",
        f"| RLAIF rows ok | {counts['rlaif_ok']} |",
        f"| RLAIF rows error | {counts['rlaif_err']} |",
        f"| API calls ok | {counts['api_ok']} |",
        f"| API calls non-ok | {counts['api_fail']} |",
        f"| API errors logged | {counts['api_errors']} |",
        "",
        "## API errors by type",
        "",
    ]
    err_by = counts.get("errors_by_type") or {}
    if not err_by:
        lines += ["(none)", ""]
    else:
        lines += ["| Type | n |", "|---|---|"]
        for k, v in sorted(err_by.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{k}` | {v} |")
        lines.append("")
    lines += ["## Successful calls by model", ""]
    ok_by = counts.get("ok_by_model") or {}
    if not ok_by:
        lines += ["(none yet)", ""]
    else:
        lines += ["| Model | n |", "|---|---|"]
        for k, v in sorted(ok_by.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{k}` | {v} |")
        lines.append("")
    notes = extra.get("notes") or []
    if notes:
        lines += ["## Notes", ""]
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    lines += [
        "## Outputs",
        "",
        "- `/cursor/stores/self/cai-pipeline/data/corpus_manifest.json`",
        "- `/cursor/stores/self/cai-pipeline/data/raw_principles.jsonl`",
        "- `/cursor/stores/self/cai-pipeline/data/constitution.json`",
        "- `/cursor/stores/self/cai-pipeline/data/prompts_dataset.jsonl`",
        "- `/cursor/stores/self/cai-pipeline/data/sl_cai_train.jsonl`",
        "- `/cursor/stores/self/cai-pipeline/data/rlaif_dpo_train.jsonl`",
        "- `/cursor/stores/self/cai-pipeline/data/state.db`",
        "",
    ]
    body = "\n".join(lines)
    COVERAGE_MD.write_text(body, encoding="utf-8")
    try:
        COVERAGE_MD_LEGACY.write_text(body, encoding="utf-8")
    except Exception:
        pass
    log(f"wrote coverage {COVERAGE_MD}")


def format_exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
