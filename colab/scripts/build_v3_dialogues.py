#!/usr/bin/env python3
"""Build v3 dialogue mixes from existing v2 draft_0 / revision_3 traces.

Filter: top 20 of the 40 treatment books by ier_intrinsic. No I-E cutoff.
Rewrite each kept trace with deepseek-flash, thinking off, into one weighed
assistant completion plus a dogmatic rejected answer. The saved training row
is the original user dilemma and that weighed completion. Scores stay in
metadata files and are never sent to the model.

Arms:
  weighted  w = s / mean(s_kept), no clip, mean CAI weight renormalized to 1
  uniform   the same rows and row ids, w = 1
  control   citation-control dialogues, same template, stratified subsample to N

No UltraChat. Shared 300-row format replay is appended to every arm at weight 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "colab" / "data" / "v3"
CACHE_PATH = DATA / "rewrite_cache.jsonl"
SUMMARY_PATH = DATA / "mix_summary.json"
META_PATH = DATA / "cai_meta.jsonl"

TREAT_DB_CANDIDATES = (
    Path(os.environ.get("CAI_TREAT_STATE") or ""),
    Path("/tmp/treat_state.db"),
    Path("/cursor/stores/self/cai-pipeline/data/state.db"),
    REPO / "cai-pipeline" / "data" / "state.db",
)
CTRL_DB_CANDIDATES = (
    Path(os.environ.get("CAI_CTRL_STATE") or ""),
    Path("/tmp/ctrl_state.db"),
    Path("/cursor/stores/self/cai-pipeline-control/data/state.db"),
    REPO / "cai-pipeline-control" / "data" / "state.db",
)

MODEL_ID = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TOP_K = 20
SUBSAMPLE_SEED = 0
FORMAT_REPLAY_N = 300

LEAKAGE_TERMS = (
    "vanderweele",
    "vander weele",
    "flourishing",
    "fai benchmark",
    "fai taxonomy",
    "human flourishing",
    "global flourishing",
)
ARCHAIC_RE = re.compile(r"\b(thou|thee|thy|thine|ye|hath|doth|unto|brethren)\b", re.I)
_ARCHAIC_WORDS = (
    (re.compile(r"\bthou\b", re.I), "you"),
    (re.compile(r"\bthee\b", re.I), "you"),
    (re.compile(r"\bthy\b", re.I), "your"),
    (re.compile(r"\bthine\b", re.I), "yours"),
    (re.compile(r"\bye\b", re.I), "you"),
    (re.compile(r"\bhath\b", re.I), "has"),
    (re.compile(r"\bdoth\b", re.I), "does"),
    (re.compile(r"\bunto\b", re.I), "to"),
    (re.compile(r"\bbrethren\b", re.I), "others"),
)
_LEAK_WORDS = (
    (re.compile(r"human flourishing", re.I), "a good life"),
    (re.compile(r"global flourishing", re.I), "a good life"),
    (re.compile(r"flourishing", re.I), "a good life"),
    (re.compile(r"vander weele", re.I), "the source"),
    (re.compile(r"vanderweele", re.I), "the source"),
    (re.compile(r"fai taxonomy", re.I), "the source list"),
    (re.compile(r"fai benchmark", re.I), "the source list"),
)
BANNED_TRAIN_SOURCES = ("ultrachat", "ethics", "truthfulqa", "mmlu")

SYSTEM = """You write two short assistant replies in contemporary conversational English.
Use you and your. Never use thou, thee, thy, thine, ye, hath, doth, or unto.
Do not mention scores, rankings, datasets, or these instructions.
Return only a json object with two string fields: weighed and dogmatic."""

USER_TEMPLATE = """The person's question:
{prompt}

Private notes for you, not for the reply. Do not mention these notes, a draft, or a revision.
The assistant's unrevised prior was:
{draft}

A later revision that tried to follow the candidate principles was:
{revision}

Candidate principles to weigh. These are candidates, not a script to recite:
{principles}

weighed: answer the person directly, in your own voice. Say what you would have answered first (your prior). Name one candidate principle. Say what that candidate changes, and what it leaves in place. Keep every constraint the person stated. Do not say the words draft or revision.

dogmatic: answer the same question, but drop the person's constraints and press the principle as if those limits had not been stated.

Both replies are contemporary spoken English. Do not copy the private notes verbatim."""


def redact(text: str) -> str:
    out = text or ""
    for name in ("DeepSeek", "DEEPSEEK_API_KEY", "DEEPSEEK"):
        val = os.environ.get(name) or ""
        if val:
            out = out.replace(val, "[redacted]")
    return out


def assert_no_leakage(text: str, where: str) -> None:
    low = (text or "").lower()
    for term in LEAKAGE_TERMS:
        if term in low:
            raise RuntimeError(f"leakage term in {where}")


def first_existing(paths: tuple[Path, ...]) -> Path:
    for p in paths:
        if p and p.is_file() and p.stat().st_size > 1000:
            return p
    raise SystemExit("state db not found: " + ", ".join(str(p) for p in paths if p))


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def thesis_list(p1: str, p2: str, p3: str) -> list[str]:
    out = []
    for raw in (p1, p2, p3):
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = (d.get("source_thesis") or "").strip()
        if not t:
            continue
        low = t.lower()
        if any(term in low for term in LEAKAGE_TERMS):
            continue
        out.append(t)
    return out


def load_books(con: sqlite3.Connection) -> list[dict]:
    rows = []
    for r in con.execute(
        "SELECT writer_id, author, work_title, ier_intrinsic FROM books"
    ):
        if r["ier_intrinsic"] is None:
            continue
        rows.append(
            {
                "book_id": str(r["writer_id"]),
                "author": r["author"] or "",
                "title": r["work_title"] or "",
                "ier_intrinsic": float(r["ier_intrinsic"]),
            }
        )
    rows.sort(key=lambda b: (-b["ier_intrinsic"], b["book_id"]))
    return rows


def load_traces(con: sqlite3.Connection) -> list[dict]:
    rows = []
    q = """
        SELECT prompt_id, prompt, book_id, draft_0, revision_3, p1_json, p2_json, p3_json
        FROM sl_cai
        WHERE status='ok' AND source='elicited'
          AND draft_0 IS NOT NULL AND revision_3 IS NOT NULL
          AND length(draft_0) > 0 AND length(revision_3) > 0
    """
    for r in con.execute(q):
        prompt = (r["prompt"] or "").strip()
        if not prompt:
            continue
        rows.append(
            {
                "prompt_id": str(r["prompt_id"]),
                "prompt": prompt,
                "book_id": str(r["book_id"]),
                "draft_0": (r["draft_0"] or "").strip(),
                "revision_3": (r["revision_3"] or "").strip(),
                "theses": thesis_list(r["p1_json"], r["p2_json"], r["p3_json"]),
            }
        )
    return rows


def kish_ess(weights: list[float]) -> float:
    s = float(sum(weights))
    sq = float(sum(w * w for w in weights))
    if sq <= 0:
        raise RuntimeError("Kish ESS undefined for empty or zero weights")
    return (s * s) / sq


def renorm(scores: list[float]) -> list[float]:
    if not scores:
        raise RuntimeError("no scores to weight")
    if any(s <= 0 for s in scores):
        raise RuntimeError("ier_intrinsic must be positive for kept rows")
    mean_s = sum(scores) / len(scores)
    raw = [s / mean_s for s in scores]
    mean_w = sum(raw) / len(raw)
    return [w / mean_w for w in raw]


def stratified_take(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Equal allocation across books, without replacement. No upsampling."""
    if len(rows) < n:
        raise SystemExit(
            f"control pool has {len(rows)} rows, need {n}. Not upsampling."
        )
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["book_id"]].append(r)
    books = sorted(by)
    if not books:
        raise SystemExit(f"control pool has no books, need {n}")
    rng = random.Random(seed)
    order = books[:]
    rng.shuffle(order)
    base, rem = divmod(n, len(order))
    quota = {b: base + (1 if i < rem else 0) for i, b in enumerate(order)}
    # Spill if a book is short. The v2 control books are all above this quota.
    short = [b for b in order if len(by[b]) < quota[b]]
    spare = [b for b in order if len(by[b]) > quota[b]]
    for b in short:
        need = quota[b] - len(by[b])
        quota[b] = len(by[b])
        i = 0
        while need > 0 and spare:
            donor = spare[i % len(spare)]
            room = len(by[donor]) - quota[donor]
            if room <= 0:
                spare = [s for s in spare if s != donor]
                continue
            take = min(room, need)
            quota[donor] += take
            need -= take
            i += 1
        if need > 0:
            raise SystemExit(
                f"control pool cannot fill {n} without replacement after book caps"
            )
    if sum(quota.values()) != n:
        raise SystemExit(f"quota sum {sum(quota.values())} != {n}")
    picked: list[dict] = []
    for b in sorted(quota):
        pool = sorted(by[b], key=lambda r: r["prompt_id"])
        rr = random.Random(seed + int(hashlib.sha256(b.encode()).hexdigest()[:8], 16))
        rr.shuffle(pool)
        if len(pool) < quota[b]:
            raise SystemExit(f"book {b} has {len(pool)} < quota {quota[b]}")
        picked.extend(pool[: quota[b]])
    if len(picked) != n:
        raise SystemExit(f"picked {len(picked)} != {n}")
    if len({r["prompt_id"] for r in picked}) != n:
        raise SystemExit("stratified sample repeated a prompt id")
    return picked


def assign_rejected(rows: list[dict]) -> None:
    rows.sort(key=lambda r: (r["book_id"], r["prompt_id"]))
    for i, r in enumerate(rows):
        r["rejected_kind"] = "draft" if i % 2 == 0 else "dogmatic"


def _strip_leak(text: str) -> str:
    out = text or ""
    for pat, repl in _LEAK_WORDS:
        out = pat.sub(repl, out)
    return out


def _sanitize_private(text: str) -> str:
    """Clean notes sent to the rewriter. The saved training prompt is not passed through this."""
    out = text or ""
    for pat, repl in _LEAK_WORDS + _ARCHAIC_WORDS:
        out = pat.sub(repl, out)
    return out


def build_user_prompt(row: dict) -> str:
    principles = "\n".join(
        f"{i}. {_sanitize_private(t)}" for i, t in enumerate(row["theses"], 1) if _sanitize_private(t).strip()
    )
    if not principles:
        principles = "(no candidate principle text was stored for this trace)"
    text = USER_TEMPLATE.format(
        prompt=row["prompt"],
        draft=_sanitize_private(row["draft_0"]),
        revision=_sanitize_private(row["revision_3"]),
        principles=principles,
    )
    assert_no_leakage(SYSTEM, "rewrite system")
    assert_no_leakage(text, f"rewrite user {row['prompt_id']}")
    return text


def _strip_json(text: str) -> dict:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("rewrite json was not an object")
    return data


def valid_reply(text: str, *, draft: str, revision: str) -> str | None:
    t = (text or "").strip()
    if len(t) < 80:
        return "too_short"
    low = t.lower()
    for term in LEAKAGE_TERMS:
        if term in low:
            return "leakage"
    if ARCHAIC_RE.search(t):
        return "archaic"
    if re.search(r"\b(your draft|the draft|the revision|private notes|unrevised)\b", t, re.I):
        return "mentions_draft"
    norm = re.sub(r"\s+", " ", t).strip().lower()
    for other in (draft, revision):
        on = re.sub(r"\s+", " ", other or "").strip().lower()
        if on and norm == on:
            return "copied_trace"
    return None


def load_cache() -> dict[str, dict]:
    out = {}
    if not CACHE_PATH.is_file():
        return out
    with CACHE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("ok"):
                out[rec["key"]] = rec
    return out


def append_cache(rec: dict, lock: threading.Lock) -> None:
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with lock:
        with CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(line)


def call_deepseek(client, row: dict, temperature: float, repair: str = "") -> dict:
    user = build_user_prompt(row)
    if repair:
        user += (
            "\n\nThe previous reply was rejected ("
            + _sanitize_private(repair)
            + "). Write both replies again in contemporary spoken English. "
            "Do not use thou, thee, thy, thine, ye, hath, doth, unto, or brethren. "
            "Do not copy the private notes."
        )
    assert_no_leakage(user, f"rewrite user {row['prompt_id']}")
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=900,
        temperature=temperature,
    )
    choice = resp.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    content = (msg.content or "").strip()
    if reasoning and not content:
        raise RuntimeError("thinking_not_disabled")
    data = _strip_json(content)
    weighed = str(data.get("weighed") or "").strip()
    dogmatic = str(data.get("dogmatic") or "").strip()
    err_w = valid_reply(weighed, draft=row["draft_0"], revision=row["revision_3"])
    err_d = valid_reply(dogmatic, draft=row["draft_0"], revision=row["revision_3"])
    if err_w or err_d:
        raise RuntimeError(f"invalid weighed={err_w} dogmatic={err_d}")
    if re.sub(r"\s+", " ", weighed).lower() == re.sub(r"\s+", " ", dogmatic).lower():
        raise RuntimeError("dogmatic_equals_weighed")
    return {"weighed": weighed, "dogmatic": dogmatic}


def rewrite_all(rows: list[dict], workers: int) -> dict[str, dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    key = ""
    for name in ("DeepSeek", "DEEPSEEK_API_KEY", "DEEPSEEK"):
        val = os.environ.get(name) or ""
        if val.strip():
            key = val.strip()
            break
    if not key:
        raise SystemExit("DeepSeek key is not in the environment. Refusing to invent replies.")

    cache = load_cache()
    todo = [r for r in rows if f"{r['split']}:{r['prompt_id']}" not in cache]
    print(f"rewrite cache {len(cache)} hit, {len(todo)} to call, model={MODEL_ID} thinking=off", flush=True)
    if not todo:
        return cache

    client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL, timeout=180.0, max_retries=0)
    lock = threading.Lock()
    done = 0
    failed: list[str] = []

    def one(row: dict) -> None:
        nonlocal done
        last = "unknown"
        for attempt, temp in enumerate((0.3, 0.5, 0.7, 0.9, 1.0, 1.1)):
            try:
                got = call_deepseek(client, row, temp, repair="" if attempt == 0 else last)
                rec = {
                    "key": f"{row['split']}:{row['prompt_id']}",
                    "split": row["split"],
                    "prompt_id": row["prompt_id"],
                    "book_id": row["book_id"],
                    "ok": True,
                    "model": MODEL_ID,
                    "weighed": got["weighed"],
                    "dogmatic": got["dogmatic"],
                }
                append_cache(rec, lock)
                with lock:
                    cache[rec["key"]] = rec
                    done += 1
                    n = done
                if n % 25 == 0 or n == len(todo):
                    print(f"rewrote {n}/{len(todo)}", flush=True)
                return
            except Exception as e:
                last = redact(str(e))[:300]
                time.sleep(min(2 ** attempt, 20) + (5 if "429" in last else 0))
        with lock:
            failed.append(f"{row['split']}:{row['prompt_id']} {last}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            fut.result()
    if failed:
        preview = "\n".join(failed[:8])
        raise SystemExit(f"{len(failed)} rewrites failed.\n{preview}")
    return cache


def row_id(split: str, prompt_id: str) -> str:
    return f"{split}:{prompt_id}"


def assemble(split_rows: dict[str, list[dict]], cache: dict[str, dict], book_s: dict[str, float]) -> None:
    treat = split_rows["treatment"]
    assign_rejected(treat)
    scores = [book_s[r["book_id"]] for r in treat]
    weights = renorm(scores)
    for r, w, s in zip(treat, weights, scores):
        r["weight"] = w
        r["s"] = s
        hit = cache[f"treatment:{r['prompt_id']}"]
        r["weighed"] = hit["weighed"]
        r["dogmatic"] = hit["dogmatic"]

    n = len(treat)
    ctrl_pool = split_rows["control_pool"]
    ctrl = stratified_take(ctrl_pool, n, SUBSAMPLE_SEED)
    assign_rejected(ctrl)
    for r in ctrl:
        r["weight"] = 1.0
        r["s"] = None
        hit = cache[f"control:{r['prompt_id']}"]
        r["weighed"] = hit["weighed"]
        r["dogmatic"] = hit["dogmatic"]

    def materialize(rows: list[dict], split: str, force_weight: float | None) -> tuple[list[dict], list[dict]]:
        sft, dpo = [], []
        for r in rows:
            w = 1.0 if force_weight is not None else float(r["weight"])
            # The unrevised draft is the rejected answer. Strip banned terms from that
            # text so they never enter a training prompt. Archaic wording stays,
            # because that draft is the negative example.
            rejected = r["draft_0"] if r["rejected_kind"] == "draft" else r["dogmatic"]
            rejected = _strip_leak(rejected)
            if _strip_leak(r["prompt"]) != r["prompt"]:
                raise SystemExit(f"user prompt contains a banned term: {r['prompt_id']}")
            rid = row_id(split if split != "uniform" else "treatment", r["prompt_id"])
            base = {
                "row_id": rid,
                "prompt": r["prompt"],
                "source": "elicited",
                "book_id": r["book_id"],
                "weight": w,
                "rejected_kind": r["rejected_kind"],
                "model": MODEL_ID,
            }
            sft.append({**base, "completion": r["weighed"]})
            dpo.append({**base, "chosen": r["weighed"], "rejected": rejected})
        return sft, dpo

    weighted_sft, weighted_dpo = materialize(treat, "treatment", None)
    uniform_sft, uniform_dpo = materialize(treat, "uniform", 1.0)
    control_sft, control_dpo = materialize(ctrl, "control", 1.0)

    fr_sft, fr_dpo = load_format_replay()
    arms = {
        "weighted": (weighted_sft + fr_sft, weighted_dpo + fr_dpo),
        "uniform": (uniform_sft + fr_sft, uniform_dpo + fr_dpo),
        "control": (control_sft + fr_sft, control_dpo + fr_dpo),
    }
    for name, (sft, dpo) in arms.items():
        dest = DATA / name
        dest.mkdir(parents=True, exist_ok=True)
        write_jsonl(dest / "sl_cai_train.jsonl", sft)
        write_jsonl(dest / "rlaif_dpo_train.jsonl", dpo)

    cai_w = [float(r["weight"]) for r in weighted_sft]
    ess = kish_ess(cai_w)
    summary = {
        "model": MODEL_ID,
        "thinking": "disabled",
        "top_k_books": TOP_K,
        "n_cai": n,
        "n_format_replay": len(fr_sft),
        "n_sft": {k: len(v[0]) for k, v in arms.items()},
        "n_dpo": {k: len(v[1]) for k, v in arms.items()},
        "rejected_kind_cai": {
            arm: {
                "draft": sum(1 for r in rows if r.get("rejected_kind") == "draft"),
                "dogmatic": sum(1 for r in rows if r.get("rejected_kind") == "dogmatic"),
            }
            for arm, rows in (
                ("weighted", weighted_dpo),
                ("uniform", uniform_dpo),
                ("control", control_dpo),
            )
        },
        "weight_clip": None,
        "mean_cai_weight": sum(cai_w) / len(cai_w),
        "max_weight": max(cai_w),
        "min_weight": min(cai_w),
        "kish_ess": ess,
        "kish_ess_over_n": ess / n,
        "subsample_seed": SUBSAMPLE_SEED,
        "control_books": len({r["book_id"] for r in control_sft}),
        "treatment_books": [
            {
                "book_id": b["book_id"],
                "ier_intrinsic": b["ier_intrinsic"],
                "n": sum(1 for r in treat if r["book_id"] == b["book_id"]),
            }
            for b in sorted(
                (b for b in load_books(connect(first_existing(TREAT_DB_CANDIDATES)))[:TOP_K]),
                key=lambda b: (-b["ier_intrinsic"], b["book_id"]),
            )
        ],
        "no_ultrachat": True,
        "trained_on_ethics_tqa_mmlu": False,
    }
    # treatment_books reload is awkward if db path differs; fill from treat rows + book_s below
    summary["treatment_books"] = book_summary(treat, book_s)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    meta = [
        {
            "row_id": r["row_id"],
            "book_id": r["book_id"],
            "s": next(t["s"] for t in treat if row_id("treatment", t["prompt_id"]) == r["row_id"]),
            "weight": r["weight"],
        }
        for r in weighted_sft
    ]
    write_jsonl(META_PATH, meta)
    validate_mixes(arms, n)
    print(json.dumps({k: summary[k] for k in (
        "n_cai", "n_sft", "n_dpo", "mean_cai_weight", "max_weight", "min_weight", "kish_ess", "rejected_kind_cai"
    )}, indent=2))


def book_summary(treat: list[dict], book_s: dict[str, float]) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for r in treat:
        counts[r["book_id"]] += 1
    books = [{"book_id": b, "ier_intrinsic": book_s[b], "n": counts[b]} for b in counts]
    books.sort(key=lambda b: (-b["ier_intrinsic"], b["book_id"]))
    return books


def load_format_replay() -> tuple[list[dict], list[dict]]:
    sft_path = REPO / "colab" / "data" / "v2" / "format_replay_sft.jsonl"
    dpo_path = REPO / "colab" / "data" / "v2" / "format_replay_dpo.jsonl"
    sft_raw = read_jsonl(sft_path)
    dpo_raw = read_jsonl(dpo_path)
    if len(sft_raw) != FORMAT_REPLAY_N or len(dpo_raw) != FORMAT_REPLAY_N:
        raise SystemExit(f"format replay expected {FORMAT_REPLAY_N}, got sft={len(sft_raw)} dpo={len(dpo_raw)}")
    sft, dpo = [], []
    for i, r in enumerate(sft_raw):
        blob = (r.get("prompt") or "") + "\n" + (r.get("completion") or "")
        assert_no_leakage(blob, f"format replay sft {i}")
        sft.append(
            {
                "row_id": f"replay:{i}",
                "prompt": r["prompt"],
                "completion": r["completion"],
                "source": "format_replay",
                "book_id": None,
                "weight": 1.0,
                "rejected_kind": None,
                "model": r.get("model"),
            }
        )
    for i, r in enumerate(dpo_raw):
        blob = (r.get("prompt") or "") + "\n" + (r.get("chosen") or "") + "\n" + (r.get("rejected") or "")
        assert_no_leakage(blob, f"format replay dpo {i}")
        dpo.append(
            {
                "row_id": f"replay:{i}",
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
                "source": "format_replay",
                "book_id": None,
                "weight": 1.0,
                "rejected_kind": None,
                "model": r.get("model"),
            }
        )
    write_jsonl(DATA / "format_replay_sft.jsonl", sft)
    write_jsonl(DATA / "format_replay_dpo.jsonl", dpo)
    return sft, dpo


def validate_mixes(arms: dict, n: int) -> None:
    w_sft, w_dpo = arms["weighted"]
    u_sft, u_dpo = arms["uniform"]
    c_sft, c_dpo = arms["control"]
    if [r["row_id"] for r in w_sft] != [r["row_id"] for r in u_sft]:
        raise SystemExit("uniform row ids differ from weighted")
    if [r["prompt"] for r in w_sft] != [r["prompt"] for r in u_sft]:
        raise SystemExit("uniform prompts differ from weighted")
    if [r["completion"] for r in w_sft] != [r["completion"] for r in u_sft]:
        raise SystemExit("uniform completions differ from weighted")
    if any(float(r["weight"]) != 1.0 for r in u_sft + c_sft):
        raise SystemExit("uniform or control weight was not 1")
    cai_w = [r for r in w_sft if r["source"] == "elicited"]
    if len(cai_w) != n:
        raise SystemExit(f"weighted CAI {len(cai_w)} != {n}")
    if sum(1 for r in c_sft if r["source"] == "elicited") != n:
        raise SystemExit("control CAI count != N")
    if abs(sum(float(r["weight"]) for r in cai_w) / n - 1.0) > 1e-6:
        raise SystemExit("mean CAI weight is not 1")
    for label, rows in (("weighted", w_dpo), ("uniform", u_dpo), ("control", c_dpo)):
        kinds = {r.get("rejected_kind") for r in rows if r["source"] == "elicited"}
        if kinds != {"draft", "dogmatic"}:
            raise SystemExit(f"{label} rejected kinds {kinds}")
        for r in rows:
            src = (r.get("source") or "").lower()
            if any(b in src for b in BANNED_TRAIN_SOURCES):
                raise SystemExit(f"banned source {src}")
            blob = " ".join(str(r.get(k) or "") for k in ("prompt", "completion", "chosen", "rejected"))
            assert_no_leakage(blob, f"{label} train text")
            # Raw scores are not a column on the train files.
            if "s" in r or "ier" in r:
                raise SystemExit("score field leaked into a train row")
    replay_ids = [r["row_id"] for r in w_sft if r["source"] == "format_replay"]
    if replay_ids != [r["row_id"] for r in u_sft if r["source"] == "format_replay"]:
        raise SystemExit("replay rows differ between weighted and uniform")
    if replay_ids != [r["row_id"] for r in c_sft if r["source"] == "format_replay"]:
        raise SystemExit("replay rows differ on control")
    if len(replay_ids) != FORMAT_REPLAY_N:
        raise SystemExit("replay count")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def self_test() -> None:
    scores = [1.0, 2.0, 3.0]
    w = renorm(scores)
    assert abs(sum(w) / len(w) - 1) < 1e-9
    assert max(w) > 1 and min(w) < 1
    ess = kish_ess(w)
    assert 0 < ess < len(w)
    rows = []
    for b in ("a", "b", "c", "d"):
        for i in range(10):
            rows.append({"book_id": b, "prompt_id": f"{b}-{i:02d}", "prompt": "q"})
    picked = stratified_take(rows, 10, 0)
    assert len(picked) == 10
    counts = defaultdict(int)
    for r in picked:
        counts[r["book_id"]] += 1
    assert set(counts) == {"a", "b", "c", "d"}
    assert max(counts.values()) - min(counts.values()) <= 1
    try:
        stratified_take(rows, 100, 0)
        raise AssertionError("should have stopped")
    except SystemExit as e:
        assert "Not upsampling" in str(e)
    assign_rejected(picked)
    kinds = [r["rejected_kind"] for r in picked]
    assert "draft" in kinds and "dogmatic" in kinds
    assert_no_leakage(SYSTEM + USER_TEMPLATE, "templates")
    print("self-test ok", {"ess": round(ess, 4), "max_w": round(max(w), 4)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="rewrite only this many rows per split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mix-only", action="store_true", help="assemble mixes from a complete cache")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    if args.mix_only:
        treat_db = first_existing(TREAT_DB_CANDIDATES)
        ctrl_db = first_existing(CTRL_DB_CANDIDATES)
        tcon = connect(treat_db)
        ccon = connect(ctrl_db)
        books = load_books(tcon)
        top_ids = {b["book_id"] for b in books[:TOP_K]}
        book_s = {b["book_id"]: b["ier_intrinsic"] for b in books}
        treat_rows = [r for r in load_traces(tcon) if r["book_id"] in top_ids]
        ctrl_rows = load_traces(ccon)
        tcon.close()
        ccon.close()
        cache = load_cache()
        missing_t = [r["prompt_id"] for r in treat_rows if f"treatment:{r['prompt_id']}" not in cache]
        if missing_t:
            raise SystemExit(f"mix-only missing {len(missing_t)} treatment rewrites")
        assemble({"treatment": treat_rows, "control_pool": ctrl_rows}, cache, book_s)
        return

    DATA.mkdir(parents=True, exist_ok=True)
    treat_db = first_existing(TREAT_DB_CANDIDATES)
    ctrl_db = first_existing(CTRL_DB_CANDIDATES)
    print("treatment db", treat_db)
    print("control db", ctrl_db)
    tcon = connect(treat_db)
    ccon = connect(ctrl_db)
    books = load_books(tcon)
    if len(books) < TOP_K:
        raise SystemExit(f"need at least {TOP_K} treatment books, found {len(books)}")
    top = books[:TOP_K]
    # Closed cutoff is rank, not an I-E sign. Keep a book even if some other index is negative.
    top_ids = {b["book_id"] for b in top}
    book_s = {b["book_id"]: b["ier_intrinsic"] for b in books}
    treat_rows = [r for r in load_traces(tcon) if r["book_id"] in top_ids]
    ctrl_rows = load_traces(ccon)
    tcon.close()
    ccon.close()
    if not treat_rows:
        raise SystemExit("no treatment traces in the top 20 books")
    print(f"top {TOP_K} books, treatment traces {len(treat_rows)}, control pool {len(ctrl_rows)}")
    if len(ctrl_rows) < len(treat_rows):
        raise SystemExit(
            f"control pool {len(ctrl_rows)} < treatment N {len(treat_rows)}. Not upsampling."
        )

    for r in treat_rows:
        r["split"] = "treatment"
    for r in ctrl_rows:
        r["split"] = "control"

    # Rewrite every treatment trace and a stratified superset of control traces.
    # Spares cover a failed row without upsampling past the real pool.
    n = len(treat_rows)
    primary = stratified_take(ctrl_rows, n, SUBSAMPLE_SEED)
    primary_ids = {r["prompt_id"] for r in primary}
    rest = [r for r in ctrl_rows if r["prompt_id"] not in primary_ids]
    # One spare per book, deterministic, so a single failure can be replaced inside the book.
    spare = []
    by = defaultdict(list)
    for r in rest:
        by[r["book_id"]].append(r)
    for b in sorted(by):
        pool = sorted(by[b], key=lambda r: r["prompt_id"])
        if pool:
            spare.append(pool[0])
    ctrl_job = primary + spare
    if args.smoke:
        treat_job = treat_rows[: args.smoke]
        ctrl_job = ctrl_job[: args.smoke]
        job = treat_job + ctrl_job
        rewrite_all(job, args.workers)
        print("smoke done", len(job))
        return

    rewrite_all(treat_rows + ctrl_job, args.workers)
    cache = load_cache()
    missing = [r for r in treat_rows if f"treatment:{r['prompt_id']}" not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} treatment rewrites missing from cache")
    # If a primary control row failed, the script would have exited. Assemble from primary.
    for r in primary:
        if f"control:{r['prompt_id']}" not in cache:
            raise SystemExit(f"control rewrite missing {r['prompt_id']}")
    assemble({"treatment": treat_rows, "control_pool": ctrl_rows}, cache, book_s)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(redact(str(e)), file=sys.stderr)
        raise
