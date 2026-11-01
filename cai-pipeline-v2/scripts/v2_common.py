#!/usr/bin/env python3
"""v2 paths, frozen constitutions, I/E-R metadata (never inserted into prompts)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

def _first_existing(*cands: Path) -> Path:
    for p in cands:
        if p.exists():
            return p
    return cands[0]


_STORE = Path("/cursor/stores/self")
_HERE = Path(__file__).resolve().parent
_REPO_GUESS = _HERE.parents[1] if _HERE.parent.name == "cai-pipeline-v2" else Path("/workspace")

V2_ROOT = _first_existing(
    Path(os.environ["CAI_V2_ROOT"]) if os.environ.get("CAI_V2_ROOT") else Path("/nonexistent"),
    _STORE / "cai-pipeline-v2",
    _REPO_GUESS / "cai-pipeline-v2",
)
V1_SCRIPTS = _first_existing(_STORE / "cai-pipeline" / "scripts", _REPO_GUESS / "cai-pipeline" / "scripts")
V1_DATA = _first_existing(_STORE / "cai-pipeline" / "data", _REPO_GUESS / "cai-pipeline" / "data")
V1_CONTROL_DATA = _first_existing(
    _STORE / "cai-pipeline-control" / "data",
    _REPO_GUESS / "cai-pipeline-control" / "data",
)
SCORES_DB = _first_existing(
    _STORE / "docs" / "intrinsic-motivation-scores.sqlite",
    _REPO_GUESS / "docs" / "intrinsic-motivation-scores.sqlite",
)
REPO = _first_existing(_REPO_GUESS, Path("/workspace"))
EVAL_DIR = _first_existing(REPO / "colab" / "data" / "eval", V2_ROOT / "data" / "eval")

sys.path.insert(0, str(V1_SCRIPTS))

LEAKAGE_TERMS = (
    "vanderweele",
    "vander weele",
    "flourishing",
    "fai benchmark",
    "fai taxonomy",
    "human flourishing",
    "global flourishing",
)

DOMAINS = ("workplace", "civic", "interpersonal", "medical-adjacent", "online")
ALPHA_DEFAULT = 1.0
WEIGHT_CLIP = (0.25, 2.0)  # cap so top books cannot dominate gradients
BOOK_FLOOR = 0.025  # uniform mix when tau is set; 1/40
NEAR_DUP = 0.85
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K_PRINCIPLES = 3
MIX = {"elicited": 2000, "ood_train": 1000, "ultrachat": 500}  # total 3500
FORMAT_REPLAY_N = 300  # extra SFT/DPO slice, default on; not from MMLU/ETHICS/TQA


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower().strip()
    return re.sub(r"\s+", " ", t)


def assert_no_leakage(text: str, where: str) -> None:
    low = (text or "").lower()
    for term in LEAKAGE_TERMS:
        if term in low:
            raise RuntimeError(f"leakage term {term!r} in {where}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_constitution(arm: str) -> list[dict]:
    if arm == "control":
        cands = [
            V2_ROOT / "data" / "control" / "constitution.json",
            V1_CONTROL_DATA / "constitution.json",
        ]
    else:
        cands = [
            V2_ROOT / "data" / "treatment" / "constitution.json",
            V1_DATA / "constitution.json",
        ]
    path = next((p for p in cands if p.is_file()), cands[0])
    data = json.loads(path.read_text(encoding="utf-8"))
    pr = data.get("principles") or []
    if not pr:
        raise SystemExit(f"empty constitution {path}")
    for i, p in enumerate(pr):
        blob = " ".join(
            str(p.get(k) or "")
            for k in ("source_thesis", "critique_request", "revision_request")
        )
        assert_no_leakage(blob, f"{path}#{i}")
        p["_principle_id"] = i
    return pr


def load_ier_by_writer() -> dict[str, float]:
    """I/E-R intrinsic from the scoring sqlite. No I−E filter. Metadata only."""
    if not SCORES_DB.is_file() or SCORES_DB.stat().st_size < 1000:
        # fall back to treatment corpus_manifest
        man = json.loads((V1_DATA / "corpus_manifest.json").read_text(encoding="utf-8"))
        return {str(r["writer_id"]): float(r["ier_intrinsic"]) for r in man}
    uri = f"file:{SCORES_DB.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    rows = con.execute(
        """
        SELECT writer_id, score FROM work_subscores
        WHERE subscale = 'ier_intrinsic' AND score IS NOT NULL
        """
    ).fetchall()
    con.close()
    out: dict[str, list[float]] = {}
    for wid, score in rows:
        out.setdefault(str(wid), []).append(float(score))
    return {k: sum(v) / len(v) for k, v in out.items()}


def principle_s(p: dict, ier: dict[str, float], default: float) -> float:
    books = [str(b) for b in (p.get("source_books") or [])]
    scores = [ier[b] for b in books if b in ier]
    if not scores:
        return default
    return float(sum(scores) / len(scores))


def attach_s(principles: list[dict], ier: dict[str, float]) -> float:
    known = [v for v in ier.values() if v is not None]
    mean_s = float(sum(known) / len(known)) if known else 1.0
    for p in principles:
        p["_s_i"] = principle_s(p, ier, mean_s)
    return mean_s


def clip_weight(w: float, lo: float = WEIGHT_CLIP[0], hi: float = WEIGHT_CLIP[1]) -> float:
    return float(min(hi, max(lo, w)))


def sample_principles(
    prompt: str,
    principles: list[dict],
    emb,
    q_emb,
    rng,
    k: int = K_PRINCIPLES,
    alpha: float = ALPHA_DEFAULT,
    use_ier: bool = True,
) -> list[dict]:
    """p ∝ max(sim, 0) * s_i^alpha (treatment) or max(sim, 0) (control/uniform)."""
    import numpy as np

    sims = emb @ q_emb
    sims = np.maximum(sims, 0.0)
    if use_ier:
        s = np.array([float(p.get("_s_i") or 0.0) for p in principles], dtype=float)
        s = np.maximum(s, 0.0)
        mass = sims * np.power(s, float(alpha))
    else:
        mass = sims
    total = float(mass.sum())
    if total <= 0:
        mass = np.ones(len(principles), dtype=float)
        total = float(mass.sum())
    p = mass / total
    k = min(int(k), len(principles))
    idx = rng.choice(len(principles), size=k, replace=False, p=p)
    return [principles[int(i)] for i in idx]


def slim_principle(p: dict) -> dict:
    """Prompt-facing constitution pair only. No scores."""
    return {
        "source_thesis": p.get("source_thesis") or "",
        "critique_request": p.get("critique_request") or "",
        "revision_request": p.get("revision_request") or "",
        "principle_id": int(p.get("_principle_id", -1)),
        "source_books": list(p.get("source_books") or []),
    }


def example_weight(ps: list[dict], mean_s: float) -> dict:
    s_vals = [float(p.get("_s_i") or 0.0) for p in ps]
    s_bar = float(sum(s_vals) / len(s_vals)) if s_vals else mean_s
    w = clip_weight(s_bar / mean_s if mean_s else 1.0)
    return {
        "s_mean_principles": s_bar,
        "s_corpus_mean": mean_s,
        "weight": w,
        "book_ids": [b for p in ps for b in (p.get("source_books") or [])],
        "principle_ids": [int(p.get("_principle_id", -1)) for p in ps],
    }
