#!/usr/bin/env python3
"""Run CAI Steps 0–4 in order. Each step is resumable via data/state.db."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_common import (  # noqa: E402
    DATA_DIR,
    MODEL_ID,
    State,
    is_peak_utc,
    log,
    next_offpeak_utc,
    next_peak_utc,
    resolve_api_key,
    write_coverage,
)

SCRIPTS = Path(__file__).resolve().parent
GEMINI_MARKERS = ("gemini", "google-genai", "gemini-2", "gemini-1.5", "gemini-2.5")


def run_mod(filename: str) -> None:
    log(f"===== {filename} =====")
    runpy.run_path(str(SCRIPTS / filename), run_name="__main__")


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _model_is_gemini(val: str) -> bool:
    low = (val or "").lower()
    return any(m in low for m in GEMINI_MARKERS)


def _scan_gemini(path: Path) -> list[str]:
    """Scan model-id fields only (not prompt text)."""
    hits = []
    if not path.exists():
        return hits
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if _model_is_gemini(str(obj.get("model") or "")):
            hits.append(f"{path.name}:model={obj.get('model')}")
        return hits
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            mid = str(rec.get("model") or "")
            if _model_is_gemini(mid):
                hits.append(f"{path.name}:line{i}:model={mid}")
                if len(hits) >= 20:
                    return hits
    return hits


def verify() -> dict:
    paths = {
        "constitution": DATA_DIR / "constitution.json",
        "prompts": DATA_DIR / "prompts_dataset.jsonl",
        "sl_cai": DATA_DIR / "sl_cai_train.jsonl",
        "rlaif": DATA_DIR / "rlaif_dpo_train.jsonl",
    }
    counts = {
        "prompts": _jsonl_count(paths["prompts"]),
        "sl_cai": _jsonl_count(paths["sl_cai"]),
        "rlaif": _jsonl_count(paths["rlaif"]),
    }
    constitution_n = 0
    constitution_model = None
    if paths["constitution"].exists():
        obj = json.loads(paths["constitution"].read_text(encoding="utf-8"))
        constitution_model = obj.get("model")
        constitution_n = len(obj.get("principles") or [])
    gemini_hits = []
    for p in paths.values():
        gemini_hits.extend(_scan_gemini(p))
    # also scan jsonl model fields
    model_ids = set()
    for key in ("prompts", "sl_cai", "rlaif"):
        p = paths[key]
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "model" in rec:
                    model_ids.add(str(rec["model"]))
    if constitution_model:
        model_ids.add(str(constitution_model))
    report = {
        "constitution_n": constitution_n,
        "constitution_model": constitution_model,
        "counts": counts,
        "model_ids": sorted(model_ids),
        "gemini_hits": gemini_hits,
    }
    log(
        "VERIFY constitution_n=%s prompts=%s sl_cai=%s rlaif=%s models=%s gemini_hits=%s"
        % (
            constitution_n,
            counts["prompts"],
            counts["sl_cai"],
            counts["rlaif"],
            sorted(model_ids),
            len(gemini_hits),
        )
    )
    return report


def main() -> None:
    resolve_api_key()
    State()  # init db
    peak = is_peak_utc()
    log(
        f"runner start model={MODEL_ID} peak_now={peak} "
        f"next_peak={next_peak_utc()} next_offpeak={next_offpeak_utc()}"
    )
    run_mod("00_select_and_fetch.py")
    run_mod("01_extract_constitution.py")
    run_mod("02_build_prompt_set.py")
    run_mod("03_stage1_sl_cai.py")
    run_mod("04_stage2_rlaif.py")
    state = State()
    report = verify()
    write_coverage(
        state,
        extra={
            "constitution_n": state.get_meta("constitution_n"),
            "prompts_total": state.get_meta("prompts_total"),
            "prompts_elicited": state.get_meta("prompts_elicited"),
            "prompts_ultrachat": state.get_meta("prompts_ultrachat"),
            "notes": [
                "Pipeline 0–4 finished (or resumed to completion). Step 5 not run.",
                f"Verify counts={report['counts']} models={report['model_ids']} gemini_hits={len(report['gemini_hits'])}",
            ],
        },
    )
    log("all steps 0-4 complete")
    if report["gemini_hits"]:
        raise SystemExit(f"gemini markers remain: {report['gemini_hits'][:8]}")
    if report["counts"]["prompts"] != 2500:
        raise SystemExit(f"prompts_dataset.jsonl has {report['counts']['prompts']} rows, expected 2500")


if __name__ == "__main__":
    main()
