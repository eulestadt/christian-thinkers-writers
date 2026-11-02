#!/usr/bin/env python3
"""Mix v1 elicited+UltraChat replay with OOD-train. Attach IER weights (treatment) or uniform (control)."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (  # noqa: E402
    MIX,
    REPO,
    V1_CONTROL_DATA,
    V1_DATA,
    V2_ROOT,
    WEIGHT_CLIP,
    attach_s,
    clip_weight,
    load_constitution,
    load_ier_by_writer,
    load_jsonl,
    norm,
    write_jsonl,
)


def thesis_index(principles: list[dict]) -> dict[str, dict]:
    out = {}
    for p in principles:
        t = (p.get("source_thesis") or "").strip()
        if t and t not in out:
            out[t] = p
    return out


def weight_from_pjson(p1, p2, p3, index, mean_s, use_ier: bool) -> dict:
    ps = []
    for raw in (p1, p2, p3):
        try:
            d = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            d = {}
        t = (d.get("source_thesis") or "").strip()
        p = index.get(t)
        if p is not None:
            ps.append(p)
    if not use_ier:
        return {"weight": 1.0, "principle_ids": [int(p.get("_principle_id", -1)) for p in ps], "book_ids": []}
    if not ps:
        return {"weight": 1.0, "principle_ids": [], "book_ids": []}
    s_vals = [float(p.get("_s_i") or 0.0) for p in ps]
    s_bar = float(sum(s_vals) / len(s_vals))
    return {
        "weight": clip_weight(s_bar / mean_s if mean_s else 1.0),
        "s_mean_principles": s_bar,
        "principle_ids": [int(p.get("_principle_id", -1)) for p in ps],
        "book_ids": [b for p in ps for b in (p.get("source_books") or [])],
    }


def v1_state_pairs(state_db: Path) -> dict[str, dict]:
    if not state_db.is_file():
        return {}
    con = sqlite3.connect(f"file:{state_db.as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        """
        SELECT prompt, source, p1_json, p2_json, p3_json, draft_0, revision_3
        FROM sl_cai WHERE status='ok'
        """
    ).fetchall()
    dpo = {}
    try:
        for prompt, chosen, rejected in con.execute(
            "SELECT s.prompt, r.chosen, r.rejected FROM rlaif r JOIN sl_cai s ON s.prompt_id=r.prompt_id WHERE r.status='ok'"
        ):
            dpo[norm(prompt)] = {"chosen": chosen, "rejected": rejected}
    except sqlite3.Error:
        pass
    con.close()
    out = {}
    for prompt, source, p1, p2, p3, d0, r3 in rows:
        out[norm(prompt)] = {
            "prompt": prompt, "source": source, "p1_json": p1, "p2_json": p2, "p3_json": p3,
            "draft_0": d0, "revision_3": r3, "dpo": dpo.get(norm(prompt)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["treatment", "control"], required=True)
    args = ap.parse_args()
    use_ier = args.arm == "treatment"
    v1_data = V1_DATA if args.arm == "treatment" else V1_CONTROL_DATA
    v1_state = v1_data / "state.db"
    principles = load_constitution(args.arm)
    ier = load_ier_by_writer() if use_ier else {}
    mean_s = attach_s(principles, ier) if use_ier else 1.0
    index = thesis_index(principles)
    meta = v1_state_pairs(v1_state)

    prompts = load_jsonl(v1_data / "prompts_dataset.jsonl")
    sft_v1 = load_jsonl(v1_data / "sl_cai_train.jsonl")
    dpo_v1 = load_jsonl(v1_data / "rlaif_dpo_train.jsonl")
    sft_by = {norm(r["prompt"]): r for r in sft_v1}
    dpo_by = {norm(r["prompt"]): r for r in dpo_v1}

    sft_out, dpo_out = [], []
    n_el = n_uc = 0
    for rec in prompts:
        src = rec.get("source")
        if src not in ("elicited", "ultrachat"):
            continue
        if src == "elicited" and n_el >= MIX["elicited"]:
            continue
        if src == "ultrachat" and n_uc >= MIX["ultrachat"]:
            continue
        p = rec["prompt"]
        k = norm(p)
        sft = sft_by.get(k)
        if not sft:
            continue
        st = meta.get(k) or {}
        w = weight_from_pjson(st.get("p1_json"), st.get("p2_json"), st.get("p3_json"), index, mean_s, use_ier)
        row = {
            "prompt": p,
            "completion": sft.get("completion"),
            "model": sft.get("model"),
            "source": src,
            "book_id": rec.get("book_id"),
            "weight": 1.0 if not use_ier else w["weight"],
            "principle_ids": w.get("principle_ids"),
            "book_ids": w.get("book_ids"),
        }
        sft_out.append(row)
        dpo = dpo_by.get(k)
        if dpo and (dpo.get("chosen") or "").strip() != (dpo.get("rejected") or "").strip():
            dpo_out.append({
                "prompt": p, "chosen": dpo.get("chosen"), "rejected": dpo.get("rejected"),
                "model": dpo.get("model"), "source": src, "book_id": rec.get("book_id"),
                "weight": row["weight"], "principle_ids": row["principle_ids"], "book_ids": row["book_ids"],
            })
        if src == "elicited":
            n_el += 1
        else:
            n_uc += 1

    ood_sft = load_jsonl(V2_ROOT / "data" / args.arm / "sl_cai_ood.jsonl")
    ood_dpo = load_jsonl(V2_ROOT / "data" / args.arm / "rlaif_ood.jsonl")
    if not use_ier:
        for r in ood_sft:
            r["weight"] = 1.0
        for r in ood_dpo:
            r["weight"] = 1.0
    sft_out.extend(ood_sft[: MIX["ood_train"]])
    dpo_out.extend(
        r for r in ood_dpo[: MIX["ood_train"]]
        if (r.get("chosen") or "").strip() != (r.get("rejected") or "").strip()
    )

    fr_sft = load_jsonl(REPO / "colab" / "data" / "v2" / "format_replay_sft.jsonl")
    fr_dpo = load_jsonl(REPO / "colab" / "data" / "v2" / "format_replay_dpo.jsonl")
    for r in fr_sft:
        r["weight"] = 1.0
        r["source"] = "format_replay"
    for r in fr_dpo:
        r["weight"] = 1.0
        r["source"] = "format_replay"
    sft_out.extend(fr_sft)
    dpo_out.extend(
        r for r in fr_dpo
        if (r.get("chosen") or "").strip() != (r.get("rejected") or "").strip()
    )

    dest = V2_ROOT / "data" / args.arm
    repo_dest = REPO / "colab" / "data" / "v2" / args.arm
    write_jsonl(dest / "sl_cai_train.jsonl", sft_out)
    write_jsonl(dest / "rlaif_dpo_train.jsonl", dpo_out)
    write_jsonl(repo_dest / "sl_cai_train.jsonl", sft_out)
    write_jsonl(repo_dest / "rlaif_dpo_train.jsonl", dpo_out)
    summary = {
        "arm": args.arm,
        "use_ier_weights": use_ier,
        "n_sft": len(sft_out),
        "n_dpo": len(dpo_out),
        "by_source_sft": {
            s: sum(1 for r in sft_out if r.get("source") == s)
            for s in ("elicited", "ultrachat", "ood", "format_replay")
        },
        "by_source_dpo": {
            s: sum(1 for r in dpo_out if r.get("source") == s)
            for s in ("elicited", "ultrachat", "ood", "format_replay")
        },
        "weight_clip": list(WEIGHT_CLIP),
        "mean_weight": round(sum(float(r.get("weight") or 1) for r in sft_out) / max(len(sft_out), 1), 4),
        "mix": MIX,
        "format_replay_n": len(fr_sft),
        "ablation": "v2 treatment + uniform weights: train the same sl_cai_train.jsonl with USE_IER_WEIGHTS=False (no third LoRA in this PR).",
    }
    (dest / "mix_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (repo_dest / "mix_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
