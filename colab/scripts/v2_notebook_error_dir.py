# Cell 13 — Error-direction on ETHICS and MMLU moral_* (wrong answers only).
# Keys unchanged. moral_scenarios is reported apart from philosophy / world_religions.
# Uses generative parsed labels when present (primary scorer); also logprob side labels.

RUN_ERROR_DIRECTION = globals().get("RUN_ERROR_DIRECTION", True)
KNOWLEDGE_SUBSETS = {"MMLU_philosophy", "MMLU_world_religions"}
MORAL_MMLU = {"MMLU_moral_scenarios", "MMLU_moral_disputes"}
ETHICS_BENCH = {"ETHICS_commonsense", "ETHICS_deontology", "ETHICS_justice", "ETHICS_virtue"}


def _gen_pred_for_error(r):
    if r.get("parse_fail"):
        return None, True
    if r.get("parsed") is not None:
        return int(r["parsed"]), False
    if r.get("parsed_gen") is not None:
        return int(r["parsed_gen"]), bool(r.get("parse_fail"))
    return None, True


def _directions_from_generative():
    out = []
    for r in item_rows or []:
        bench = r.get("benchmark")
        if bench in ETHICS_BENCH:
            pred, fail = _gen_pred_for_error(r)
            gold = r.get("gold")
            if gold is None:
                # v2 Cell 4 stores gold when patched; otherwise skip direction (need gold)
                continue
            d = ethics_error_direction(pred, int(gold), fail)
            if d:
                out.append({"arm": r["arm"], "benchmark": bench, "item": r["item"],
                            "direction": d, "scorer": "generative", "parse_fail": int(fail)})
        elif bench in MORAL_MMLU:
            pred, fail = _gen_pred_for_error(r)
            if fail or pred is None or r.get("correct"):
                if fail:
                    continue
                continue
            ch = r.get("choices") or []
            gold = r.get("gold")
            if gold is None or not ch:
                continue
            d = mmlu_error_direction(pred, int(gold), ch)
            if d:
                out.append({"arm": r["arm"], "benchmark": bench, "item": r["item"],
                            "direction": d, "scorer": "generative", "parse_fail": 0})
    return out


def _directions_from_logprob(path: Path):
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        bench = r.get("benchmark")
        gold = r.get("gold")
        pred = r.get("pred_logprob")
        if gold is None or pred is None:
            continue
        if int(r.get("correct_logprob") or 0) == 1:
            continue
        if bench in ETHICS_BENCH:
            d = ethics_error_direction(int(pred), int(gold), False)
        elif bench in MORAL_MMLU:
            d = mmlu_error_direction(int(pred), int(gold), r.get("choices") or [])
        else:
            continue
        if d:
            out.append({"arm": r["arm"], "benchmark": bench, "item": r["item"],
                        "direction": d, "scorer": "logprob", "parse_fail": 0})
    return out


if RUN_ERROR_DIRECTION:
    Path(RES_DIR).mkdir(parents=True, exist_ok=True)
    gen_dir = _directions_from_generative()
    lp_path = Path(RES_DIR) / "logprob" / "items.jsonl"
    lp_dir = _directions_from_logprob(lp_path)

    frames = []
    if gen_dir:
        frames.append(pd.DataFrame(summarize_directions(gen_dir)).assign(scorer="generative"))
    if lp_dir:
        frames.append(pd.DataFrame(summarize_directions(lp_dir)).assign(scorer="logprob"))
    if not frames:
        print(
            "Error-direction: need Cell 4 gold/choices/parsed fields and/or logprob/items.jsonl. "
            "Knowledge subsets (philosophy, world_religions) are not mixed into this table."
        )
    else:
        df_ed = pd.concat(frames, ignore_index=True)
        # Split moral_scenarios from other moral_* and from knowledge (knowledge omitted).
        df_ed["family"] = df_ed["benchmark"].map(lambda b: (
            "moral_scenarios" if b == "MMLU_moral_scenarios"
            else "moral_disputes" if b == "MMLU_moral_disputes"
            else "ETHICS" if str(b).startswith("ETHICS_")
            else "other"
        ))
        out_p = Path(RES_DIR) / "error_direction.csv"
        df_ed.to_csv(out_p, index=False)
        print("Error-direction (wrong answers only; keys unchanged)")
        print("ETHICS: 1=unacceptable, 0=acceptable. stricter = pred 1 / gold 0. looser = pred 0 / gold 1.")
        print("MMLU moral_scenarios: stricter = more 'Wrong' labels than the key. mixed = same count, swapped pair.")
        print("moral_disputes is mostly knowledge; unclassified means the choice text is not Wrong/Not-wrong.")
        print("philosophy / world_religions are knowledge subsets and are omitted from this table.")
        print(df_ed.to_string(index=False))
        print("Wrote", out_p)
        for fam in ("ETHICS", "moral_scenarios", "moral_disputes"):
            sub = df_ed[df_ed["family"] == fam]
            if len(sub):
                print(f"--- {fam} ---")
                print(sub.to_string(index=False))
else:
    print("RUN_ERROR_DIRECTION=False")
