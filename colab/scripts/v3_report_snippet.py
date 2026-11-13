
# Reported v3 result: mean of seeds 0, 1, 2 at WiSE-FT lambda 0.5.
# Each seed is listed beside the mean. The best seed is not the result.
# lambda 1 on arm A is a reference. lambda 0 is the base row already in SCORE_ARMS.

def _parse_slot(name: str):
    if name == "base":
        return "base", None, 0.0
    parts = str(name).split("_")
    if len(parts) == 3 and parts[1].startswith("s") and parts[2].startswith("l"):
        return parts[0], int(parts[1][1:]), float(parts[2][1:])
    return str(name), None, None


def _paired_bootstrap(diff, n_boot=10000, seed=0):
    diff = np.asarray(diff, dtype=float)
    if diff.size < 2:
        raise RuntimeError("paired bootstrap needs at least two items")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


if RUN_EVAL and item_rows:
    _assert_v3_write_paths()
    slot_frame = pd.DataFrame(benchmark_summary)
    slot_frame.to_csv(f"{RES_DIR}/objective_by_slot.csv")
    long_rows = []
    for slot, scores in benchmark_summary.items():
        arm, seed, lam = _parse_slot(slot)
        for bench, acc in scores.items():
            long_rows.append({
                "slot": slot, "arm": arm, "seed": seed, "lambda": lam,
                "benchmark": bench, "accuracy": acc,
            })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(f"{RES_DIR}/objective_by_seed.csv", index=False)

    reported = long_df[(long_df["arm"] == "base") | (np.isclose(long_df["lambda"], REPORTED_LAMBDA))]
    mean_rows = []
    benches = sorted(reported["benchmark"].unique())
    for bench in benches:
        rec = {"benchmark": bench}
        for arm in ("base", "weighted", "uniform", "control"):
            g = reported[(reported["benchmark"] == bench) & (reported["arm"] == arm)]
            rec[arm] = round(float(g["accuracy"].mean()), 2) if len(g) else None
        mean_rows.append(rec)
    df_obj = pd.DataFrame(mean_rows).set_index("benchmark")
    df_obj.to_csv(f"{RES_DIR}/objective_alignment_benchmarks.csv")
    df_obj.to_csv(f"{RES_DIR}/v3_reported_means_lambda0.5.csv")
    print("REPORTED RESULT: mean of seeds 0, 1, 2 at WiSE-FT lambda 0.5")
    print(df_obj)
    print("Each seed at lambda 0.5, plus base once:")
    show = reported.sort_values(["benchmark", "arm", "seed"])
    print(show.to_string(index=False))
    ref = long_df[(long_df["arm"] == "weighted") & (np.isclose(long_df["lambda"], 1.0))]
    print("Reference only, arm A at lambda 1 (not the reported result):")
    print(ref.sort_values(["benchmark", "seed"]).to_string(index=False))

    by_item = {}
    for rec in item_rows:
        arm, seed, lam = _parse_slot(rec["arm"])
        if arm != "base" and not (lam is not None and abs(lam - REPORTED_LAMBDA) < 1e-6):
            continue
        key = (rec["benchmark"], int(rec["item"]))
        by_item.setdefault(key, {}).setdefault(arm, []).append(int(rec["correct"]))
    per_item = {}
    for key, arms in by_item.items():
        if not {"weighted", "uniform", "control"} <= set(arms):
            continue
        per_item[key] = {arm: float(np.mean(vals)) for arm, vals in arms.items()}

    intervals = []
    benches = sorted({k[0] for k in per_item}) + ["pooled_primary"]
    for bench in benches:
        keys = [k for k in sorted(per_item) if bench == "pooled_primary" or k[0] == bench]
        if len(keys) < 2:
            continue
        a = np.array([per_item[k]["weighted"] for k in keys])
        b = np.array([per_item[k]["uniform"] for k in keys])
        c = np.array([per_item[k]["control"] for k in keys])
        for label, diff in (("A-B", a - b), ("A-C", a - c)):
            mean, lo, hi = _paired_bootstrap(diff, n_boot=10000, seed=0)
            intervals.append({
                "contrast": label,
                "benchmark": bench,
                "n_items": len(keys),
                "mean_pp": round(100 * mean, 3),
                "ci95_low_pp": round(100 * lo, 3),
                "ci95_high_pp": round(100 * hi, 3),
                "note": "Item means across seeds 0, 1, 2 at lambda 0.5, then a paired bootstrap over items.",
            })
    (Path(RES_DIR) / "paired_bootstrap.json").write_text(
        json.dumps(intervals, indent=2) + "\n", encoding="utf-8"
    )
    print("95% paired bootstrap on Arm A minus Arm B and Arm A minus Arm C (percentage points):")
    print(pd.DataFrame(intervals).to_string(index=False))
elif RUN_EVAL:
    print("RUN_EVAL finished with no item_rows; paired bootstrap was not computed.")
else:
    print("RUN_EVAL=False — reported means and paired bootstrap wait for a GPU eval.")
