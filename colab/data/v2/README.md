# v2 training JSONL (additive)

`format_replay_{sft,dpo}.jsonl` — 300 format-following pairs (default on). Letter or 0/1 chosen; verbose-then-conclude rejected. Not from MMLU / ETHICS / TruthfulQA.

`treatment/` and `control/` hold the v2 mix after `cai-pipeline-v2/scripts/06_mix_weighted.py`:

- `sl_cai_train.jsonl` / `rlaif_dpo_train.jsonl` — 3,800 SFT rows (2,000 elicited + 1,000 OOD-train + 500 UltraChat + 300 format replay). DPO drops pairs where chosen equals rejected.
- `sl_cai_ood.jsonl` / `rlaif_ood.jsonl` — OOD-only sidecars (n=1000).
- `mix_summary.json` — counts and I/E-R vs uniform weights.

Treatment uses I/E-R example weights clipped to (0.25, 2.0). Control is uniform `weight=1.0`. Do not overwrite v1 `data/treatment` or `data/control`.
