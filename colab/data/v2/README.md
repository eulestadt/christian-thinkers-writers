# v2 training JSONL (additive)

`format_replay_{sft,dpo}.jsonl` — 300 format-following pairs (default on). Letter or 0/1 chosen; verbose-then-conclude rejected. Not from MMLU / ETHICS / TruthfulQA.

`treatment/` and `control/` receive the 3,500-row mix (elicited + OOD-train + UltraChat) plus format replay after `cai-pipeline-v2/scripts/06_mix_weighted.py` runs. Until then the v2 notebook warns if those files are missing. Do not overwrite v1 `data/treatment` or `data/control`.
