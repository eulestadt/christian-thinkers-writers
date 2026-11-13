# v3 training JSONL

Current mix for [`cai_3way_experiment_and_eval_v3.ipynb`](../../cai_3way_experiment_and_eval_v3.ipynb). Built by [`scripts/build_v3_dialogues.py`](../../scripts/build_v3_dialogues.py) from the v2 `draft_0` / `revision_3` traces. The v1 notebook, the v1_fixed notebook, and the v2 notebook stay untouched. The v2 mix remains in [`../v2/`](../v2/).

v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 keeps the top 20 books by `ier_intrinsic` (no I−E filter) and rewrites each kept trace into one user message and one weighed assistant reply. UltraChat is not in this mix. The shared 300-row format replay is kept, at weight 1.

| Arm | CAI rows | Weight | Format replay | SFT / DPO rows |
|---|---:|---|---:|---:|
| `weighted/` | 1,001 | `w = s / mean(s)`, no clip, mean 1 | 300 | 1,301 |
| `uniform/` | 1,001, same row ids as weighted | every weight 1 | 300 | 1,301 |
| `control/` | 1,001, stratified by book | every weight 1 | 300 | 1,301 |

Kish ESS of the weighted CAI slice: **789.94**. Max weight: **2.248**. Counts are also in `mix_summary.json`. Per-row scores live in `cai_meta.jsonl` and are not columns on the train files.

Every arm is trained with R-DPO α=0.01 and β=0.1, seeds 0, 1, and 2. Rejected answers alternate between the unrevised draft and a dogmatic rewrite. The reported score is WiSE-FT λ=0.5. The notebook stops if these files are missing.
