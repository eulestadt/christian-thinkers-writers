# v3 training JSONL

Current mix for [`cai_3way_experiment_and_eval_v3.ipynb`](../../cai_3way_experiment_and_eval_v3.ipynb). Built by [`scripts/build_v3_dialogues.py`](../../scripts/build_v3_dialogues.py) from the v2 `draft_0` / `revision_3` traces. The v2 mix remains in [`../v2/`](../v2/).

v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas, clips weights to 0.25–2, and draws rows with replacement. v3 keeps the top 20 books by `ier_intrinsic` because LIMA and AlpaGasus found a small high-scoring slice beat the full set, and rewrites each kept trace into one deliberative user message and one weighed assistant reply. The mix is those CAI rows plus a shared 300-row format replay at weight 1, so the letter-format scorer still sees a letter.

| Arm | CAI rows | Weight | Format replay | SFT / DPO rows |
|---|---:|---|---:|---:|
| `weighted/` | 1,001 | `w = s / mean(s)`, no clip, mean 1 | 300 | 1,301 |
| `uniform/` | 1,001, same row ids as weighted | every weight 1 | 300 | 1,301 |
| `control/` | 1,001, stratified by book | every weight 1 | 300 | 1,301 |

Kish ESS of the weighted CAI slice: **789.94**, logged because uncapped weights raise variance. Max weight: **2.248**. Counts are also in `mix_summary.json`. Per-row scores live in `cai_meta.jsonl` as trainer metadata.

Every arm is trained with R-DPO α=0.01 and β=0.1 (Park et al.), on seeds 0, 1, and 2 (Dodge et al.). Rejected answers alternate between the unrevised draft and a dogmatic rewrite (Sharma et al.). The reported score, when Colab writes it, is WiSE-FT λ=0.5 (Wortsman et al.). The notebook stops if these files are missing. Why: [`../../../docs/study/v3-methods.md`](../../../docs/study/v3-methods.md).
