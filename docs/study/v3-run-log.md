# v3 run log

Living step log for the Llama 3.2 3B v3 Colab session. Times are America/New_York (ET), with UTC in parentheses where a Colab stamp exists.

**Status: mix is on GitHub. Colab provision has started. Nine QLoRA trains are not done.**

## Session

| | |
|---|---|
| Session name | `ctw-v3` |
| Account | `eulestadt@gmail.com` |
| GPU | pending (requested L4; T4 if L4 is unavailable) |
| Notebook | `colab/cai_3way_experiment_and_eval_v3.ipynb` |
| Branch | `cursor/v3-high-intrinsic-dialogue-dd22` |
| Writes | `adapters/{ACTIVE_MODEL}/v3/`, `results/{ACTIVE_MODEL}/v3/` only |
| Compute cap | about 240 Colab compute units from the starting balance |
| Starting CU | **506.04** (2026-11-23 20:24 ET / 2026-11-24 01:24 UTC) |
| Soft stop | remaining would fall below **266.04** |
| Remaining CU | pending (re-check after attach) |

## Mix already on GitHub (not a train)

| | |
|---|---|
| CAI rows per arm | 1,001 |
| Format replay | 300 |
| SFT / DPO rows | 1,301 / 1,301 |
| Kish ESS (weighted CAI slice) | 789.94 |
| Max weight | 2.248 |
| Min weight | 0.571 |
| R-DPO | α=0.01, β=0.1 |
| Seeds | 0, 1, 2 |
| Reported eval | WiSE-FT λ=0.5 |

Source: [`colab/data/v3/mix_summary.json`](../../colab/data/v3/mix_summary.json).

## Steps

| # | Step | Status | Time (ET) | Notes |
|---|---|---|---|---|
| 0 | v3 mix + unexecuted notebook on GitHub | done | 2026-11-08 | PR #3; Kish ESS 789.94, max weight 2.248 |
| 1 | Study folder stub (`docs/study/`) | done | 2026-11-23 | methods, pipeline coverage, email skeleton; results pending |
| 2 | `colab usage` starting balance | done | 2026-11-23 20:24 | **506.04** CU; rate 0.00/hr; 0 active assignments |
| 3 | Provision Colab (`ctw-v3`, L4 requested) | in progress | 2026-11-23 20:25 | `colab new -s ctw-v3 --gpu L4` |
| 4 | Clone repo / copy `colab/data/v3` onto the runtime | pending | | Cell 1 must see the mix |
| 5 | Train `weighted` seed 0 | pending | | adapter `weighted_seed0_final_lora` |
| 6 | Train `weighted` seed 1 | pending | | |
| 7 | Train `weighted` seed 2 | pending | | |
| 8 | Train `uniform` seed 0 | pending | | same shuffle as weighted at this seed |
| 9 | Train `uniform` seed 1 | pending | | |
| 10 | Train `uniform` seed 2 | pending | | |
| 11 | Train `control` seed 0 | pending | | |
| 12 | Train `control` seed 1 | pending | | |
| 13 | Train `control` seed 2 | pending | | |
| 14 | Eval λ=0.5, all arms × seeds | pending | | reported result |
| 15 | Eval λ=1 Arm A (three seeds) | pending | | unblended reference |
| 16 | Eval λ=0 base (once) | pending | | no fine-tune |
| 17 | Style / dogmatic-vs-reflective (50 items, λ=0.5) | pending | | |
| 18 | Download adapters, results, executed notebook | pending | | copy into this folder and `colab/results/` |
| 19 | Rewrite [`email-vanderweele.md`](email-vanderweele.md) from measured numbers | pending | | do not send the skeleton |

Stop if remaining units would drop more than about 240 from the starting balance. Do not overwrite v1, v1_fixed, or v2 adapters.

## Hardware / CU notes

```
2026-11-23 20:24 ET — starting_cu: 506.04
2026-11-23 20:25 ET — requested GPU: L4 (session ctw-v3)
runtime: pending attach
gpu: pending
cuda: pending
cu_after_each_train: pending
```
