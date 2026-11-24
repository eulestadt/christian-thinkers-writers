# v3 3B study run log

Session name: `ctw-v3`. Account: `eulestadt@gmail.com`. Notebook: `colab/cai_3way_experiment_and_eval_v3.ipynb` on branch `cursor/v3-high-intrinsic-dialogue-dd22`. Compute cap: ~240 CU from starting balance. Hardware preference: L4, then T4, then A100. H100 is forbidden.

All timestamps are recorded as UTC and US/Eastern.

## 2026-11-24 01:24:24 UTC / 2026-11-23 20:24:24 US/Eastern — start

- `colab usage` starting balance: **506.04** compute units
- Usage rate: 0.00/hr
- Active assignments: 0
- Soft stop if remaining would fall below **266.04** (506.04 − 240)
- Working directory: `/tmp/ctw-v3-study`
- Isolated Colab CLI config: `/tmp/ctw-v3-study/colab-sessions.json`
- Clone: `https://github.com/eulestadt/christian-thinkers-writers` @ `cursor/v3-high-intrinsic-dialogue-dd22` (`00c3b54 Document v3 as the current Colab run.`)
- `colab/data/v3` present: weighted / uniform / control JSONL, format replay, `mix_summary.json` (CAI n=1001, Kish ESS=789.94, max weight=2.248)
- Artifacts destination: `/cursor/stores/bc-08e136d5-97fc-446b-aba1-a19e05e70ce1/docs/study/artifacts/`

## 2026-11-24 01:25:31 UTC / 2026-11-23 20:25:31 US/Eastern — provision

Requesting one session: `colab new -s ctw-v3 --gpu L4`.

## 2026-11-24 01:25:37 UTC / 2026-11-23 20:25:37 US/Eastern — session READY

- Session: `ctw-v3` id `gpu-l4-s-ass1a1-wpezaz2um9st`
- Hardware: **NVIDIA L4** Standard GPU, 23034 MiB VRAM, CUDA 13.0, driver 580.82.07
- Host: 12 vCPU, 52 GiB RAM, 236G overlay
- Torch: 2.14.1+cu130
- `colab usage`: balance **506.04**, rate **1.94/hr**, active assignments 1
- Projected 240 CU ≈ 123 hours at this rate. Nine 3B QLoRA trains should fit.

## 2026-11-24 01:26:29 UTC / 2026-11-23 20:26:29 US/Eastern — clone on VM

- `git clone --depth 1 --branch cursor/v3-high-intrinsic-dialogue-dd22` → `/content/christian-thinkers-writers`
- HEAD `00c3b54 Document v3 as the current Colab run.`
- `colab/data/v3` present (weighted / uniform / control JSONL). Would have failed loudly if missing.
- Seeded `/content/drive/MyDrive/CAI_Experiment` as a local directory (no Drive mount; notebook BASE_DIR still lands there).
- Notebook uploaded to `/content/cai_3way_experiment_and_eval_v3.ipynb`

## 2026-11-24 01:26:49 UTC / 2026-11-23 20:26:49 US/Eastern — install

- Cell 3 (Unsloth + TRL) finished at 01:27:36 UTC. Unsloth 2026.11.10, FastLanguageModel ok.
- xformers does not match torch 2.14.1+cu130; Unsloth falls back to PyTorch attention. Training continues.

## 2026-11-24 01:28:20 UTC / 2026-11-23 20:28:20 US/Eastern — environment cell

- Cell 4 verified v3 mixes: 1301 SFT / 1301 DPO per arm (1001 CAI + 300 replay). Kish ESS 789.94, max weight 2.2475.
- Write paths: `adapters/llama32_3b/v3` and `results/llama32_3b/v3` under `/content/drive/MyDrive/CAI_Experiment`
- CUDA=True GPU=NVIDIA L4. Balance after cell 4: **505.94** CU (~0.10 spent).
- Cell 6 cached `docs/intrinsic-motivation-scores.sqlite` (527 works, 99637 chunks) and treatment `state.db` read-only on local NVMe.

## 2026-11-24 01:31:36 UTC / 2026-11-23 20:31:36 US/Eastern — train functions loaded

- Cell 8 loaded with `RUN_TRAINING=False` so the nine trains can be launched one arm×seed at a time, with a download after each.
- Next: weighted seed 0.

## 2026-11-24 01:31:59 UTC / 2026-11-23 20:31:59 US/Eastern — TRAIN weighted seed 0 start

First QLoRA train. Base `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` downloaded on this VM.

## 2026-11-24 01:48:12 UTC / 2026-11-23 20:48:12 US/Eastern — TRAIN weighted seed 0 done

- Elapsed: **971.5 s** (~16.2 min). SFT 82 steps then R-DPO 82 steps (α=0.01, β=0.1).
- Adapter: `/content/drive/MyDrive/CAI_Experiment/adapters/llama32_3b/v3/weighted_seed0_final_lora`
- Downloaded to `/cursor/stores/bc-08e136d5-97fc-446b-aba1-a19e05e70ce1/docs/study/artifacts/adapters/llama32_3b/v3/weighted_seed0_final_lora` (93M safetensors) plus `weighted_seed0_trainer_state.json`
- Train transcript: `/cursor/stores/bc-08e136d5-97fc-446b-aba1-a19e05e70ce1/docs/study/artifacts/logs/train_weighted_seed0.colab.txt`
- `colab usage`: balance **505.46**, rate 1.94/hr. Cumulative spent from 506.04: **0.58 CU**


## 2026-11-24 01:49:35 UTC / 2026-11-23 20:49:35 US/Eastern — TRAIN uniform seed 0 start

```
Current balance: 505.29 compute units
Usage rate: 1.94/hr
Active assignments: 1
```

## 2026-11-24 02:06:15 UTC / 2026-11-23 21:06:15 US/Eastern — TRAIN uniform seed 0 done

- Adapter downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/uniform_seed0_final_lora`
- Cumulative CU spent from start: **1.23** (balance 504.81)

```
Current balance: 504.81 compute units
Usage rate: 1.94/hr
Active assignments: 1
```

## 2026-11-24 02:06:23 UTC / 2026-11-23 21:06:23 US/Eastern — TRAIN control seed 0 start

```
Current balance: 504.81 compute units
Usage rate: 1.94/hr
Active assignments: 1
```
