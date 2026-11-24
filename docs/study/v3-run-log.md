# v3 Colab run log

This is the living log of the Llama 3.2 3B v3 session. Times are America/New_York, with the original UTC stamp in parentheses. The session name is `ctw-v3`, on account `eulestadt@gmail.com`. The notebook is `colab/cai_3way_experiment_and_eval_v3.ipynb` on branch `cursor/v3-high-intrinsic-dialogue-dd22`. Writes go under `adapters/{ACTIVE_MODEL}/v3/` and `results/{ACTIVE_MODEL}/v3/` so earlier runs keep their folders. The compute cap is about 240 Colab units from the starting balance. Hardware preference is L4, then T4, then A100.

The mix this session trains is already on GitHub: 1,001 CAI rows per arm plus 300 format-replay rows, Kish ESS 789.94, max weight 2.248, R-DPO α=0.01 and β=0.1, reported evaluation at WiSE-FT λ=0.5. Counts are in [`colab/data/v3/mix_summary.json`](../../colab/data/v3/mix_summary.json). The mix is frozen on disk because Cell 1 reads those JSONL files rather than rebuilding the slice. Why those choices are the default is [`v3-methods.md`](v3-methods.md).

All nine QLoRA trains finished, then primary eval and the fifty-item style set. The session stopped at 1:40 a.m. Eastern on 24 November 2026. **10.12 compute units** were spent (506.04 − 495.92). Measured scores are in [`results.md`](results.md).

## 23 November 2026, 8:24 p.m. Eastern (2026-11-24 01:24:24 UTC) — start

`colab usage` showed a starting balance of 506.04 compute units, a usage rate of 0.00 per hour, and no active assignments. The soft stop is a remaining balance of 266.04, which is 506.04 minus 240. The working directory on the coordinator was `/tmp/ctw-v3-study`, with an isolated Colab CLI config at `/tmp/ctw-v3-study/colab-sessions.json`. The clone target was `https://github.com/eulestadt/christian-thinkers-writers` at `cursor/v3-high-intrinsic-dialogue-dd22`, then at commit `00c3b54` (“Document v3 as the current Colab run.”). `colab/data/v3` was present: weighted, uniform, and control JSONL, format replay, and `mix_summary.json` (CAI n=1001, Kish ESS=789.94, max weight=2.248). Artifacts are downloaded to the private study store under `docs/study/artifacts/`.

## 23 November 2026, 8:25 p.m. Eastern (01:25:31 UTC) — provision

One session was requested with `colab new -s ctw-v3 --gpu L4`.

## 23 November 2026, 8:25 p.m. Eastern (01:25:37 UTC) — session ready

The session came up as `ctw-v3`, id `gpu-l4-s-ass1a1-wpezaz2um9st`, on an NVIDIA L4 Standard GPU with 23034 MiB of VRAM, CUDA 13.0, driver 580.82.07. The host had 12 vCPUs, 52 GiB of RAM, and a 236G overlay. Torch was 2.14.1+cu130. After attach, `colab usage` still read 506.04, now at 1.94 units per hour with one active assignment. At that rate, 240 units is about 123 hours.

## 23 November 2026, 8:26 p.m. Eastern (01:26:29 UTC) — clone on the VM

The VM ran `git clone --depth 1 --branch cursor/v3-high-intrinsic-dialogue-dd22` into `/content/christian-thinkers-writers`. HEAD was still `00c3b54`. `colab/data/v3` was present (weighted, uniform, and control JSONL); Cell 1 would have stopped if it had been missing, because the trainer reads those JSONL files rather than rebuilding the mix. Rebuilding on the VM would also have redrawn the size-matched control. `/content/drive/MyDrive/CAI_Experiment` was seeded as a local directory rather than a Drive mount, so the notebook’s `BASE_DIR` still lands there. The notebook was uploaded to `/content/cai_3way_experiment_and_eval_v3.ipynb`.

## 23 November 2026, 8:26 p.m. Eastern (01:26:49 UTC) — install

Cell 3 (Unsloth and TRL) finished at 01:27:36 UTC. Unsloth 2026.11.10 loaded `FastLanguageModel` successfully. xformers does not match torch 2.14.1+cu130, so Unsloth falls back to PyTorch attention. Training continues on that fallback.

## 23 November 2026, 8:28 p.m. Eastern (01:28:20 UTC) — environment cell

Cell 4 checked the v3 mixes: 1,301 SFT rows and 1,301 DPO pairs per arm (1,001 CAI plus 300 replay). Kish ESS 789.94, max weight 2.2475. Write paths are `adapters/llama32_3b/v3` and `results/llama32_3b/v3` under `/content/drive/MyDrive/CAI_Experiment`. CUDA was true on the NVIDIA L4. Balance after cell 4 was 505.94, about 0.10 spent. Cell 6 copied `docs/intrinsic-motivation-scores.sqlite` (527 works, 99,637 chunks) and the treatment `state.db` onto local NVMe and opened them read-only, because Drive FUSE is a poor SQLite disk.

## 23 November 2026, 8:31 p.m. Eastern (01:31:36 UTC) — train functions loaded

Cell 8 loaded with `RUN_TRAINING=False` so the nine trains can be launched one arm and seed at a time, with a download after each. That is how the log can name a finished adapter before the next seed starts, and how a crash on seed 1 cannot take seed 0 with it. Next up was weighted seed 0.

## 23 November 2026, 8:31 p.m. Eastern (01:31:59 UTC) — weighted seed 0 started

This was the first QLoRA train. The VM downloaded the 4-bit base `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`.

## 23 November 2026, 8:48 p.m. Eastern (01:48:12 UTC) — weighted seed 0 finished

The train took 971.5 seconds, about 16.2 minutes: 82 SFT steps, then 82 R-DPO steps at α=0.01 and β=0.1. The adapter landed at `/content/drive/MyDrive/CAI_Experiment/adapters/llama32_3b/v3/weighted_seed0_final_lora`. It was downloaded to the private study store as `docs/study/artifacts/adapters/llama32_3b/v3/weighted_seed0_final_lora` (93 MB safetensors) plus `weighted_seed0_trainer_state.json`. The train transcript is `docs/study/artifacts/logs/train_weighted_seed0.colab.txt`. `colab usage` then read 505.46 at 1.94 per hour. Cumulative spend from 506.04 was 0.58 units. Those weights stay out of git because this repository’s Git LFS tracks the census jsonl and the scores sqlite.

## 23 November 2026, 8:49 p.m. Eastern (01:49:35 UTC) — uniform seed 0 started

Balance at the start of this train was 505.29 compute units, still 1.94 per hour, one active assignment. Uniform seed 0 is the Byrd and Lipton ablation: same rows as weighted seed 0, weights 1, same seed, so a later A-minus-B gap cannot be “the shuffle changed.”

## 23 November 2026, 9:06 p.m. Eastern (02:06:15 UTC) — uniform seed 0 finished

The adapter was downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/uniform_seed0_final_lora`. Cumulative spend from the start of the session was 1.23 units; the balance was 504.81, still at 1.94 per hour with one assignment.

## 23 November 2026, 9:06 p.m. Eastern (02:06:23 UTC) — control seed 0 started

Control seed 0 began immediately after the uniform download, at balance 504.81.

## 23 November 2026, 9:23 p.m. Eastern (02:23:01 UTC) — control seed 0 finished

The adapter was downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/control_seed0_final_lora`. Cumulative spend 1.72 units; balance 504.32.

## 23 November 2026, 9:23 p.m. Eastern (02:23:05 UTC) — weighted seed 1 started

Balance 504.32, still 1.94 per hour.

## 23 November 2026, 9:39 p.m. Eastern (02:39:55 UTC) — weighted seed 1 finished

The adapter was downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/weighted_seed1_final_lora`. Cumulative spend 2.36 units; balance 503.68.

## 23 November 2026, 9:39 p.m. Eastern (02:39:59 UTC) — uniform seed 1 started, then Jupyter dropped

Jupyter `colab exec` lost the websocket before this train entered the kernel (`RuntimeError: Connection was lost` at the `/content` chdir handshake). Session `ctw-v3` stayed alive (L4, idle). The kernel was restarted at 10:01 p.m. Eastern (03:01:49 UTC); cell 4 reloaded; further `colab exec` handshakes kept dropping. Remaining trains moved to a detached SSH process, `/content/v3_train_one_standalone.py`, still writing only under `adapters/llama32_3b/v3`. Finished before the drop: weighted 0, uniform 0, control 0, weighted 1.

## 23 November 2026, 10:13 p.m. Eastern (03:13:22 UTC) — uniform seed 1 started (SSH)

## 23 November 2026, 10:29 p.m. Eastern (03:29:23 UTC) — uniform seed 1 finished

Elapsed 955.1 seconds. Adapter downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/uniform_seed1_final_lora`. Remaining at that point: control 1, weighted 2, uniform 2, control 2.

## 23 November 2026, 10:31 p.m. Eastern (03:31:08 UTC) — control seed 1 started (SSH)

Balance 502.06, 1.94 per hour.

## 23 November 2026, 10:48 p.m. Eastern (03:48:28 UTC) — control seed 1 finished

Adapter downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/control_seed1_final_lora`. Cumulative spend 4.46 units; balance 501.58.

## 23 November 2026, 10:48 p.m. Eastern (03:48:35 UTC) — weighted seed 2 started (SSH)

## 23 November 2026, 11:05 p.m. Eastern (04:05:57 UTC) — weighted seed 2 finished

Adapter downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/weighted_seed2_final_lora`. Cumulative spend 5.11 units; balance 500.93.

## 23 November 2026, 11:06 p.m. Eastern (04:06:06 UTC) — uniform seed 2 started (SSH)

## 23 November 2026, 11:23 p.m. Eastern (04:23:48 UTC) — uniform seed 2 finished

Adapter downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/uniform_seed2_final_lora`. Cumulative spend 5.59 units; balance 500.45.

## 23 November 2026, 11:23 p.m. Eastern (04:23:56 UTC) — control seed 2 started (SSH)

## 23 November 2026, 11:40 p.m. Eastern (04:40:45 UTC) — control seed 2 finished

Adapter downloaded to `docs/study/artifacts/adapters/llama32_3b/v3/control_seed2_final_lora`. Cumulative spend 6.24 units; balance 499.80. All nine trains were done.

## 23 November 2026, 11:41 p.m. Eastern (04:41:26 UTC) — eval started

Standalone `/content/v3_eval_standalone.py` on the same L4. Primary metric: 8-token generation plus `parse_letter` / `parse_01`. Slots: the base, nine λ=0.5 adapters, and three weighted λ=1 references, then 50-item style judgments at λ=0.5.

## 24 November 2026, 1:38 a.m. Eastern (06:38:49 UTC) — eval finished

Results written under `/content/drive/MyDrive/CAI_Experiment/results/llama32_3b/v3` and downloaded to the private study store and, in this repository, to [`artifacts/results/llama32_3b/v3/`](artifacts/results/llama32_3b/v3/). Balance during download: 496.08 CU.

## 24 November 2026, 1:40 a.m. Eastern (06:40:08 UTC) — stop

`colab stop -s ctw-v3`. Final `colab usage`: balance **495.92**, rate 0.00 per hour, no active assignments. **CU spent: 10.12** (506.04 − 495.92). All nine trains finished. Primary eval and style eval finished.
