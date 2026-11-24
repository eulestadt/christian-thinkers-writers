# v3 results (Llama 3.2 3B)

**No measured scores yet.** This page fills when Colab writes `results/{ACTIVE_MODEL}/v3/`. Do not copy numbers from v1 or v2. Do not invent Mistral 24B or Qwen2-VL-72B rows.

Reported metric: 8-token generation + `parse_letter`, mean of seeds 0, 1, and 2 at **WiSE-FT λ=0.5**. List each seed beside the mean. Paired-bootstrap 95% intervals on Arm A minus Arm B and Arm A minus Arm C.

## Four honest arms

| Arm | What it is | λ | Seed mean | Seed 0 | Seed 1 | Seed 2 |
|---|---|---:|---:|---:|---:|---:|
| Base | Llama 3.2 3B Instruct, no fine-tune | 0 | _pending_ | — | — | — |
| C, citation-control | size-matched, weights 1 | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |
| B, uniform high-intrinsic | same 1,001 rows as A, weights 1 | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |
| A, weighted high-intrinsic | `w = s / mean`, no clip | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |

Reference row (not a fifth experimental arm): Arm A at λ=1, seeds 0–2 — _pending_.

## Contrasts (pending)

| Contrast | Point | 95% paired-bootstrap interval |
|---|---|---|
| A − B (weights vs filter) | _pending_ | _pending_ |
| A − C (high-intrinsic vs citation-control) | _pending_ | _pending_ |

## Mix (already measured; not a train result)

| | |
|---|---|
| CAI rows | 1,001 |
| Format replay | 300 |
| Kish ESS | 789.94 |
| Max weight | 2.248 |

## Style / dogmatic-vs-reflective (50 items, λ=0.5)

_pending_

JSON and CSVs from Colab land under `colab/results/` and are copied next to this page when they exist.
