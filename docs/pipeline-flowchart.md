# Experiment pipeline

Whole-experiment map for the Christian Thinkers Writers CAI archive: census and closed-form scoring, treatment vs citation-control Constitutional AI, GitHub + Drive layout, and the current v3 Colab run.

![Experiment pipeline flowchart](pipeline-flowchart.png)

GitHub also renders the mermaid source below. Regenerable from [`pipeline-flowchart.mmd`](pipeline-flowchart.mmd).

**Isolation:** extract, critique, revision, and judge prompts use author-text vocabulary only. No VanderWeele domains, FAI, or other flourishing taxonomies.

```mermaid
flowchart TB
  WP["Wikipedia Category: Christian writers"]
  CDB[("christian-writers.sqlite<br/>14,128 writers · 163,063 works")]
  SC["Closed-form Hoge + Gorsuch I/E-R<br/>one English PD work per author<br/>no LLM"]
  SDB[("intrinsic-motivation-scores.sqlite<br/>527 scored works")]
  T40["Top 40 by I/E-R intrinsic<br/>no I-E filter"]

  WP --> CDB --> SC --> SDB --> T40

  ISO[["Isolation: no VanderWeele, FAI,<br/>or flourishing taxonomies in extract,<br/>critique, revision, or judge prompts"]]

  TB["Treatment: 40 PD books from census"]
  TC["deepseek-flash constitution<br/>thinking off"]
  TP["2,500 prompts"]
  TSL["SL-CAI"]
  TRL["RLAIF"]
  T40 --> TB --> TC --> TP --> TSL --> TRL
  ISO -.-> TC

  CB["Control: 40 most-cited English PD<br/>1500–1930, excluding treatment<br/>works and authors"]
  CC["Same deepseek-flash loop<br/>constitution → 2,500 prompts<br/>→ SL-CAI → RLAIF"]
  T40 -.-> CB --> CC
  ISO -.-> CC

  GH["GitHub public archive<br/>eulestadt/christian-thinkers-writers"]
  CDB --> GH
  SDB --> GH
  TRL --> GH
  CC --> GH

  JL["Drive folder Alignment Research / CAI_Experiment<br/>JSONL copy for Colab"]
  NF["sqlite is NOT queried on Drive FUSE"]
  GH --> JL
  GH --> NF

  NV["Colab: local NVMe copy of sqlite<br/>read-only"]
  V3["v3: top 20 by ier_intrinsic<br/>one weighed turn, no UltraChat<br/>1,001 CAI + 300 format replay"]
  ARMS["Arms A/B/C<br/>weighted, uniform, size-matched control<br/>R-DPO α=0.01 β=0.1, seeds 0–2"]
  L3["Llama 3.2 3B only<br/>report WiSE-FT λ=0.5"]
  NF --> NV
  JL --> V3
  NV --> V3
  TRL --> V3
  CC --> V3
  V3 --> ARMS --> L3

  classDef db fill:#eef2ff,stroke:#4338ca,color:#111827
  classDef treat fill:#ecfdf5,stroke:#047857,color:#111827
  classDef ctrl fill:#fff7ed,stroke:#c2410c,color:#111827
  classDef iso fill:#fefce8,stroke:#a16207,color:#111827
  classDef gh fill:#f5f3ff,stroke:#6d28d9,color:#111827
  classDef col fill:#ecfeff,stroke:#0e7490,color:#111827
  class CDB,SDB db
  class TB,TC,TP,TSL,TRL treat
  class CB,CC ctrl
  class ISO,NF iso
  class GH gh
  class JL,NV,V3,ARMS,L3 col
```

## Stages

1. **Census.** Walk Wikipedia `Category:Christian writers` into [`christian-writers.sqlite`](christian-writers.sqlite) (14,128 writers, 163,063 works). Coverage: [census](christian-writers-coverage-report.md).
2. **Closed-form I/E-R scoring (no LLM).** Hoge + Gorsuch I/E-R on one English public-domain work per kept author → [`intrinsic-motivation-scores.sqlite`](intrinsic-motivation-scores.sqlite) (527 scored). Rank by I/E-R intrinsic only (no I−E filter) → top 40 treatment books. v3 trains on ranks 1–20 of that list. Coverage: [scoring](intrinsic-motivation-coverage.md), [method](intrinsic-motivation-scoring.md).
3. **Treatment CAI.** Those 40 PD books → `deepseek-flash` constitution (thinking off) → 2,500 prompts → SL-CAI → RLAIF. Tree: [`cai-pipeline/`](../cai-pipeline/). Coverage: [treatment](cai-deepseek-coverage.md).
4. **Control CAI.** 40 most-cited English PD works, publication years 1500–1930, excluding the treatment works and their authors → the same `deepseek-flash` loop. Tree: [`cai-pipeline-control/`](../cai-pipeline-control/). Coverage: [control](cai-control-coverage.md).
5. **GitHub public archive.** This repo is the citable source of truth (including Git LFS sqlite).
6. **Drive.** Shared folder **Alignment Research** / `CAI_Experiment` holds a JSONL copy for Colab. **Do not** open sqlite on Drive FUSE (`database is locked` / metadata touches).
7. **Colab v3 (current run).** Top 20 books by `ier_intrinsic`, one weighed turn rewritten from the v2 traces, no UltraChat, 300-row format replay kept. Three arms: weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, and a size-matched control. R-DPO α=0.01, β=0.1. Rejected answers alternate between the unrevised draft and a dogmatic rewrite. Seeds 0, 1, and 2. Reported score: mean at WiSE-FT λ=0.5. Published mix: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. Llama 3.2 3B only. Copy sqlite onto local NVMe, `chmod 0444`, open `file:...?mode=ro&immutable=1`. Notebook: [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb). The v1 notebook, the v1_fixed notebook, and the v2 notebook stay untouched. Coverage: [Colab](cai-3way-colab.md).

## How v3 differs from v1 and v2

v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 is the filter, the weighed rewrite, the three arms, R-DPO, the three seeds, and the λ=0.5 report described in stage 7. Adapters and results go under `adapters/{ACTIVE_MODEL}/v3/` and `results/{ACTIVE_MODEL}/v3/`. The nine QLoRA trains run on Colab.
