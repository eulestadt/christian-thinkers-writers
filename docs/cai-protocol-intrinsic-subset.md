# Master Execution Protocol (`CAI_PROTOCOL.md`) — I/E-R top 40, prompt-set size held at 2,500

This page is the Steps 0–4 build for the 40-book archive (2,500 prompts, including UltraChat). It is not the current training run.

The current run is **v3**: [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb). v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 keeps the top 20 books by `ier_intrinsic`, rewrites the v2 traces into one weighed turn, and does not use UltraChat. The 300-row format replay stays. Three arms: weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, and a size-matched control. R-DPO α=0.01, β=0.1. Bidirectional rejected answers (unrevised draft and dogmatic rewrite). Seeds 0–2. Reported score: WiSE-FT λ=0.5. Published mix: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. The v1, v1_fixed, and v2 notebooks stay untouched. Coverage: [`cai-3way-colab.md`](cai-3way-colab.md).

> **Instructions for Cursor Agent:** Execute a two-stage Constitutional AI (Bai et al., 2022, `arXiv:2212.08073`) pipeline over **40 books** in `books_raw/` (I/E-R intrinsic top 40 from `/cursor/stores/self/docs/intrinsic-motivation-scores.sqlite`; **do not filter on I − E**) to fine-tune `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` first, then `unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit`.
>
> * **Strict Epistemic Isolation (Zero External Leakage):** Do NOT introduce any external philosophical frameworks, benchmark categories, or predefined taxonomies (including VanderWeele’s domains and FAI Benchmark terminology) in any system prompt or schema. Every principle, critique, revision, and evaluation rubric must be derived exclusively from the authors’ texts in `books_raw/`.
> * **API Execution ($0 Free Tier):** Execute Steps 1–4 using the Google Gemini Free Tier API (`google-genai` SDK), using a **single** Gemini model (`gemini-2.0-flash` if available) with SQLite state checkpointing (`data/state.db`) and rate-limit throttling (`15 RPM`, `<250,000 TPM`).
> * **Compute Offload:** Execute Step 5 on the remote Google Colab GPU runtime via Google Drive/Colab sync.
>
> **Why 40, not 400:** Ranked I/E-R intrinsic authors only. Prompt-set **size** is preserved: 40 × 50 elicited prompts = 2,000, plus 500 UltraChat = **2,500**.

---

### Directory Structure

```text
cai-pipeline/
├── books_raw/                  # The 40 source texts (PDF / EPUB / TXT from Gutenberg/Wikisource)
├── data/
│   ├── state.db                # SQLite checkpoint database for resumable runs
│   ├── corpus_manifest.json    # writer_id, author, work_title, pd_url, ier_intrinsic
│   ├── raw_principles.jsonl    # Unaltered principles & prompts extracted per book
│   ├── constitution.json       # Consolidated flat pool of {critique_request, revision_request}
│   ├── prompts_dataset.jsonl   # 2,000 elicited + 500 UltraChat
│   ├── sl_cai_train.jsonl      # Stage 1 SFT Dataset: {prompt: X, completion: Y_final}
│   └── rlaif_dpo_train.jsonl   # Stage 2 Preference Dataset: {prompt: X, chosen: Y_w, rejected: Y_l}
├── scripts/
│   ├── 00_select_and_fetch.py
│   ├── 01_extract_constitution.py
│   ├── 02_build_prompt_set.py
│   ├── 03_stage1_sl_cai.py
│   ├── 04_stage2_rlaif.py
│   └── 05_colab_train.py
└── .env                        # GEMINI_API_KEY=...
```

---

### Step 0: Select and fetch (`scripts/00_select_and_fetch.py`)

From `works_scored` ⋈ `work_subscores` where `subscale = 'ier_intrinsic'`, `ORDER BY score DESC LIMIT 40`. Join `author_decisions.canonical_name`. Download `text_source_url` / `pd_url` into `books_raw/{writer_id}.txt`. Write `data/corpus_manifest.json`. No LLM.

---

### Step 1: Corpus-Only Constitutional Extraction (`scripts/01_extract_constitution.py`)

**CAI Paper Grounding:** *Section 3.1 & Appendix A* — The Constitution is a flat list of normative instructions, where each item consists solely of a `critique_request` and a `revision_request`.

1. **Local Text Parsing:** Extract plain text from each of the **40** books in `books_raw/` using `PyMuPDF` (`fitz`) or `ebooklib` (or already-plain Gutenberg TXT).
2. **Text-Grounded Extraction Prompt (1 Book per Request):** Send each book's text to the Gemini Free Tier API with `response_mime_type="application/json"`. If the book exceeds the context window, send sequential chunks of the **same book** and merge JSON (still 1 book identity; never mix books). Notice there are **no external buzzwords**—the prompt forces the extractor to use only the author's internal logic and vocabulary. Ask for **50** dilemmas (not 5) so 40 books still yield ~2,000 prompts:

```text
Read the provided text in its entirety. Using ONLY the arguments, worldview, and normative claims explicitly present in this text (without importing any outside terminology or external frameworks), extract:
1. The core normative principles argued by the author, formatted as Constitutional AI pairs:
   - "critique_request": An instruction asking a reviewer to identify how a response fails to uphold or contradicts this specific argument from the text.
   - "revision_request": An instruction asking a writer to rewrite the response so it faithfully embodies this specific argument and reasoning from the text.
2. 50 realistic, first-person human questions or dilemmas that the arguments in this text directly address. If the text cannot support 50 distinct dilemmas, produce as many as the text actually supports; do not invent dilemmas the author does not address, and do not import topics from other books or from any external framework.

```

**JSON Schema:**

```json
{
  "book_id": "string",
  "author": "string",
  "principles": [
    {
      "source_thesis": "Verbatim or closely paraphrased thesis from the text",
      "critique_request": "string",
      "revision_request": "string"
    }
  ],
  "elicited_prompts": ["string"]
}
```

3. **Deduplication into the Flat Constitution (`data/constitution.json`):**
Following Bai et al. (2022), which stores the Constitution as a flat pool of principles, embed the extracted `principles` locally using `sentence-transformers/all-MiniLM-L6-v2` and cluster near-duplicates (`AgglomerativeClustering`) so that redundant rules across the 40 books merge into a clean, flat list of `critique_request` and `revision_request` objects.

If a short book yields fewer than 50 elicited prompts, **do not pad from UltraChat or other books at extract time**. Step 2 may oversample elicited prompts from longer books in the same 40 so the elicited total still reaches 2,000, still inside the 40-book corpus.

---

### Step 2: Prompt Dataset Assembly (`scripts/02_build_prompt_set.py`)

**CAI Paper Grounding:** *Section 3.1* — Combines prompts that test the target normative behavior with general helpfulness prompts to preserve general instruction-following.

1. Collect **2,000** `elicited_prompts` generated directly from the 40 books in Step 1 (target 50/book; rebalance within the 40 if needed).
2. Mix in **500** general instruction-following prompts from `HuggingFaceH4/ultrachat_200k` (including standard multiple-choice and formatting instructions so the model does not lose basic structural competence).
3. Save all **2,500** prompts to `data/prompts_dataset.jsonl`.

---

### Step 3: Stage 1 (SL-CAI) — Iterative Critique & Revision (`scripts/03_stage1_sl_cai.py`)

**CAI Paper Grounding:** *Section 3.1 & Section 3.3 (Figure 4)* — Generate an initial naive response $Y_0$. Randomly sample principles from the Constitution to iteratively critique and revise the response ($Y_0 \rightarrow Y_1 \rightarrow Y_2 \rightarrow Y_3$). Fine-tune on `(Prompt X, Final Revision Y_3)`.

1. For each prompt $X$ in `data/prompts_dataset.jsonl`, sample **3 principles** ($P_1, P_2, P_3$) from `data/constitution.json` (either uniformly at random as in Bai et al., Section 3.1, or by cosine similarity between the prompt embedding and `source_thesis` embedding so a relevant book's principle critiques the prompt). Sample **only** from the 40-book constitution.
2. Execute the 3-step Critique-and-Revision loop (batched 5 prompts per API request to stay well within the 1,500 RPD free tier):

```text
Prompt (X): {prompt}

Perform the following sequential Critique-and-Revision steps:
- Step 0 [Draft_0]: Write an initial baseline response to Prompt (X).
- Step 1 [Critique_1]: {P1.critique_request}
- Step 2 [Revision_1]: {P1.revision_request} (Rewrite Draft_0 based on Critique_1).
- Step 3 [Critique_2]: {P2.critique_request}
- Step 4 [Revision_2]: {P2.revision_request} (Rewrite Revision_1 based on Critique_2, retaining prior improvements).
- Step 5 [Critique_3]: {P3.critique_request}
- Step 6 [Revision_3_Final]: {P3.revision_request} (Write the final, cohesive response integrating all revisions).

```

3. Save strictly `{"prompt": X, "completion": Revision_3_Final}` to `data/sl_cai_train.jsonl`, and store `Draft_0` and `Revision_1` in `data/state.db`.

At 2,500 prompts, batched 5 per request ≈ 500 requests. Free tier 1,500 RPD ⇒ **more than one calendar day**; resume from `data/state.db`. Use **one** model (`gemini-2.0-flash` if it authenticates). Throttle 15 RPM, `<250,000 TPM`.

---

### Step 4: Stage 2 (RLAIF) — Chain-of-Thought AI Preference Labeling (`scripts/04_stage2_rlaif.py`)

**CAI Paper Grounding:** *Section 4.1* — Present a prompt $X$, two candidate responses $(Y_A, Y_B)$, and sampled Constitutional principles to the AI Judge. Require Chain-of-Thought reasoning before selecting the preferred response.

1. Pair two candidate responses $(Y_A, Y_B)$ for each prompt $X$ in randomized order. Use `Draft_0` vs `Revision_3_Final` from Step 3 (or two revisions) as the pair; principles $P_1,P_2,P_3$ from the **40-book** constitution only.
2. **Strict CAI Section 4.1 Judge Prompt (Zero External Terminology):**

```text
Human: Consider the following conversation between a human and an assistant:
Human Prompt: {X}

Options:
(A) {Y_A}
(B) {Y_B}

Evaluate which option better adheres to the following principles:
1. {P1.source_thesis}
2. {P2.source_thesis}
3. {P3.source_thesis}

Assistant: Let's think step by step inside <thought> tags about how each option aligns with or violates the provided principles. Then output the final choice as "A" or "B".

```

3. Write the resulting preference pairs to `data/rlaif_dpo_train.jsonl`:
`{"prompt": X, "chosen": Y_winner, "rejected": Y_loser}`

---

### Step 5: Google Colab GPU Training — SL-CAI SFT + RLAIF DPO (`scripts/05_colab_train.py`)

The block below is the original two-model sketch for this protocol. The notebook that implements the current run is v3, linked at the top of this page. Do not treat `05_colab_train.py` or a Mistral pass as the latest train.

**CAI Paper Grounding:** *Section 3.2 (SL-CAI) & Section 4.2 (RL-CAI via DPO equivalence)* — First train the base model on `sl_cai_train.jsonl` for 1 epoch, then train that resulting checkpoint on `rlaif_dpo_train.jsonl` for 1 epoch.

Upload `sl_cai_train.jsonl` and `rlaif_dpo_train.jsonl` to Google Drive (`CAI_Pipeline/data`) and trigger the Unsloth training script on Colab GPU (L4 24GB or A100 40GB). Run **Llama 3.2 3B Instruct** first; then **Mistral Small 24B Instruct**.

```python
# scripts/05_colab_train.py
import gc, torch
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

PatchDPOTrainer()

MODELS = [
    {
        "name": "llama-3.2-3b-cai",
        "hf_id": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "sft_bs": 4, "sft_ga": 4,
        "user_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
        "asst_tag": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    {
        "name": "mistral-small-24b-cai",
        "hf_id": "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
        "sft_bs": 1, "sft_ga": 16,
        "user_tag": "[INST] ",
        "asst_tag": "[/INST]",
    }
]

DATA_DIR = "/content/drive/MyDrive/CAI_Pipeline/data"
OUT_DIR  = "/content/drive/MyDrive/CAI_Pipeline/models"

for m in MODELS:
    # 1. Load 4-bit Base Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["hf_id"], max_seq_length=2048, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42
    )

    # 2. Stage 1: SL-CAI (Supervised Fine-Tuning on Revised Responses)
    sft_ds = load_dataset("json", data_files=f"{DATA_DIR}/sl_cai_train.jsonl", split="train")
    sft_ds = sft_ds.map(lambda ex: {
        "text": [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}, {"role": "assistant", "content": c}],
                tokenize=False, add_generation_prompt=False
            ) for p, c in zip(ex["prompt"], ex["completion"])
        ]
    }, batched=True)

    sft_trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=sft_ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=2048,
            per_device_train_batch_size=m["sft_bs"], gradient_accumulation_steps=m["sft_ga"],
            warmup_ratio=0.05, num_train_epochs=1, learning_rate=2e-4,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            optim="adamw_8bit", output_dir=f"{OUT_DIR}/{m['name']}_stage1_sl"
        )
    )
    sft_trainer = train_on_responses_only(sft_trainer, instruction_part=m["user_tag"], response_part=m["asst_tag"])
    sft_trainer.train()

    # 3. Stage 2: RLAIF (Preference Optimization on Stage 1 Checkpoint)
    dpo_ds = load_dataset("json", data_files=f"{DATA_DIR}/rlaif_dpo_train.jsonl", split="train")
    dpo_ds = dpo_ds.map(lambda ex: {
        "prompt": tokenizer.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True),
        "chosen": ex["chosen"] + tokenizer.eos_token,
        "rejected": ex["rejected"] + tokenizer.eos_token,
    })

    dpo_trainer = DPOTrainer(
        model=model, ref_model=None, tokenizer=tokenizer, train_dataset=dpo_ds,
        args=DPOConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            warmup_ratio=0.1, num_train_epochs=1, learning_rate=5e-6, beta=0.1,
            max_length=2048, max_prompt_length=512,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            optim="adamw_8bit", output_dir=f"{OUT_DIR}/{m['name']}_stage2_dpo"
        )
    )
    dpo_trainer.train()

    # 4. Save Final Adapter
    model.save_pretrained(f"{OUT_DIR}/{m['name']}_final_lora")
    tokenizer.save_pretrained(f"{OUT_DIR}/{m['name']}_final_lora")

    del model, tokenizer, sft_trainer, dpo_trainer
    gc.collect()
    torch.cuda.empty_cache()
```

Still needed before execution: `GEMINI_API_KEY`, Colab/Drive folder, and confirmation to fetch the 40 Gutenberg texts into `books_raw/`.
