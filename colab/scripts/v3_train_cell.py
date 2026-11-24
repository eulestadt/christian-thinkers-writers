# v3 training. Three arms, seeds 0, 1, and 2. One epoch of weighted SFT, then R-DPO.
# Weights multiply the whole per-sequence loss once. Rows are not drawn with replacement.
# Scores stay in the weight column. They are never written into a prompt.

import math
import torch
import torch.nn.functional as F

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

TRAIN_ARMS = ("weighted", "uniform", "control")  # A, B, C
TRAIN_SEEDS = (0, 1, 2)
RDPO_ALPHA = 0.01
RDPO_BETA = float(DPO_BETA)
LEAKAGE_TERMS_V3 = (
    "vanderweele",
    "vander weele",
    "flourishing",
    "fai benchmark",
    "fai taxonomy",
    "human flourishing",
    "global flourishing",
)


def _v3_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _read_jsonl(path) -> list:
    p = Path(path)
    if not p.is_file() or p.stat().st_size < 1000:
        raise FileNotFoundError(
            f"v3 data missing: {p}. Refusing to train. "
            "Cell 1 must find colab/data/v3/{weighted,uniform,control}/"
            "sl_cai_train.jsonl on the cloned branch."
        )
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise FileNotFoundError(f"v3 data empty: {p}")
    return rows


def _shuffle_rows(rows: list, seed: int) -> list:
    out = list(rows)
    rng = random.Random(int(seed))
    rng.shuffle(out)
    return out


def _assert_text_clean(text: str, where: str) -> None:
    low = (text or "").lower()
    for term in LEAKAGE_TERMS_V3:
        if term in low:
            raise RuntimeError(f"leakage term in {where}")


def _kish(weights: list) -> float:
    s = float(sum(weights))
    sq = float(sum(w * w for w in weights))
    if sq <= 0:
        raise RuntimeError("Kish ESS undefined")
    return (s * s) / sq


def _check_loaded_mixes(mixes: dict) -> None:
    w_sft = mixes["weighted"]["sft"]
    u_sft = mixes["uniform"]["sft"]
    c_sft = mixes["control"]["sft"]
    if [r["row_id"] for r in w_sft] != [r["row_id"] for r in u_sft]:
        raise RuntimeError("Arm A and Arm B row ids differ")
    if [r["prompt"] for r in w_sft] != [r["prompt"] for r in u_sft]:
        raise RuntimeError("Arm A and Arm B prompts differ")
    if [r.get("completion") for r in w_sft] != [r.get("completion") for r in u_sft]:
        raise RuntimeError("Arm A and Arm B completions differ")
    for seed in TRAIN_SEEDS:
        a = _shuffle_rows(w_sft, seed)
        b = _shuffle_rows(u_sft, seed)
        if [r["row_id"] for r in a] != [r["row_id"] for r in b]:
            raise RuntimeError(f"seed {seed} shuffle diverged between weighted and uniform")
    for arm, rows in (("uniform", u_sft), ("control", c_sft)):
        bad = [r["row_id"] for r in rows if float(r.get("weight") or 0) != 1.0]
        if bad:
            raise RuntimeError(f"{arm} has weights other than 1, e.g. {bad[:3]}")
    cai = [r for r in w_sft if r.get("source") == "elicited"]
    if not cai:
        raise RuntimeError("weighted mix has no elicited rows")
    n = len(cai)
    if sum(1 for r in c_sft if r.get("source") == "elicited") != n:
        raise RuntimeError("control CAI count does not match the weighted arm")
    weights = [float(r["weight"]) for r in cai]
    mean_w = sum(weights) / n
    if abs(mean_w - 1.0) > 1e-3:
        raise RuntimeError(f"mean CAI weight is {mean_w}, expected 1 after renormalizing")
    if min(weights) <= 0:
        raise RuntimeError("non-positive weight")
    ess = _kish(weights)
    print(
        f"CAI n={n} Kish ESS={ess:.2f} max weight={max(weights):.4f} "
        f"min weight={min(weights):.4f} mean weight={mean_w:.4f}"
    )
    summary_path = Path(repo) / "colab" / "data" / "v3" / "mix_summary.json"
    for cand in (Path(V3_DATA_DIR) / "mix_summary.json", summary_path):
        if cand.is_file():
            summary = json.loads(cand.read_text(encoding="utf-8"))
            print(
                "mix_summary Kish ESS",
                round(float(summary["kish_ess"]), 2),
                "max weight",
                round(float(summary["max_weight"]), 4),
            )
            break
    for arm, pack in mixes.items():
        for kind, rows in pack.items():
            n_replay = sum(1 for r in rows if r.get("source") == "format_replay")
            if n_replay != 300:
                raise RuntimeError(f"{arm} {kind} format_replay count is {n_replay}, expected 300")
            for r in rows:
                src = str(r.get("source") or "").lower()
                if src in ("ultrachat", "ethics", "truthfulqa", "mmlu") or "ultrachat" in src:
                    raise RuntimeError(f"banned source {src} in {arm} {kind}")
                keys = ("prompt", "completion") if kind == "sft" else ("prompt", "chosen", "rejected")
                _assert_text_clean("\n".join(str(r.get(k) or "") for k in keys), f"{arm} {kind} {r.get('row_id')}")
                if "s" in r or "ier_intrinsic" in r:
                    raise RuntimeError("score column is in a train file")
        kinds = {r.get("rejected_kind") for r in pack["dpo"] if r.get("source") == "elicited"}
        if kinds != {"draft", "dogmatic"}:
            raise RuntimeError(f"{arm} rejected kinds are {kinds}")
    print("v3 mixes checked: three arms, shared replay, both rejected kinds, no UltraChat")


def _load_mixes() -> dict:
    mixes = {}
    for arm in TRAIN_ARMS:
        root = Path(V3_ARM_RO[arm])
        mixes[arm] = {
            "sft": _read_jsonl(root / "sl_cai_train.jsonl"),
            "dpo": _read_jsonl(root / "rlaif_dpo_train.jsonl"),
        }
        if [r["row_id"] for r in mixes[arm]["sft"]] != [r["row_id"] for r in mixes[arm]["dpo"]]:
            raise RuntimeError(f"{arm} SFT and DPO row order differ")
    _check_loaded_mixes(mixes)
    return mixes


def _encode_pair(tokenizer, user: str, assistant: str, max_len: int):
    full = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=True,
        add_generation_prompt=False,
    ))
    prompt = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    ))
    if full[: len(prompt)] == prompt:
        plen = len(prompt)
    else:
        header = tokenizer.encode(PRESET["asst_tag"], add_special_tokens=False)
        plen = None
        for i in range(0, len(full) - len(header) + 1):
            if full[i : i + len(header)] == header:
                plen = i + len(header)
                break
        if plen is None:
            raise RuntimeError("could not locate the assistant span in the chat template")
    full = full[: max_len]
    if plen >= len(full) or plen < 1:
        raise RuntimeError(f"no completion tokens inside max_seq_length ({plen} vs {len(full)})")
    return full, plen


def _device_of(model):
    return next(model.parameters()).device


def _pad_batch(seqs, pad_id: int):
    width = max(len(s) for s in seqs)
    ids, attn = [], []
    for s in seqs:
        ids.append(s + [pad_id] * (width - len(s)))
        attn.append([1] * len(s) + [0] * (width - len(s)))
    return ids, attn


def _forward_logits(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(out, "logits", None)
    if logits is None:
        raise RuntimeError(
            "the model forward returned no logits. "
            "UNSLOTH_RETURN_LOGITS=1 must be set before unsloth is imported. "
            "Restart the runtime and run the install cell, then this cell."
        )
    return logits


def _completion_logprobs(model, seqs, prompt_lens):
    pad_id = tokenizer_pad_id(model)
    ids, attn = _pad_batch(seqs, pad_id)
    device = _device_of(model)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn, dtype=torch.long, device=device)
    logits = _forward_logits(model, input_ids, attention_mask)
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    tok = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    vals = []
    for i, (seq, plen) in enumerate(zip(seqs, prompt_lens)):
        start = plen - 1
        end = len(seq) - 1
        if end <= start:
            raise RuntimeError("empty completion logprob")
        vals.append(tok[i, start:end].sum())
    return torch.stack(vals)


def tokenizer_pad_id(model):
    tok = tokenizer_pad_id.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return int(tok.pad_token_id)


def _sft_loss(model, tokenizer, batch):
    seqs, lens, labels, weights = [], [], [], []
    for row in batch:
        _assert_text_clean(row["prompt"] + "\n" + row["completion"], row.get("row_id"))
        full, plen = _encode_pair(tokenizer, row["prompt"], row["completion"], MAX_SEQ_LEN)
        lab = [-100] * plen + full[plen:]
        seqs.append(full)
        lens.append(plen)
        labels.append(lab)
        weights.append(float(row["weight"]))
    pad_id = tokenizer_pad_id(model)
    ids, attn = _pad_batch(seqs, pad_id)
    lab_pad, _ = _pad_batch(labels, -100)
    device = _device_of(model)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn, dtype=torch.long, device=device)
    label_t = torch.tensor(lab_pad, dtype=torch.long, device=device)
    logits = _forward_logits(model, input_ids, attention_mask)
    shift_logits = logits[:, :-1, :]
    shift_labels = label_t[:, 1:]
    tok = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100)
    denom = mask.sum(dim=1).clamp(min=1)
    per_seq = (tok * mask).sum(dim=1) / denom
    w = torch.tensor(weights, dtype=per_seq.dtype, device=per_seq.device)
    return (per_seq * w).mean()


def _rdpo_loss(model, tokenizer, batch):
    chosen_seqs, chosen_lens, rej_seqs, rej_lens = [], [], [], []
    y_w, y_l, weights = [], [], []
    for row in batch:
        _assert_text_clean(
            "\n".join(str(row.get(k) or "") for k in ("prompt", "chosen", "rejected")),
            row.get("row_id"),
        )
        c_ids, c_plen = _encode_pair(tokenizer, row["prompt"], row["chosen"], MAX_SEQ_LEN)
        r_ids, r_plen = _encode_pair(tokenizer, row["prompt"], row["rejected"], MAX_SEQ_LEN)
        chosen_seqs.append(c_ids)
        rej_seqs.append(r_ids)
        chosen_lens.append(c_plen)
        rej_lens.append(r_plen)
        y_w.append(len(c_ids) - c_plen)
        y_l.append(len(r_ids) - r_plen)
        weights.append(float(row["weight"]))
    n = len(chosen_seqs)
    policy = _completion_logprobs(model, chosen_seqs + rej_seqs, chosen_lens + rej_lens)
    policy_w, policy_l = policy[:n], policy[n:]
    # Reference is the base policy with the adapter off. That is TRL's PEFT default
    # when ref_model is None, which is what the v2 trainer used. Beta keeps this
    # policy close to that base. WiSE-FT then scales the adapter at eval.
    with torch.no_grad():
        ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        with ctx:
            ref = _completion_logprobs(model, chosen_seqs + rej_seqs, chosen_lens + rej_lens)
    ref_w = ref[:n].detach()
    ref_l = ref[n:].detach()
    len_w = torch.tensor(y_w, dtype=policy_w.dtype, device=policy_w.device)
    len_l = torch.tensor(y_l, dtype=policy_w.dtype, device=policy_w.device)
    w = torch.tensor(weights, dtype=policy_w.dtype, device=policy_w.device)
    # Park et al. 2024, the length term inside the logistic. No clip, no resampling.
    margin = RDPO_BETA * ((policy_w - ref_w) - (policy_l - ref_l)) + RDPO_ALPHA * (len_w - len_l)
    losses = -w * F.logsigmoid(margin)
    return losses.mean(), float(margin.detach().mean().cpu())


def _lr_at(step: int, total: int, base_lr: float, warmup_ratio: float) -> float:
    warm = max(1, int(total * warmup_ratio))
    if step < warm:
        return base_lr * float(step + 1) / float(warm)
    remain = max(1, total - warm)
    return base_lr * max(0.0, float(total - step) / float(remain))


def _adamw(model, lr: float):
    import bitsandbytes as bnb
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable LoRA parameters")
    return bnb.optim.AdamW8bit(params, lr=lr), params


def _run_stage(model, tokenizer, rows, seed, arm, stage: str):
    # One row per forward so the weight multiplies that row's whole loss, and so
    # R-DPO logits fit on a 16GB GPU. Effective batch stays the v2 preset.
    preset_bs = int(PRESET["sft_bs"] if stage == "sft" else PRESET["dpo_bs"])
    preset_ga = int(PRESET["sft_ga"] if stage == "sft" else PRESET["dpo_ga"])
    bs = 1
    ga = preset_bs * preset_ga
    base_lr = float(SFT_LR if stage == "sft" else DPO_LR)
    warmup = 0.05 if stage == "sft" else 0.1
    shuffled = _shuffle_rows(rows, seed)
    opt, params = _adamw(model, base_lr)
    n_batches = max(1, math.ceil(len(shuffled) / bs))
    opt_steps = max(1, math.ceil(n_batches / ga))
    model.train()
    model.config.use_cache = False
    opt.zero_grad(set_to_none=True)
    logs = []
    micro = 0
    step = 0
    last_margin = None
    for start in range(0, len(shuffled), bs):
        batch = shuffled[start : start + bs]
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, opt_steps, base_lr, warmup)
        if stage == "sft":
            loss = _sft_loss(model, tokenizer, batch)
            margin = None
        else:
            loss, margin = _rdpo_loss(model, tokenizer, batch)
            last_margin = margin
        (loss / ga).backward()
        micro += 1
        if micro % ga == 0 or start + bs >= len(shuffled):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 5 == 0 or step == opt_steps:
                rec = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "lr": opt.param_groups[0]["lr"],
                    "stage": stage,
                    "arm": arm,
                    "seed": int(seed),
                }
                if margin is not None:
                    rec["margin"] = margin
                logs.append(rec)
                print(rec, flush=True)
    del opt
    return logs


def _adapter_dir(arm: str, seed: int) -> Path:
    path = Path(ADAPT_DIR) / f"{arm}_seed{int(seed)}_final_lora"
    resolved = path.resolve()
    if resolved.name != f"{arm}_seed{int(seed)}_final_lora" or "v3" not in resolved.parts:
        raise RuntimeError(f"refusing to write adapter outside v3: {resolved}")
    if resolved.parent.name != "v3":
        raise RuntimeError(f"adapter parent must be v3, got {resolved}")
    return path


def _adapter_ready(path: Path) -> bool:
    return (path / "adapter_config.json").is_file()


def train_v3_arm(arm: str, seed: int, mixes: dict) -> None:
    _assert_v3_write_paths()
    dest = _adapter_dir(arm, seed)
    if SKIP_TRAINING_IF_ADAPTERS_EXIST and _adapter_ready(dest):
        print(f"SKIP {arm} seed {seed}: {dest}")
        return
    from unsloth import FastLanguageModel

    print(f"\n{'=' * 60}\nTRAIN {arm} seed {seed}  R-DPO alpha {RDPO_ALPHA} beta {RDPO_BETA}\n{'=' * 60}", flush=True)
    _v3_seed(seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    _v3_seed(seed)
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=int(seed),
    )
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_pad_id.tokenizer = tokenizer

    sft_logs = _run_stage(model, tokenizer, mixes[arm]["sft"], seed, arm, "sft")
    dpo_logs = _run_stage(model, tokenizer, mixes[arm]["dpo"], seed, arm, "dpo")
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    tokenizer.save_pretrained(str(dest))
    state = {
        "arm": arm,
        "seed": int(seed),
        "rdpo_alpha": RDPO_ALPHA,
        "rdpo_beta": RDPO_BETA,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "sft_lr": SFT_LR,
        "dpo_lr": DPO_LR,
        "log_history": sft_logs + dpo_logs,
    }
    (Path(ADAPT_DIR) / f"{arm}_seed{int(seed)}_trainer_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    print(f"Saved {arm} seed {seed} to {dest}", flush=True)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if ACTIVE_MODEL != "llama32_3b":
    raise RuntimeError("v3 trains Llama 3.2 3B only")
if LORA_ATTENTION_ONLY:
    raise RuntimeError("v3 keeps all seven LoRA projections. LORA_ATTENTION_ONLY must stay False.")

if RUN_TRAINING:
    _mixes = _load_mixes()
    for _seed in TRAIN_SEEDS:
        for _arm in TRAIN_ARMS:
            train_v3_arm(_arm, _seed, _mixes)
    print("v3 training pass finished. Nine trains are arm x seed. The reported eval scale is lambda 0.5.")
else:
    print("RUN_TRAINING=False — not launching the nine QLoRA trains in this session.")
