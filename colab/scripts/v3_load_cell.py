# v3 eval stack. WiSE-FT is an adapter scale: lambda 0.5 is the reported result.
# lambda 1 is the unblended reference on arm A. lambda 0 is the base model, scored once.
# Adapters are loaded only from adapters/{ACTIVE_MODEL}/v3. v1 and v2 LoRAs are not fallbacks.

from peft import PeftModel

TRAIN_ARMS = ("weighted", "uniform", "control")
TRAIN_SEEDS = (0, 1, 2)
REPORTED_LAMBDA = 0.5
REFERENCE_LAMBDA = 1.0

model = None
tokenizer = None


def _slot(arm: str, seed: int, lam: float) -> str:
    return f"{arm}_s{int(seed)}_l{float(lam):.1f}"


SCORE_ARMS = ["base"]
for _seed in TRAIN_SEEDS:
    for _arm in TRAIN_ARMS:
        SCORE_ARMS.append(_slot(_arm, _seed, REPORTED_LAMBDA))
    SCORE_ARMS.append(_slot("weighted", _seed, REFERENCE_LAMBDA))
ARMS = list(SCORE_ARMS)
print("v3 score slots", len(SCORE_ARMS), SCORE_ARMS)


def _adapter_path(arm: str, seed: int) -> Path:
    path = Path(ADAPT_DIR) / f"{arm}_seed{int(seed)}_final_lora"
    if path.parent.resolve().name != "v3":
        raise RuntimeError(f"refusing adapter outside v3: {path}")
    return path


def _require_v3_adapters() -> None:
    missing = []
    for arm in TRAIN_ARMS:
        for seed in TRAIN_SEEDS:
            path = _adapter_path(arm, seed)
            if not (path / "adapter_config.json").is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "v3 adapters missing. Train the nine QLoRA runs in this notebook first. "
            + " | ".join(missing)
        )


def _set_lora_lambda(peft_model, adapter_name: str, lam: float) -> None:
    found = 0
    for module in peft_model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict) or adapter_name not in scaling:
            continue
        base = getattr(module, "_v3_base_scale", None)
        if base is None:
            base = {}
            module._v3_base_scale = base
        if adapter_name not in base:
            base[adapter_name] = float(scaling[adapter_name])
        scaling[adapter_name] = base[adapter_name] * float(lam)
        found += 1
    if found == 0:
        raise RuntimeError(f"WiSE-FT scale found no LoRA layers for {adapter_name}")


def activate_arm(slot: str) -> str:
    """Load the v3 adapter named by a score slot and set its WiSE-FT scale."""
    if model is None:
        raise RuntimeError("eval model is not loaded")
    if slot == "base":
        return "base"
    parts = slot.split("_")
    # weighted_s0_l0.5 or uniform_s1_l0.5 or control_s2_l1.0
    if len(parts) != 3 or not parts[1].startswith("s") or not parts[2].startswith("l"):
        raise RuntimeError(f"bad score slot {slot}")
    arm = parts[0]
    seed = int(parts[1][1:])
    lam = float(parts[2][1:])
    if arm not in TRAIN_ARMS:
        raise RuntimeError(f"unknown arm in slot {slot}")
    name = f"{arm}_s{seed}"
    if name not in getattr(model, "peft_config", {}):
        path = _adapter_path(arm, seed)
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing v3 adapter {path}")
        model.load_adapter(str(path), adapter_name=name)
    model.set_adapter(name)
    _set_lora_lambda(model, name, lam)
    return name


def load_eval_stack():
    global model, tokenizer
    _assert_v3_write_paths()
    _require_v3_adapters()
    from unsloth import FastLanguageModel

    base_model, tokenizer_ = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    first = _adapter_path("weighted", 0)
    print("load v3 adapters from", ADAPT_DIR)
    peft_model = PeftModel.from_pretrained(base_model, str(first), adapter_name="weighted_s0")
    for arm in TRAIN_ARMS:
        for seed in TRAIN_SEEDS:
            name = f"{arm}_s{seed}"
            if name in peft_model.peft_config:
                continue
            peft_model.load_adapter(str(_adapter_path(arm, seed)), adapter_name=name)
    FastLanguageModel.for_inference(peft_model)
    peft_model.eval()
    _set_lora_lambda(peft_model, "weighted_s0", REPORTED_LAMBDA)
    return peft_model, tokenizer_


def generate_with_arm(arm: str, messages: list, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("call load_eval_stack before generate_with_arm")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt")
    device = _device_of_eval(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature))
    else:
        gen_kwargs.update(do_sample=False)
    ctx = model.disable_adapter() if arm == "base" else nullcontext()
    if arm != "base":
        activate_arm(arm)
    with torch.inference_mode():
        with ctx:
            out = model.generate(**inputs, **gen_kwargs)
    n_in = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def _device_of_eval(peft_model):
    try:
        return next(peft_model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if RUN_EVAL or RUN_ARENA or RUN_OOD_EVAL or RUN_OOD_APPLY or RUN_LOGPROB_EVAL or RUN_STYLE:
    model, tokenizer = load_eval_stack()
    print("Loaded base + v3 adapters:", list(getattr(model, "peft_config", {}).keys()))
else:
    print("Skipping v3 adapter load (no GPU eval this session).")
