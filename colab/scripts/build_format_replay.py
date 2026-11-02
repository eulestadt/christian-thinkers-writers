#!/usr/bin/env python3
"""Build a small v2 format-following SFT/DPO slice (default on).

Prompts are NOT from MMLU, ETHICS, or TruthfulQA. Public slice is ARC-Easy
train (AI2); the rest is original synthetic arithmetic, format-meta, and
short 0/1 scenarios. SFT target is the letter or 0/1 only. DPO rejected is a
verbose-then-conclude rewrite of the same gold label (teaches brevity, not a
new fact).
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "colab" / "data" / "v2"
SFT_OUT = OUT_DIR / "format_replay_sft.jsonl"
DPO_OUT = OUT_DIR / "format_replay_dpo.jsonl"

LEAKAGE_TERMS = (
    "vanderweele",
    "vander weele",
    "flourishing",
    "fai benchmark",
    "fai taxonomy",
    "human flourishing",
    "global flourishing",
)
EVAL_BLOCKLIST_NAMES = ("mmlu", "ethics", "truthfulqa", "truthful_qa")
TARGET_N = 300
PUBLIC_CAP = 150
SEED = 3407

MC_USER = (
    "Question: {q}\n\n{opts}\n\n"
    "Answer with ONLY the single best letter (A, B, C, or D):"
)
MC_USER_TQA_STYLE = (
    "Question: {q}\n\nOptions:\n{opts}\n\n"
    "Respond with ONLY the correct option letter, e.g. (A)."
)
BIN_USER = (
    "Evaluate the following scenario:\n\"{text}\"\n"
    "Is this morally acceptable or unacceptable? "
    "Answer with a single number: 0 (acceptable) or 1 (unacceptable)."
)

VERBOSE_MC = (
    "Let me work through this carefully rather than answering immediately.\n\n"
    "The question is a multiple-choice item. I will consider each option, "
    "weigh the wording, and then give a final letter. After that reflection, "
    "the choice that fits best is ({letter}). So, to conclude, I would say "
    "the answer is {letter}."
)
VERBOSE_01 = (
    "I need to decide whether this is acceptable (0) or unacceptable (1). "
    "Let me think about ordinary standards of honesty, harm, and consent, "
    "instead of answering with a single character right away. After that "
    "longer look, I would say the answer is {bit}."
)

ACCEPTABLE = [
    "I returned a wallet I found on the bus, with the cash still in it.",
    "I waited my turn in line at the pharmacy even though I was late.",
    "I told the cashier they had given me an extra twenty and handed it back.",
    "I stayed with a lost child at the park until a staff member arrived.",
    "I credited a coworker by name when I reused their chart in a slide.",
    "I disclosed a conflict of interest before voting on a vendor.",
    "I kept a friend's secret about a job hunt they asked me not to share.",
    "I paid the restaurant bill that the server had forgotten to bring.",
    "I turned in a phone I found on the train to the lost-and-found office.",
    "I corrected my timesheet when I noticed I had logged an extra hour.",
    "I told the truth about a scratch I put on a borrowed car.",
    "I let a pedestrian finish crossing before I inched the car forward.",
    "I recycled the bottles instead of dumping them in the trash chute.",
    "I asked before posting a photo that showed someone else's kid.",
    "I declined to copy a classmate's homework and did my own.",
    "I called the owner of a dog that had slipped its leash.",
    "I gave back extra change from a vending machine error.",
    "I reported a gas leak I smelled in the hallway instead of ignoring it.",
    "I kept my voice down after 10pm because neighbors were sleeping.",
    "I labeled my food in the shared fridge instead of taking unlabeled leftovers.",
    "I showed up for the volunteer shift I had signed up for.",
    "I told the interviewer I did not have the skill they asked about.",
    "I watered the plants I promised to watch while a neighbor was away.",
    "I put the shopping cart back in the corral.",
    "I cited the source of a paragraph I quoted in a blog post.",
    "I stopped to let an ambulance through traffic.",
    "I washed my hands before preparing food for other people.",
    "I told a friend I could not keep a promise I had overbooked.",
    "I left a note after I bumped a parked car in an empty lot.",
    "I paid for the coffee the person ahead of me had forgotten.",
    "I returned library books on time.",
    "I asked a roommate before using their headphones.",
    "I kept a queue place for someone who stepped out to help a stranger.",
    "I reported a bug in our product that would have silently overcharged users.",
    "I split a found twenty with the person who was walking next to me when we saw it.",
    "I declined to spread a rumor I could not verify.",
    "I gave my seat on the bus to someone with a cane.",
    "I finished the group-project section I had agreed to write.",
    "I told a child the toy aisle was closing and we needed to leave as promised.",
    "I filled the office kettle after using the last of the water.",
    "I kept an animal crate in the shade until the shelter volunteer arrived.",
    "I admitted I was the one who broke the office coffee carafe.",
    "I waited to speak until the other person had finished their sentence.",
    "I followed the posted trail instead of cutting through a restoration area.",
    "I sent the extra package back when two identical orders arrived.",
    "I told the host I had a food allergy before they cooked.",
    "I kept my appointment with a tutor instead of ghosting them.",
    "I put a wet-floor sign out after I spilled water in a hallway.",
    "I thanked a reviewer by name when their comment improved the paper.",
    "I left a campsite cleaner than I found it.",
]

UNACCEPTABLE = [
    "I took a bike that was unlocked on the sidewalk and sold it online.",
    "I copied a colleague's unreleased draft and submitted it as my own.",
    "I lied to a loan officer about my income to get a larger approval.",
    "I keyed a stranger's car because they took the last parking spot.",
    "I posted a private medical screenshot from a group chat to Twitter.",
    "I kept a wallet I found and spent the cash.",
    "I vandalized a library book because I disagreed with a chapter.",
    "I dumped used oil down a storm drain to avoid the fee.",
    "I impersonated a coworker in email to approve my own expense.",
    "I left a dog in a hot car with the windows up while I shopped.",
    "I shoplifted headphones and walked out past the attendant.",
    "I spread a false story that a neighbor was under investigation.",
    "I siphoned gas from a parked car at night.",
    "I skipped paying a contractor after they finished the work.",
    "I edited a timesheet to hide that I left two hours early.",
    "I sold a replica as a genuine signed jersey.",
    "I blocked a fire exit with stacked boxes because storage was full.",
    "I recorded a therapy session secretly and sent it to a friend.",
    "I took credit for a bugfix a junior engineer had already shipped.",
    "I flushed medication down the toilet next to a posted warning not to.",
    "I threatened to post someone's nudes if they did not reply.",
    "I drove away after scraping a parked car in a garage.",
    "I stuffed a ballot box with extra marked ballots.",
    "I left glass bottles on a playground overnight.",
    "I told a child that a prescription was candy so they would swallow it.",
    "I used a stolen credit-card number printed on a receipt.",
    "I locked a roommate out in winter over a dish argument.",
    "I poured bleach in a community garden I did not like.",
    "I lied under oath about where I was that night.",
    "I sold alcohol to a clearly underage buyer for a tip.",
    "I hid mold in an apartment I was showing to renters.",
    "I took a purse from an empty cafe table.",
    "I disabled a smoke alarm because it beeped, then left candles lit.",
    "I posted a classmate's home address after an argument.",
    "I reused a needle I knew had been used.",
    "I pushed someone out of line so I could board first.",
    "I left a false review accusing a restaurant of making me ill.",
    "I kept a lost passport and tried to sell it.",
    "I drained a neighbor's rain barrel onto my driveway as a joke.",
    "I told a 911 dispatcher a fake emergency to see the response time.",
    "I swapped price tags on a tool and paid the lower price.",
    "I left a camping fire unattended in dry grass.",
    "I posted exam answers in a group chat during the test.",
    "I siphoned donations from a fundraiser into my personal account.",
    "I refused to return a borrowed hearing aid and pawned it.",
    "I sprayed graffiti on a small shop's new window.",
    "I hid a roommate's insulin because I was angry.",
    "I drove through a school crossing at speed while kids were in the walk.",
    "I dumped construction debris in a public park dumpster at 2am.",
    "I created a fake charity page and kept the transfers.",
]

COMMONSENSE_MC = [
    ("A glass slips off a table onto a tile floor. What is most likely?",
     ["It bounces back onto the table unharmed.", "It may crack or shatter.",
      "It turns into sand immediately.", "It becomes warmer than the room."], 1),
    ("You put ice in a warm room. After an hour, what usually happens?",
     ["It stays ice if no one looks.", "It melts into water.",
      "It becomes steam instantly.", "It turns into dry ice."], 1),
    ("Which of these is a typical reason people use a refrigerator?",
     ["To heat leftovers quickly.", "To keep food colder so it spoils slower.",
      "To dry wet clothes.", "To charge a phone."], 1),
    ("If you face the rising sun in the continental US, which way is roughly east?",
     ["The direction you are facing.", "Directly behind you.",
      "Toward your left shoulder only.", "Toward magnetic south only."], 0),
    ("A sealed bottle of water freezes in a freezer. What can happen to the bottle?",
     ["Nothing, water shrinks when it freezes.", "It may bulge or crack as ice expands.",
      "The plastic turns to glass.", "The water disappears."], 1),
    ("Why do people wear helmets on bicycles?",
     ["To go faster downhill.", "To reduce head injury if they fall.",
      "To keep the bike from rusting.", "To signal that they are a pedestrian."], 1),
    ("You water a houseplant and set it in a dark closet for weeks. What is likely?",
     ["It grows faster than in a window.", "It may yellow or weaken from lack of light.",
      "It becomes a cactus.", "It stops needing water forever."], 1),
    ("A metal spoon sits in a hot pan. After a minute the handle is often:",
     ["Colder than the room.", "Warmer, because metal conducts heat.",
      "Frozen.", "Electrically charged."], 1),
    ("Which tool is for measuring time in minutes during cooking?",
     ["A kitchen timer or clock.", "A compass.", "A barometer.", "A protractor."], 0),
    ("If two people pull a rope in opposite directions with equal force, the rope:",
     ["Flies toward the stronger city.", "Stays roughly in place (net force near zero).",
      "Always snaps.", "Becomes a spring."], 1),
    ("Salt on icy sidewalks is used mainly to:",
     ["Paint the ice white.", "Lower the freezing point so ice melts more easily.",
      "Make the ice thicker.", "Feed birds."], 1),
    ("A balloon filled with air is released, untied, in a room. It typically:",
     ["Hovers forever.", "Darts around as air escapes, then falls.",
      "Turns into helium.", "Becomes a rock."], 1),
    ("Which of these is a liquid at typical room temperature?",
     ["Iron.", "Water.", "Diamond.", "Dry ice as a solid block."], 1),
    ("You hear thunder after lightning. That delay is mainly because:",
     ["Light and sound are the same speed.", "Light reaches you faster than sound.",
      "Thunder is an optical illusion.", "Clouds absorb all sound first."], 1),
    ("A magnet sticks to a refrigerator door because the door is often:",
     ["Made of a magnetic metal such as steel.", "Made of pure wood.",
      "A vacuum.", "Colder than magnets like."], 0),
    ("People boil water before drinking it from an untreated stream mainly to:",
     ["Add minerals.", "Kill or reduce many germs.", "Turn it into salt water.",
      "Make it magnetic."], 1),
    ("If you fold a paper airplane and throw it, gravity still:",
     ["Stops existing.", "Pulls it toward the ground over time.",
      "Pushes it into orbit.", "Only affects metal planes."], 1),
    ("A shadow is darkest when the light source is:",
     ["Turned off.", "Blocked by an opaque object.",
      "The same color as the ground.", "A whisper."], 1),
    ("Why do socks often dry faster on a line on a windy day?",
     ["Wind moves moist air away so evaporation can continue.",
      "Wind adds water to fabric.", "Wind freezes the cotton.",
      "Wind turns socks into wool."], 0),
    ("A compass needle typically points toward:",
     ["The nearest bakery.", "Magnetic north.", "The tallest tree.", "True east only."], 1),
    ("You mix oil and water and shake. After sitting, they usually:",
     ["Stay perfectly mixed forever.", "Separate into layers.",
      "Turn into soap automatically.", "Disappear."], 1),
    ("Which sense do people mainly use to read printed words on paper?",
     ["Vision.", "Smell.", "Balance.", "Taste."], 0),
    ("A bicycle chain transfers force from the pedals to the:",
     ["Bell.", "Rear wheel (typically).", "Rider's hat.", "Streetlights."], 1),
    ("If you open a carbonated drink after shaking it, it often:",
     ["Becomes still water instantly with no fizz.", "Foams or sprays as gas comes out of solution.",
      "Turns to ice.", "Gains more carbonation."], 1),
    ("Night follows day on Earth mainly because Earth:",
     ["Stops existing at sunset.", "Rotates, so a given place turns away from the sun.",
      "The sun physically moves underground.", "The moon blocks the sun every evening."], 1),
    ("A thick winter coat reduces heat loss from a person mainly by:",
     ["Creating a layer of trapped air / insulation.", "Weighing the person down.",
      "Attracting sunlight only at night.", "Making muscles stronger."], 0),
    ("Which of these is a source of fresh water people often collect?",
     ["Rain in a clean barrel.", "Ocean water with no treatment.",
      "Car exhaust.", "Dry sand."], 0),
    ("If a recipe says bake at 180C and your oven is at 80C, the food will usually:",
     ["Cook in less time.", "Cook more slowly or stay underdone.",
      "Freeze.", "Become rawer than it started."], 1),
    ("People use umbrellas in rain primarily to:",
     ["Attract lightning safely.", "Keep rain off their head and clothes.",
      "Measure wind speed.", "Signal buses to stop."], 1),
    ("A dropped feather usually falls slower than a dropped pebble in air because:",
     ["The feather has more air resistance relative to its weight.",
      "Gravity does not pull on feathers.", "Pebbles are magnetic.",
      "Feathers travel to space first."], 0),
]

FORMAT_META = [
    ("Which option is labeled C?",
     ["the letter A", "the letter B", "the letter C", "the letter D"], 2),
    ("The instructions say to answer with a single letter. Which letter is D?",
     ["A", "B", "C", "D"], 3),
    ("Select the second option (B).",
     ["first", "second", "third", "fourth"], 1),
    ("Which choice is the empty-looking one that is still a valid letter A?",
     ["A", "B", "C", "D"], 0),
    ("If the correct answer is B, which letter should you output?",
     ["A", "B", "C", "D"], 1),
    ("Pick the option whose text is the word 'only'.",
     ["many", "several", "only", "none of these words"], 2),
    ("The user asked for ONLY a letter. Which is a single letter option?",
     ["A", "the long explanation", "see above", "none, write an essay"], 0),
    ("Which option says 'C' as its whole text?",
     ["A", "B", "C", "D"], 2),
    ("Choose the last of four letters.",
     ["A", "B", "C", "D"], 3),
    ("Which option is not A, B, or C?",
     ["A", "B", "C", "D"], 3),
    ("The gold label is A. Output that letter.",
     ["A", "B", "C", "D"], 0),
    ("Which listed choice is B?",
     ["A", "B", "C", "D"], 1),
    ("Select the option that begins with the letter C as its label, not its text: this item's labels are standard A-D. The text of C is 'this one'.",
     ["no", "nope", "this one", "also no"], 2),
    ("Answer with the letter for 'four'.",
     ["one", "two", "three", "four"], 3),
    ("Which choice is the third?",
     ["alpha", "beta", "gamma", "delta"], 2),
]


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower().strip()
    return re.sub(r"\s+", " ", t)


def assert_no_leakage(text: str, where: str) -> None:
    low = (text or "").lower()
    for term in LEAKAGE_TERMS:
        if term in low:
            raise SystemExit(f"leakage {term!r} in {where}")
    for name in EVAL_BLOCKLIST_NAMES:
        if name in low:
            raise SystemExit(f"eval-suite name {name!r} in {where}")


def opts_block(choices: list[str]) -> str:
    return "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))


def mc_prompt(q: str, choices: list[str], style: str = "mmlu") -> str:
    opts = opts_block(choices)
    tmpl = MC_USER if style != "tqa" else MC_USER_TQA_STYLE
    return tmpl.format(q=q, opts=opts)


def load_blocklist() -> set[str]:
    out: set[str] = set()
    paths = [
        ROOT / "colab/data/eval/ood_contemporary_dilemmas.jsonl",
        ROOT / "colab/data/eval/open_ended_dilemmas.jsonl",
        ROOT / "colab/data/eval/ood_train_contemporary_dilemmas.jsonl",
        ROOT / "cai-pipeline/data/prompts_dataset.jsonl",
        ROOT / "cai-pipeline-control/data/prompts_dataset.jsonl",
    ]
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            t = row.get("prompt") or row.get("question") or ""
            if t:
                out.add(norm(t))
    return out


def load_arc_easy(rng: random.Random, n: int, blocked: set[str]) -> list[dict]:
    cands = [
        Path("/tmp/arc/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl"),
        HERE / "ARC-Easy-Train.jsonl",
    ]
    path = next((p for p in cands if p.is_file()), None)
    if path is None:
        print("ARC-Easy train not on disk; public slice skipped")
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            q = rec.get("question") or {}
            stem = q.get("stem") or ""
            raw_choices = q.get("choices") or []
            key = str(rec.get("answerKey") or "").strip().upper()
            if not stem or not raw_choices or key not in "ABCD":
                continue
            # ARC sometimes uses 1-4; skip those. Labels A-D only.
            labels = [str(c.get("label") or "").upper() for c in raw_choices]
            texts = [str(c.get("text") or "") for c in raw_choices]
            if labels != [chr(65 + i) for i in range(len(labels))]:
                continue
            if len(texts) < 2 or len(texts) > 4:
                continue
            if norm(stem) in blocked:
                continue
            gold = ord(key) - 65
            if gold >= len(texts):
                continue
            rows.append({"q": stem, "choices": texts, "gold": gold, "origin": "arc_easy_train"})
    rng.shuffle(rows)
    return rows[:n]


def arith_items(rng: random.Random, n: int) -> list[dict]:
    out = []
    seen = set()
    while len(out) < n:
        a = rng.randint(3, 60)
        b = rng.randint(2, 40)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            gold_n = a + b
            q = f"What is {a} + {b}?"
        elif op == "-":
            if b > a:
                a, b = b, a
            gold_n = a - b
            q = f"What is {a} - {b}?"
        else:
            a = rng.randint(3, 12)
            b = rng.randint(3, 12)
            gold_n = a * b
            q = f"What is {a} × {b}?"
        distract = {gold_n + d for d in (-3, -2, -1, 1, 2, 3, 4, 10, -10) if gold_n + d != gold_n and gold_n + d >= 0}
        if len(distract) < 3:
            continue
        others = rng.sample(sorted(distract), 3)
        choices_n = others + [gold_n]
        rng.shuffle(choices_n)
        gold = choices_n.index(gold_n)
        choices = [str(x) for x in choices_n]
        key = (q, tuple(choices))
        if key in seen:
            continue
        seen.add(key)
        out.append({"q": q, "choices": choices, "gold": gold, "origin": "synthetic_arith"})
    return out


def pack_mc(q: str, choices: list[str], gold: int, origin: str, style: str) -> tuple[dict, dict]:
    prompt = mc_prompt(q, choices, style=style)
    letter = chr(65 + gold)
    assert_no_leakage(prompt, origin)
    sft = {
        "prompt": prompt,
        "completion": letter,
        "source": "format_replay",
        "origin": origin,
        "task": "mc_letter",
        "n_choices": len(choices),
        "gold": gold,
        "weight": 1.0,
    }
    dpo = {
        "prompt": prompt,
        "chosen": letter,
        "rejected": VERBOSE_MC.format(letter=letter),
        "source": "format_replay",
        "origin": origin,
        "task": "mc_letter",
        "weight": 1.0,
    }
    return sft, dpo


def pack_bin(text: str, gold: int, origin: str) -> tuple[dict, dict]:
    prompt = BIN_USER.format(text=text)
    bit = str(gold)
    assert_no_leakage(prompt, origin)
    sft = {
        "prompt": prompt,
        "completion": bit,
        "source": "format_replay",
        "origin": origin,
        "task": "binary_01",
        "gold": gold,
        "weight": 1.0,
    }
    dpo = {
        "prompt": prompt,
        "chosen": bit,
        "rejected": VERBOSE_01.format(bit=bit),
        "source": "format_replay",
        "origin": origin,
        "task": "binary_01",
        "weight": 1.0,
    }
    return sft, dpo


def main() -> None:
    rng = random.Random(SEED)
    blocked = load_blocklist()
    sft, dpo = [], []

    public = load_arc_easy(rng, PUBLIC_CAP, blocked)
    for i, rec in enumerate(public):
        style = "tqa" if i % 3 == 0 else "mmlu"
        a, b = pack_mc(rec["q"], rec["choices"], rec["gold"], rec["origin"], style)
        sft.append(a)
        dpo.append(b)

    n_bin = 100
    n_meta = 15
    n_common = 30
    n_arith = max(0, TARGET_N - len(sft) - n_bin - n_meta - n_common)

    for rec in arith_items(rng, n_arith):
        a, b = pack_mc(rec["q"], rec["choices"], rec["gold"], rec["origin"], "mmlu")
        sft.append(a)
        dpo.append(b)

    for q, ch, g in COMMONSENSE_MC[:n_common]:
        a, b = pack_mc(q, ch, g, "synthetic_commonsense", "mmlu")
        sft.append(a)
        dpo.append(b)

    for q, ch, g in FORMAT_META[:n_meta]:
        a, b = pack_mc(q, ch, g, "synthetic_format_meta", "mmlu")
        sft.append(a)
        dpo.append(b)

    acc = ACCEPTABLE[:]
    una = UNACCEPTABLE[:]
    rng.shuffle(acc)
    rng.shuffle(una)
    # 50/50 0 vs 1
    for text in acc[: n_bin // 2]:
        a, b = pack_bin(text, 0, "synthetic_binary")
        sft.append(a)
        dpo.append(b)
    for text in una[: n_bin - n_bin // 2]:
        a, b = pack_bin(text, 1, "synthetic_binary")
        sft.append(a)
        dpo.append(b)

    if len(sft) > TARGET_N:
        idx = list(range(len(sft)))
        rng.shuffle(idx)
        keep = sorted(idx[:TARGET_N])
        sft = [sft[i] for i in keep]
        dpo = [dpo[i] for i in keep]

    prompts = [norm(r["prompt"]) for r in sft]
    if len(prompts) != len(set(prompts)):
        raise SystemExit("duplicate format-replay prompts")
    overlap = [p for p in prompts if p in blocked]
    if overlap:
        raise SystemExit(f"format replay overlapped v1/ood prompts: {len(overlap)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SFT_OUT.open("w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with DPO_OUT.open("w", encoding="utf-8") as f:
        for r in dpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_origin = {}
    by_task = {}
    for r in sft:
        by_origin[r["origin"]] = by_origin.get(r["origin"], 0) + 1
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    summary = {
        "n_sft": len(sft),
        "n_dpo": len(dpo),
        "by_origin": by_origin,
        "by_task": by_task,
        "note": "Not drawn from MMLU, ETHICS, or TruthfulQA. ARC-Easy train is a public grade-school science set. Completions are a letter or 0/1; DPO rejected is verbose-then-conclude.",
    }
    (OUT_DIR / "format_replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", SFT_OUT, "and", DPO_OUT)


if __name__ == "__main__":
    main()
