# Christian Thinkers Writers — project context

Phoenix Wang is building a corpus and fine-tuning experiments to test whether historic Christian writing, especially intrinsically motivated virtue/moral theology, can improve AI alignment. The near-term academic contact is Dr. Tyler VanderWeele (Human Flourishing; Christianity & Public Health). Phoenix attended that talk and wants a nomination/review path toward arXiv, citing *Measuring AI Alignment with Human Flourishing*.

## Research question

Does aligning language models with a Christian, intrinsically motivated virtue ethic — “rightly ordered love” (Augustine) and related tradition — reduce alignment failures relative to generic, new-age, or cross-religious virtue corpora?

Related questions Phoenix has already named:

- What is the correct ordering of love, and would aligning AI with virtue reduce alignment issues?
- Do intrinsically motivated texts outperform extrinsically motivated texts?
- Can architecture or training that draws on how humans are spiritually ordered (analogy: attention residuals drawing on human cognition) help with lab-level alignment pauses and public worry?
- Prefer sources that build on prior work (e.g. virtue ethics), not fundamentalist “others are wrong, I am right” polemics.

Fine-tuning data should draw on David Williams’ work on intrinsically motivated motivations.

## Early experiment sketch (rough, already run)

Alignment buckets Phoenix described:

0. Control — no fine-tuning  
1. Control — general fine-tuning bucket  
2. Experimental — general new-age wisdom literature (Phoenix does not consider this a proper control, because those texts still reflect truths that come through Jesus)  
3. Experimental — virtue ethics across religions  
4. Experimental — intrinsically motivated virtue/moral Christian theology  

Open-weight models already tried (overnight, ~$200 compute, agentic workflow):

- Llama 3.2B Instruct with QLoRA (4-bit)
- Mistral Small 24B Instruct with QLoRA (4-bit) + gradient checkpointing
- Qwen2-VL-72B

Future work named: frontier open-weight models (e.g. DeepSeek R1), which needs more compute.

Collecting fine-tuning material is the bottleneck: (1) identify texts, (2) align models to that material.

## Corpus census (first engineering task)

Complete Wikipedia category walk of [Category:Christian writers](https://en.wikipedia.org/wiki/Category:Christian_writers), including every nested subcategory, plus each writer’s published works. Completeness is the bar: do not miss a writer or a listed work. This census is the inventory for later source selection, not itself the fine-tuning mix.
