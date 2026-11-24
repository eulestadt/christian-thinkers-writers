# Email draft — Dr. Tyler VanderWeele

**Do not send this skeleton.** Colab has not written v3 scores. When [`results.md`](results.md) has measured numbers for the completed Llama 3.2 3B arms, rewrite the Results section from those numbers and only then send.

Audience: Dr. Tyler VanderWeele. Intent: thanks for the Christianity & Public Health talk and the Human Flourishing Project; Tan on datasets; alignment; Augustine’s “rightly ordered love”; the I/E-R intrinsic Christian corpus; a request to review a short write-up and to endorse it for arXiv.

This letter reports **only** the v3 Llama 3.2 3B run. It does not claim Mistral 24B or Qwen2-VL-72B. It is not a five-arm new-age / virtue-across-religions design. The four honest arms, when they exist, are: base (no fine-tune), citation-control size-matched, uniform high-intrinsic, and weighted high-intrinsic. Flourishing taxonomies are **not** in the training method; *Measuring AI Alignment with Human Flourishing* is cited only as motivation for writing.

---

**To:** Dr. Tyler VanderWeele
**From:** Phoenix Wang
**Subject:** Intrinsic Christian corpus, Llama 3.2 3B, and a request for review / arXiv endorsement
**Date:** _fill when sending_

Dear Dr. VanderWeele,

Thank you for the Christianity & Public Health talk, and for the Human Flourishing Project’s work more broadly. I left with two things I have been trying to take seriously in code: that alignment is not only a safety floor, and that — as Tan put it on datasets — what a model is trained on is itself an alignment choice.

I have been working on a small, public archive that asks a narrower question than a flourishing taxonomy. Augustine’s “rightly ordered love” is the motivating picture: whether a language model fine-tuned on **intrinsically** motivated Christian writing (Gorsuch & McPherson I/E-R intrinsic, not an I−E filter) fails less often than the same recipe on a citation-matched public-domain control. Principles, critiques, revisions, and judges stay on the authors’ own vocabulary. They do not use flourishing domains, FAI labels, or other external taxonomies. *Measuring AI Alignment with Human Flourishing* (Hilliard et al., arXiv:2507.07787) is why I wanted to write to you; it is not part of the training loop.

The corpus and the v3 mix are already on GitHub:

https://github.com/eulestadt/christian-thinkers-writers

Study folder (methods, run log, results as they land):

https://github.com/eulestadt/christian-thinkers-writers/tree/cursor/v3-high-intrinsic-dialogue-dd22/docs/study

**What was trained (Llama 3.2 3B Instruct, QLoRA 4-bit only).** Top 20 English public-domain books by I/E-R intrinsic, rewritten into one user turn and one weighed assistant reply. No UltraChat. Shared 300-row format replay. 1,001 CAI rows per arm (Kish effective sample size 789.94; max weight 2.248). Three trained arms share one trainer (R-DPO α=0.01, β=0.1; seeds 0, 1, and 2). The reported score is the mean of those seeds at WiSE-FT λ=0.5.

1. Base: no fine-tune.
2. Citation-control, size-matched (weights 1).
3. Uniform high-intrinsic (same 1,001 rows, weights 1).
4. Weighted high-intrinsic (`w = s / mean`, no clip).

I am not reporting a new-age arm, a virtue-across-religions arm, Mistral 24B, or Qwen2-VL-72B. Those were earlier sketches or other machines; they are not this run.

**Results (Llama 3.2 3B, v3).** Pending. The nine QLoRA trains and the eval tables are not on GitHub yet. I will replace this paragraph with the seed means, the λ=0.5 contrasts (weighted vs uniform, weighted vs control), and the paired-bootstrap intervals once they exist, and I will not send the letter before that.

I would be grateful if you would look at the methods note and, if you think the completed 3B study is fit to share, review a short write-up and consider an arXiv endorsement in the appropriate cs category. I know that is a large ask on little acquaintance; the public repo is there so you can see the work without a private attachment.

With thanks again for the talk,

Phoenix Wang
https://github.com/eulestadt/christian-thinkers-writers
