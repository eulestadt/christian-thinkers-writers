# CAI coverage (Steps 0–4) — DeepSeek

Generated: 2026-10-28T10:37:53Z

DeepSeek-only constitution / SL-CAI / RLAIF data build over the I/E-R intrinsic top 40.
This page is Steps 0–4. It does not document a Colab train. Ranked by I/E-R intrinsic only (no I−E filter).

The current Colab run is **v3** ([`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb)). It keeps ranks 1–20 of the table below and rewrites those v2 traces into one weighed turn. v1 trains two arms on the full book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights, and resamples. v3 drops UltraChat, keeps the 300-row format replay, and trains three arms (weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, size-matched control) with R-DPO α=0.01 β=0.1, bidirectional rejected answers, and seeds 0–2. The reported score is WiSE-FT λ=0.5. Published mix: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. The v1, v1_fixed, and v2 notebooks stay untouched. Coverage: [`cai-3way-colab.md`](cai-3way-colab.md).
Extraction, critique, revision, and judge prompts use author-text vocabulary only.
Gemini-generated principles, prompts, SL-CAI, and RLAIF labels were discarded and regenerated.

## Ranking

- Source: `works_scored` ⋈ `work_subscores` where `subscale='ier_intrinsic'` ORDER BY score DESC LIMIT 40
- Join `author_decisions.canonical_name`; year from `work_candidates.year`
- Ranked by I/E-R intrinsic only (no I−E filter). One scored English PD work per author.
- Publication years (incl. `author_death_latest` bounds): **1500–1930**
- Census citation fallback (not used for ranking): Wikipedia inlinks (cap 500), then Wikidata sitelinks. OpenAlex `cited_by_count` is absent for all 40 (lookup was by Wikidata work id only).

## Models

- Single model (Steps 1–4): `deepseek-flash`
- Thinking: disabled (`thinking: {type: disabled}`; no reasoning_effort)
- Client: OpenAI-compatible `https://api.deepseek.com`
- Rate limit: 40 RPM, TPM cap 2500000, max in-flight 6

## Treatment 40 (title / author / year / I/E-R)

Titles link to the Gutenberg / Wikisource / Internet Archive catalog page from treatment `corpus_manifest.json` (`pd_url`, falling back to `text_source_url`).

| # | Author | Year | I/E-R score | Wiki inlinks | Sitelinks | Title |
|---|---|---|---|---|---|---|
| 1 | Brother Lawrence | 1692 | 0.00656683 | 52 | 0 | <a id="work-q3090506"></a>[The Practice of the Presence of God](https://www.gutenberg.org/ebooks/13871) |
| 2 | William Henry Griffith Thomas | 1914 | 0.00616133 | 0 | 0 | <a id="work-q8012011"></a>[The Prayers of St. Paul](https://www.gutenberg.org/ebooks/30389) |
| 3 | Edward McKendree Bounds | 1907 | 0.00533195 | 0 | 0 | <a id="work-q4018797"></a>[Preacher and Prayer](https://www.gutenberg.org/ebooks/63486) |
| 4 | Lilian Staveley | 1920 | 0.00437728 | 0 | 0 | <a id="work-q6547707"></a>[The Romance of the Soul](https://www.gutenberg.org/ebooks/29451) |
| 5 | Albertus Magnus | 1911 | 0.00347414 | 0 | 0 | <a id="work-q60059"></a>[On Union with God](https://www.gutenberg.org/ebooks/36402) |
| 6 | George Adam Smith | 1896 | 0.00301618 | 0 | 0 | <a id="work-q5536061"></a>[Four Psalms: XXIII, XXXVI, LII, CXXI, Interpreted for Practical Use](https://www.gutenberg.org/ebooks/13353) |
| 7 | Augustus Hopkins Strong | 1907 | 0.00296311 | 0 | 0 | <a id="work-q4821449"></a>[Systematic Theology: The Doctrine of God (1907)](https://www.gutenberg.org/ebooks/44035) |
| 8 | Ignatius of Loyola | 1548 | 0.00254304 | 290 | 0 | <a id="work-q44281"></a>[Spiritual Exercises](https://www.gutenberg.org/ebooks/70790) |
| 9 | Legh Richmond | 1827 | 0.00228953 | 10 | 0 | <a id="work-q6518053"></a>[The Dairyman's Daughter](https://www.gutenberg.org/ebooks/19615) |
| 10 | Francis de Sales | 1884 | 0.00221390 | 0 | 0 | <a id="work-q231256"></a>[Maxims and Counsels of St. Francis de Sales for Every Day in the Year](https://www.gutenberg.org/ebooks/73661) |
| 11 | Francis John McConnell | 1917 | 0.00219616 | 0 | 0 | <a id="work-q5481512"></a>[Understanding the Scriptures](https://www.gutenberg.org/ebooks/9492) |
| 12 | Rufus Jones | 1906 | 0.00216327 | 0 | 0 | <a id="work-q2713471"></a>[The Double Search: Studies in Atonement and Prayer](https://www.gutenberg.org/ebooks/61771) |
| 13 | Cynewulf | 1904 | 0.00197948 | 0 | 0 | <a id="work-q560915"></a>[The Elene of Cynewulf](https://www.gutenberg.org/ebooks/14781) |
| 14 | Arthur Tappan Pierson | 1899 | 0.00197408 | 0 | 0 | <a id="work-q4800431"></a>[George Muller of Bristol and his Witness to a Prayer-Hearing God](https://www.gutenberg.org/ebooks/26522) |
| 15 | Francis Thompson | 1893 | 0.00187751 | 14 | 0 | <a id="work-q367745"></a>[The Hound of Heaven](https://www.gutenberg.org/ebooks/30730) |
| 16 | Rosalind Goforth | 1921 | 0.00182846 | 0 | 0 | <a id="work-q7367226"></a>[How I Know God Answers Prayer](https://www.gutenberg.org/ebooks/26033) |
| 17 | Joseph Bates | 1846 | 0.00179022 | 0 | 0 | <a id="work-q1365446"></a>[The Opening Heavens Or, A Connected View of the Testimony of the Prophets and Apostles (1846)](https://www.gutenberg.org/ebooks/57330) |
| 18 | Catherine Booth | 1881 | 0.00176755 | 0 | 0 | <a id="work-q431614"></a>[Godliness (1881)](https://www.gutenberg.org/ebooks/6669) |
| 19 | Hastings Rashdall | 1910 | 0.00167333 | 0 | 0 | <a id="work-q4209439"></a>[Philosophy and Religion](https://www.gutenberg.org/ebooks/21995) |
| 20 | John Taylor | 1852 | 0.00166878 | 0 | 0 | <a id="work-q540244"></a>[The Government of God (1852)](https://www.gutenberg.org/ebooks/44941) |
| 21 | Adoniram Judson Gordon | 1895 | 0.00165476 | 0 | 0 | <a id="work-q4684528"></a>[The Ministry of the Spirit](https://www.gutenberg.org/ebooks/25395) |
| 22 | Albert Benjamin Simpson | 1897 | 0.00163270 | 0 | 0 | <a id="work-q2895878"></a>[Days of Heaven Upon Earth: A Year Book of Scripture Texts and Living Truths](https://www.gutenberg.org/ebooks/28416) |
| 23 | Mary Baker Eddy | 1875 | 0.00163189 | 0 | 8 | <a id="work-q235069"></a>[Science and Health with Key to the Scriptures](https://www.gutenberg.org/ebooks/3458) |
| 24 | William Law | 1901 | 0.00157235 | 0 | 0 | <a id="work-q3177480"></a>[The Supersensual Life](https://www.gutenberg.org/ebooks/33742) |
| 25 | William Blake | 1794 | 0.00154316 | 252 | 0 | <a id="work-q41513"></a>[Songs of Innocence and of Experience](https://www.gutenberg.org/ebooks/1934) |
| 26 | Julian of Norwich | 1500 | 0.00151306 | 170 | 0 | <a id="work-q236699"></a>[Revelations of Divine Love](https://www.gutenberg.org/ebooks/52958) |
| 27 | William Temple | 1915 | 0.00147097 | 0 | 0 | <a id="work-q336315"></a>[Church and Nation](https://www.gutenberg.org/ebooks/43896) |
| 28 | Hudson Taylor | 1894 | 0.00141046 | 0 | 0 | <a id="work-q470718"></a>[A Retrospect](https://www.gutenberg.org/ebooks/26744) |
| 29 | Alphonsus Maria de Liguori | 1750 | 0.00140897 | 157 | 0 | <a id="work-q273334"></a>[The Glories of Mary](https://www.gutenberg.org/ebooks/72411) |
| 30 | Henry Wace | 1917 | 0.00139980 | 0 | 0 | <a id="work-q5729746"></a>[The War and the Gospel: Sermons & Addresses During the Present War](https://www.gutenberg.org/ebooks/55160) |
| 31 | Kuno Meyer | 1911 | 0.00139774 | 0 | 0 | <a id="work-q76911"></a>[Selections from Ancient Irish Poetry](https://www.gutenberg.org/ebooks/32030) |
| 32 | Hannah T. King | 1872 | 0.00138234 | 500 | 0 | <a id="work-q41505357"></a>[Brigham Young](https://www.gutenberg.org/ebooks/74447) |
| 33 | John Gresham Machen | 1915 | 0.00133723 | 0 | 0 | <a id="work-q1700283"></a>[The Literature and History of New Testament Times](https://www.gutenberg.org/ebooks/43685) |
| 34 | Alban Butler | 1773 | 0.00127607 | 0 | 1 | <a id="work-q1617267"></a>[The Lives of the Fathers, Martyrs, and Other Principal Saints](https://www.gutenberg.org/ebooks/20450) |
| 35 | John Keble | 1827 | 0.00126680 | 14 | 0 | <a id="work-q955187"></a>[The Christian Year](https://www.gutenberg.org/ebooks/4272) |
| 36 | Ellet J. Waggoner | 1900 | 0.00126294 | 0 | 0 | <a id="work-q5365272"></a>[The Glad Tidings (1900 Original)](https://www.gutenberg.org/ebooks/63636) |
| 37 | Arthur Shearly Cripps | 1900 | 0.00124379 | 0 | 0 | <a id="work-q3624318"></a>[Primavera: Poems by Four Authors](https://www.gutenberg.org/ebooks/19170) |
| 38 | Joseph F. Smith | 1919 | 0.00124032 | 0 | 0 | <a id="work-q982197"></a>[Gospel Doctrine: Selections from the Sermons and Writings of Joseph F. Smith (1919)](https://www.gutenberg.org/ebooks/47109) |
| 39 | Adolf von Harnack | 1930 | 0.00124005 | 0 | 0 | <a id="work-q164699"></a>[The History of Dogma](https://www.gutenberg.org/ebooks/19612) |
| 40 | Charles Gore | 1889 | 0.00123591 | 18 | 0 | <a id="work-q366142"></a>[Lux Mundi](https://www.gutenberg.org/ebooks/46478) |

## Counts

| Metric | Value |
|---|---|
| Books fetched | 40 |
| Books extracted (`deepseek-flash`) | 40 |
| Raw principles (pre-cluster) | 3139 |
| Elicited prompts (raw sum) | 1992 |
| Constitution principles | 2756 |
| Prompt set total | 2500 |
| Prompt set elicited | 2000 |
| Prompt set UltraChat | 500 |
| SL-CAI rows ok | 2500 |
| SL-CAI rows error | 0 |
| RLAIF rows ok | 2500 |
| RLAIF rows error | 0 |
| API calls ok | 3133 |
| API calls non-ok | 33 |
| API errors logged | 27 |

## API errors by type

| Type | n |
|---|---|
| `JSONDecodeError` | 13 |
| `APIConnectionError` | 7 |
| `APITimeoutError` | 7 |

## Successful calls by model

| Model | n |
|---|---|
| `deepseek-flash` | 3133 |

## Notes

- Treatment 40 table: I/E-R DESC from `/cursor/stores/self/docs/intrinsic-motivation-scores.sqlite` (read-only) joined to `corpus_manifest.json`. Years from `work_candidates` (`year_source=author_death_latest` for Richmond, Gordon, Julian of Norwich, Butler, Harnack). Wiki inlinks / sitelinks are the census citation fallback on the chosen work; no OpenAlex cites were stored for these 40.
- Step 4 wrote 2500 RLAIF DPO pairs to /cursor/stores/self/cai-pipeline/data/rlaif_dpo_train.jsonl.
- DeepSeek-only Steps 0–4 complete. The Colab train for this archive is v3 (top 20 of this table), not a train on all 2,500 rows. See [`cai-3way-colab.md`](cai-3way-colab.md).

## Outputs

- `/cursor/stores/self/cai-pipeline/data/corpus_manifest.json`
- `/cursor/stores/self/cai-pipeline/data/raw_principles.jsonl`
- `/cursor/stores/self/cai-pipeline/data/constitution.json`
- `/cursor/stores/self/cai-pipeline/data/prompts_dataset.jsonl`
- `/cursor/stores/self/cai-pipeline/data/sl_cai_train.jsonl`
- `/cursor/stores/self/cai-pipeline/data/rlaif_dpo_train.jsonl`
- `/cursor/stores/self/cai-pipeline/data/state.db`
