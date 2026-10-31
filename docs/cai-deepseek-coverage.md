# CAI coverage (Steps 0–4) — DeepSeek

Generated: 2026-10-28T10:37:53Z

DeepSeek-only constitution / SL-CAI / RLAIF data build over the I/E-R intrinsic top 40.
No Colab, GPU, Unsloth, or Drive. Ranked by I/E-R intrinsic only (no I−E filter).
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

| # | Author | Year | I/E-R score | Wiki inlinks | Sitelinks | Title |
|---|---|---|---|---|---|---|
| 1 | Brother Lawrence | 1692 | 0.00656683 | 52 | 0 | The Practice of the Presence of God |
| 2 | William Henry Griffith Thomas | 1914 | 0.00616133 | 0 | 0 | The Prayers of St. Paul |
| 3 | Edward McKendree Bounds | 1907 | 0.00533195 | 0 | 0 | Preacher and Prayer |
| 4 | Lilian Staveley | 1920 | 0.00437728 | 0 | 0 | The Romance of the Soul |
| 5 | Albertus Magnus | 1911 | 0.00347414 | 0 | 0 | On Union with God |
| 6 | George Adam Smith | 1896 | 0.00301618 | 0 | 0 | Four Psalms: XXIII, XXXVI, LII, CXXI, Interpreted for Practical Use |
| 7 | Augustus Hopkins Strong | 1907 | 0.00296311 | 0 | 0 | Systematic Theology: The Doctrine of God (1907) |
| 8 | Ignatius of Loyola | 1548 | 0.00254304 | 290 | 0 | Spiritual Exercises |
| 9 | Legh Richmond | 1827 | 0.00228953 | 10 | 0 | The Dairyman's Daughter |
| 10 | Francis de Sales | 1884 | 0.00221390 | 0 | 0 | Maxims and Counsels of St. Francis de Sales for Every Day in the Year |
| 11 | Francis John McConnell | 1917 | 0.00219616 | 0 | 0 | Understanding the Scriptures |
| 12 | Rufus Jones | 1906 | 0.00216327 | 0 | 0 | The Double Search: Studies in Atonement and Prayer |
| 13 | Cynewulf | 1904 | 0.00197948 | 0 | 0 | The Elene of Cynewulf |
| 14 | Arthur Tappan Pierson | 1899 | 0.00197408 | 0 | 0 | George Muller of Bristol and his Witness to a Prayer-Hearing God |
| 15 | Francis Thompson | 1893 | 0.00187751 | 14 | 0 | The Hound of Heaven |
| 16 | Rosalind Goforth | 1921 | 0.00182846 | 0 | 0 | How I Know God Answers Prayer |
| 17 | Joseph Bates | 1846 | 0.00179022 | 0 | 0 | The Opening Heavens Or, A Connected View of the Testimony of the Prophets and Apostles (1846) |
| 18 | Catherine Booth | 1881 | 0.00176755 | 0 | 0 | Godliness (1881) |
| 19 | Hastings Rashdall | 1910 | 0.00167333 | 0 | 0 | Philosophy and Religion |
| 20 | John Taylor | 1852 | 0.00166878 | 0 | 0 | The Government of God (1852) |
| 21 | Adoniram Judson Gordon | 1895 | 0.00165476 | 0 | 0 | The Ministry of the Spirit |
| 22 | Albert Benjamin Simpson | 1897 | 0.00163270 | 0 | 0 | Days of Heaven Upon Earth: A Year Book of Scripture Texts and Living Truths |
| 23 | Mary Baker Eddy | 1875 | 0.00163189 | 0 | 8 | Science and Health with Key to the Scriptures |
| 24 | William Law | 1901 | 0.00157235 | 0 | 0 | The Supersensual Life |
| 25 | William Blake | 1794 | 0.00154316 | 252 | 0 | Songs of Innocence and of Experience |
| 26 | Julian of Norwich | 1500 | 0.00151306 | 170 | 0 | Revelations of Divine Love |
| 27 | William Temple | 1915 | 0.00147097 | 0 | 0 | Church and Nation |
| 28 | Hudson Taylor | 1894 | 0.00141046 | 0 | 0 | A Retrospect |
| 29 | Alphonsus Maria de Liguori | 1750 | 0.00140897 | 157 | 0 | The Glories of Mary |
| 30 | Henry Wace | 1917 | 0.00139980 | 0 | 0 | The War and the Gospel: Sermons & Addresses During the Present War |
| 31 | Kuno Meyer | 1911 | 0.00139774 | 0 | 0 | Selections from Ancient Irish Poetry |
| 32 | Hannah T. King | 1872 | 0.00138234 | 500 | 0 | Brigham Young |
| 33 | John Gresham Machen | 1915 | 0.00133723 | 0 | 0 | The Literature and History of New Testament Times |
| 34 | Alban Butler | 1773 | 0.00127607 | 0 | 1 | The Lives of the Fathers, Martyrs, and Other Principal Saints |
| 35 | John Keble | 1827 | 0.00126680 | 14 | 0 | The Christian Year |
| 36 | Ellet J. Waggoner | 1900 | 0.00126294 | 0 | 0 | The Glad Tidings (1900 Original) |
| 37 | Arthur Shearly Cripps | 1900 | 0.00124379 | 0 | 0 | Primavera: Poems by Four Authors |
| 38 | Joseph F. Smith | 1919 | 0.00124032 | 0 | 0 | Gospel Doctrine: Selections from the Sermons and Writings of Joseph F. Smith (1919) |
| 39 | Adolf von Harnack | 1930 | 0.00124005 | 0 | 0 | The History of Dogma |
| 40 | Charles Gore | 1889 | 0.00123591 | 18 | 0 | Lux Mundi |

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
- DeepSeek-only Steps 0–4 complete. Step 5 (Colab/GPU/Unsloth/Drive) was not run.

## Outputs

- `/cursor/stores/self/cai-pipeline/data/corpus_manifest.json`
- `/cursor/stores/self/cai-pipeline/data/raw_principles.jsonl`
- `/cursor/stores/self/cai-pipeline/data/constitution.json`
- `/cursor/stores/self/cai-pipeline/data/prompts_dataset.jsonl`
- `/cursor/stores/self/cai-pipeline/data/sl_cai_train.jsonl`
- `/cursor/stores/self/cai-pipeline/data/rlaif_dpo_train.jsonl`
- `/cursor/stores/self/cai-pipeline/data/state.db`
