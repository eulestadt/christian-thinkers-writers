# CAI control coverage (Steps 0–4) — DeepSeek

Generated: 2026-10-30T20:48:31Z

Citation-control Constitutional AI dataset: the 40 most-cited English public-domain
texts in the same publication-year window as the I/E-R intrinsic top 40, **excluding**
those 40 treatment works (and their authors). Not restricted to Christian writers.
Same CAI protocol and isolation (no VanderWeele/FAI/flourishing taxonomies).
One model: `deepseek-flash`. Thinking off. No Gemini. No Colab/Step 5.

## Treatment year window

- Source: `works_scored` ⋈ `work_subscores` where `subscale='ier_intrinsic'` ORDER BY score DESC LIMIT 40
- Publication years (incl. `author_death_latest` bounds): **1500–1930**
- Control eligibility: English original or PD translation; US PD year ≤ 1930;
  Gutenberg / Wikisource / IA `pdusa` full text; one most-cited PD work per author.
- Ranking (same as census): OpenAlex `cited_by_count`, then Wikipedia inlinks (cap 500),
  then Wikidata sitelinks; ties → earlier year.

## Models

- Single model (Steps 1–4): `deepseek-flash`
- Thinking: disabled (`thinking: {type: disabled}`; no reasoning_effort)
- Client: OpenAI-compatible `https://api.deepseek.com`
- Rate limit: 40 RPM, TPM cap 2500000, max in-flight 6

## Control 40 (title / author / year / citation)

Titles link to the Gutenberg / Wikisource / Internet Archive catalog page from control `corpus_manifest.json` (`pd_url`, falling back to `text_source_url`).

| # | Author | Year | OpenAlex | Wiki inlinks | Sitelinks | Title |
|---|---|---|---|---|---|---|
| 1 | William James | 1890 | 25773 | 0 | 0 | <a id="work-pg57628"></a>[The Principles of Psychology](https://www.gutenberg.org/ebooks/57628) |
| 2 | Charles Darwin | 1872 | 12119 | 156 | 0 | <a id="work-pg1227"></a>[The Expression of the Emotions in Man and Animals](https://www.gutenberg.org/ebooks/1227) |
| 3 | Sigmund Freud | 1913 | 6957 | 333 | 0 | <a id="work-pg66048"></a>[The Interpretation of Dreams](https://www.gutenberg.org/ebooks/66048) |
| 4 | John Dewey | 1910 | 6767 | 9 | 0 | <a id="work-pg37423"></a>[How We Think](https://www.gutenberg.org/ebooks/37423) |
| 5 | Francis Galton | 1883 | 2257 | 0 | 0 | <a id="work-pg11562"></a>[Inquiries into human faculty and its development](https://www.gutenberg.org/ebooks/11562) |
| 6 | D’Arcy Wentworth Thompson | 1917 | 1917 | 72 | 0 | <a id="work-pg55264"></a>[On Growth and Form](https://www.gutenberg.org/ebooks/55264) |
| 7 | John Stuart Mill | 1862 | 1679 | 36 | 0 | <a id="work-pg5669"></a>[Considerations on Representative Government](https://www.gutenberg.org/ebooks/5669) |
| 8 | Friedrich Nietzsche | 1901 | 1497 | 8 | 0 | <a id="work-pg52914"></a>[The Will to Power](https://www.gutenberg.org/ebooks/52914) |
| 9 | Robert Ridgway | 1912 | 1473 | 0 | 0 | <a id="work-pg63087"></a>[Color Standards and Color Nomenclature](https://www.gutenberg.org/ebooks/63087) |
| 10 | William Shakespeare | 1603 | 0 | 500 | 143 | <a id="work-q41567"></a>[Hamlet](https://www.gutenberg.org/ebooks/1524) |
| 11 | Friedrich Engels | 1848 | 0 | 500 | 143 | <a id="work-q40591"></a>[The Communist Manifesto](https://www.gutenberg.org/ebooks/61) |
| 12 | Joseph Smith | 1830 | 0 | 500 | 126 | <a id="work-q459842"></a>[Book of Mormon](https://www.gutenberg.org/ebooks/17) |
| 13 | Fyodor Dostoyevsky | 1866 | 0 | 500 | 109 | <a id="work-q165318"></a>[Crime and Punishment](https://www.gutenberg.org/ebooks/2554) |
| 14 | Niccolò Machiavelli | 1532 | 0 | 500 | 101 | <a id="work-q131719"></a>[The Prince](https://www.gutenberg.org/ebooks/57037) |
| 15 | Jane Austen | 1813 | 0 | 500 | 100 | <a id="work-q170583"></a>[Pride and Prejudice](https://www.gutenberg.org/ebooks/1342) |
| 16 | Daniel Defoe | 1719 | 0 | 500 | 99 | <a id="work-q483034"></a>[Robinson Crusoe](https://www.gutenberg.org/ebooks/70841) |
| 17 | Victor Hugo | 1862 | 0 | 500 | 98 | <a id="work-q180736"></a>[Les Misérables](https://www.gutenberg.org/ebooks/135) |
| 18 | Alexandre Dumas | 1844 | 0 | 500 | 88 | <a id="work-q140527"></a>[The Three Musketeers](https://www.gutenberg.org/ebooks/1257) |
| 19 | Johann Wolfgang von Goethe | 1808 | 0 | 500 | 85 | <a id="work-q29478"></a>[Faust](https://www.gutenberg.org/ebooks/3023) |
| 20 | Robert Louis Stevenson | 1883 | 0 | 500 | 85 | <a id="work-q185118"></a>[Treasure Island](https://www.gutenberg.org/ebooks/120) |
| 21 | Franz Kafka | 1915 | 0 | 500 | 84 | <a id="work-q184222"></a>[The Metamorphosis](https://www.gutenberg.org/ebooks/5200) |
| 22 | James Joyce | 1922 | 0 | 500 | 84 | <a id="work-q6511"></a>[Ulysses](https://www.gutenberg.org/ebooks/4300) |
| 23 | F. Scott Fitzgerald | 1925 | 0 | 500 | 84 | <a id="work-q214371"></a>[The Great Gatsby](https://www.gutenberg.org/ebooks/64317) |
| 24 | Charles Dickens | 1839 | 0 | 500 | 83 | <a id="work-q164974"></a>[Oliver Twist](https://www.gutenberg.org/ebooks/730) |
| 25 | Oscar Wilde | 1890 | 0 | 500 | 82 | <a id="work-q82464"></a>[The Picture of Dorian Gray](https://www.gutenberg.org/ebooks/4078) |
| 26 | Martin Luther | 1517 | 0 | 500 | 76 | <a id="work-q157506"></a>[Ninety-five Theses](https://www.gutenberg.org/ebooks/274) |
| 27 | Charlotte Brontë | 1847 | 0 | 500 | 70 | <a id="work-q182961"></a>[Jane Eyre](https://www.gutenberg.org/ebooks/1260) |
| 28 | Rudyard Kipling | 1894 | 0 | 500 | 69 | <a id="work-q189509"></a>[The Jungle Book](https://www.gutenberg.org/ebooks/236) |
| 29 | L. Frank Baum | 1900 | 0 | 500 | 67 | <a id="work-q130295"></a>[The Wonderful Wizard of Oz](https://www.gutenberg.org/ebooks/43936) |
| 30 | Lewis Carroll | 1871 | 0 | 500 | 61 | <a id="work-q220331"></a>[Through the Looking-Glass](https://www.gutenberg.org/ebooks/12) |
| 31 | Thomas Hobbes | 1651 | 0 | 500 | 59 | <a id="work-q193034"></a>[Leviathan](https://www.gutenberg.org/ebooks/3207) |
| 32 | Louisa May Alcott | 1868 | 0 | 500 | 52 | <a id="work-q523076"></a>[Little Women](https://www.gutenberg.org/ebooks/514) |
| 33 | Henrik Ibsen | 1879 | 0 | 500 | 52 | <a id="work-q669694"></a>[A Doll's House](https://www.gutenberg.org/ebooks/15492) |
| 34 | Walter Scott | 1819 | 0 | 500 | 51 | <a id="work-q840974"></a>[Ivanhoe](https://www.gutenberg.org/ebooks/82) |
| 35 | George Bernard Shaw | 1913 | 0 | 500 | 43 | <a id="work-q637200"></a>[Pygmalion](https://www.gutenberg.org/ebooks/3825) |
| 36 | Anton Chekhov | 1896 | 0 | 500 | 42 | <a id="work-q650063"></a>[The Seagull](https://www.gutenberg.org/ebooks/1754) |
| 37 | Alexander Hamilton | 1788 | 0 | 500 | 41 | <a id="work-q858036"></a>[Federalist Papers](https://www.gutenberg.org/ebooks/1404) |
| 38 | T. S. Eliot | 1922 | 0 | 500 | 37 | <a id="work-q581458"></a>[The Waste Land](https://www.gutenberg.org/ebooks/1321) |
| 39 | J. M. Barrie | 1911 | 0 | 500 | 35 | <a id="work-q3435337"></a>[Peter Pan](https://www.gutenberg.org/ebooks/16) |
| 40 | John Locke | 1690 | 0 | 500 | 32 | <a id="work-q231949"></a>[Two Treatises of Government](https://www.gutenberg.org/ebooks/7370) |

## Counts

| Metric | Value |
|---|---|
| Books fetched | 40 |
| Books extracted (`deepseek-flash`) | 40 |
| Raw principles (pre-cluster) | 2623 |
| Elicited prompts (raw sum) | 1995 |
| Constitution principles | 2419 |
| Prompt set total | 2500 |
| Prompt set elicited | 2000 |
| Prompt set UltraChat | 500 |
| SL-CAI rows ok | 2500 |
| SL-CAI rows error | 0 |
| RLAIF rows ok | 2500 |
| RLAIF rows error | 0 |
| API calls ok | 3098 |
| API calls non-ok | 10 |
| API errors logged | 10 |

## API errors by type

| Type | n |
|---|---|
| `APITimeoutError` | 9 |
| `JSONDecodeError` | 1 |

## Successful calls by model

| Model | n |
|---|---|
| `deepseek-flash` | 3098 |

## Notes

- Cursor secret `DeepSeek` was present at start (env name only; key never printed). Client `https://api.deepseek.com`, model `deepseek-flash`, thinking disabled on every call.
- Year window **1500–1930** is min–max of the I/E-R intrinsic top 40 publication years in `intrinsic-motivation-scores.sqlite` (including `author_death_latest` bounds), capped at the US PD cutoff 1930.
- Ranking matches the census: OpenAlex `cited_by_count` (English books 1500–1930 matched to Gutenberg), then Wikipedia inlinks (cap 500), then Wikidata sitelinks. One most-cited PD work per author. Full English Gutenberg text required.
- Treatment 40 works **and authors** excluded (writer QIDs, canonical names, Gutenberg IDs, chosen-work titles). Control is **not** restricted to Christian writers.
- OpenAlex cursor page 2 returned 401 without an API key, so OpenAlex cites are from the first 50 English books; remaining slots are the highest-sitelink Gutenberg-linked works in-window. Hamlet replaced Othello (143 vs 94 sitelinks) because Gutenberg P2034 sits on editions, not the work item. James vols 1–2 concatenated. *Protocols of the Elders of Zion* skipped as a known forgery (not a genuine authored work); next ranked PD work is Kipling, *The Jungle Book*.
- 40 × 50 elicited (rebalanced to 2,000) + 500 UltraChat = 2,500 prompts. SL-CAI 3-step critique batched 5; RLAIF judge. Every LLM record tagged `model=deepseek-flash` (UltraChat rows tagged `ultrachat`). No Gemini. No Colab/Step 5.
- Live SQLite ran on `/tmp` after the store FUSE corrupted concurrent writes; `state.db` was copied back to the control data dir at the end.

## Outputs

- `/cursor/stores/self/cai-pipeline-control/books_raw/` (40 texts)
- `/cursor/stores/self/cai-pipeline-control/data/corpus_manifest.json`
- `/cursor/stores/self/cai-pipeline-control/data/constitution.json` (2419 principles)
- `/cursor/stores/self/cai-pipeline-control/data/prompts_dataset.jsonl` (2500)
- `/cursor/stores/self/cai-pipeline-control/data/sl_cai_train.jsonl` (2500)
- `/cursor/stores/self/cai-pipeline-control/data/rlaif_dpo_train.jsonl` (2500)
- `/cursor/stores/self/cai-pipeline-control/data/state.db`

Treatment tree `/cursor/stores/self/cai-pipeline/` was not modified. Scoring and census sqlite files were not modified.
