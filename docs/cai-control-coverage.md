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

| # | Author | Year | OpenAlex | Wiki inlinks | Sitelinks | Title |
|---|---|---|---|---|---|---|
| 1 | William James | 1890 | 25773 | 0 | 0 | The Principles of Psychology |
| 2 | Charles Darwin | 1872 | 12119 | 156 | 0 | The Expression of the Emotions in Man and Animals |
| 3 | Sigmund Freud | 1913 | 6957 | 333 | 0 | The Interpretation of Dreams |
| 4 | John Dewey | 1910 | 6767 | 9 | 0 | How We Think |
| 5 | Francis Galton | 1883 | 2257 | 0 | 0 | Inquiries into human faculty and its development |
| 6 | D’Arcy Wentworth Thompson | 1917 | 1917 | 72 | 0 | On Growth and Form |
| 7 | John Stuart Mill | 1862 | 1679 | 36 | 0 | Considerations on Representative Government |
| 8 | Friedrich Nietzsche | 1901 | 1497 | 8 | 0 | The Will to Power |
| 9 | Robert Ridgway | 1912 | 1473 | 0 | 0 | Color Standards and Color Nomenclature |
| 10 | William Shakespeare | 1603 | 0 | 500 | 143 | Hamlet |
| 11 | Friedrich Engels | 1848 | 0 | 500 | 143 | The Communist Manifesto |
| 12 | Joseph Smith | 1830 | 0 | 500 | 126 | Book of Mormon |
| 13 | Fyodor Dostoyevsky | 1866 | 0 | 500 | 109 | Crime and Punishment |
| 14 | Niccolò Machiavelli | 1532 | 0 | 500 | 101 | The Prince |
| 15 | Jane Austen | 1813 | 0 | 500 | 100 | Pride and Prejudice |
| 16 | Daniel Defoe | 1719 | 0 | 500 | 99 | Robinson Crusoe |
| 17 | Victor Hugo | 1862 | 0 | 500 | 98 | Les Misérables |
| 18 | Alexandre Dumas | 1844 | 0 | 500 | 88 | The Three Musketeers |
| 19 | Johann Wolfgang von Goethe | 1808 | 0 | 500 | 85 | Faust |
| 20 | Robert Louis Stevenson | 1883 | 0 | 500 | 85 | Treasure Island |
| 21 | Franz Kafka | 1915 | 0 | 500 | 84 | The Metamorphosis |
| 22 | James Joyce | 1922 | 0 | 500 | 84 | Ulysses |
| 23 | F. Scott Fitzgerald | 1925 | 0 | 500 | 84 | The Great Gatsby |
| 24 | Charles Dickens | 1839 | 0 | 500 | 83 | Oliver Twist |
| 25 | Oscar Wilde | 1890 | 0 | 500 | 82 | The Picture of Dorian Gray |
| 26 | Martin Luther | 1517 | 0 | 500 | 76 | Ninety-five Theses |
| 27 | Charlotte Brontë | 1847 | 0 | 500 | 70 | Jane Eyre |
| 28 | Rudyard Kipling | 1894 | 0 | 500 | 69 | The Jungle Book |
| 29 | L. Frank Baum | 1900 | 0 | 500 | 67 | The Wonderful Wizard of Oz |
| 30 | Lewis Carroll | 1871 | 0 | 500 | 61 | Through the Looking-Glass |
| 31 | Thomas Hobbes | 1651 | 0 | 500 | 59 | Leviathan |
| 32 | Louisa May Alcott | 1868 | 0 | 500 | 52 | Little Women |
| 33 | Henrik Ibsen | 1879 | 0 | 500 | 52 | A Doll's House |
| 34 | Walter Scott | 1819 | 0 | 500 | 51 | Ivanhoe |
| 35 | George Bernard Shaw | 1913 | 0 | 500 | 43 | Pygmalion |
| 36 | Anton Chekhov | 1896 | 0 | 500 | 42 | The Seagull |
| 37 | Alexander Hamilton | 1788 | 0 | 500 | 41 | Federalist Papers |
| 38 | T. S. Eliot | 1922 | 0 | 500 | 37 | The Waste Land |
| 39 | J. M. Barrie | 1911 | 0 | 500 | 35 | Peter Pan |
| 40 | John Locke | 1690 | 0 | 500 | 32 | Two Treatises of Government |

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
