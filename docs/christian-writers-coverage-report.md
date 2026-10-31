# Christian writers database — coverage report

Generated: 2026-10-27T17:29:53Z
Source root: https://en.wikipedia.org/wiki/Category:Christian_writers
Method: MediaWiki `categorymembers` (ns=0|14, cmlimit=500, continue pagination, maxlag=5), then Wikipedia `query` extracts/pageprops and revision wikitext, then Wikidata `wbgetentities` + SPARQL (P50 author, P800 notable work).

## Counts

| Metric | Value |
|---|---|
| Subcategories visited | 977 |
| Category-member page rows (pre-dedup) | 25552 |
| Unique main-namespace pages in the tree | 15766 |
| Pages classified as writers (after QID/title dedup) | 14128 |
| Pages excluded (lists/works/dabs/non-person/sports leakage) | 4627 |
| Works (deduplicated title+year per writer) | 163063 |
| Writers with 0 works listed | 4804 |
| Writers with works_status=fetch_failed | 0 |
| API errors logged | 0 |

## Category tree (first level under Category:Christian writers)

- Category:Barnabas: 0 subcats, 15 direct pages
- Category:Christian bloggers: 1 subcats, 34 direct pages
- Category:Christian devotional writers: 4 subcats, 2 direct pages
- Category:Christian hagiographers: 2 subcats, 108 direct pages
- Category:Christian music songwriters: 1 subcats, 166 direct pages
- Category:Christian novelists: 0 subcats, 145 direct pages
- Category:Christian poets: 6 subcats, 94 direct pages
- Category:Christian scholars: 7 subcats, 39 direct pages
- Category:Christian writers by denomination: 10 subcats, 0 direct pages
- Category:Christian writers by nationality: 17 subcats, 0 direct pages
- Category:Christian writers by subject area: 3 subcats, 0 direct pages
- Category:Church Fathers: 6 subcats, 93 direct pages
- Category:Mark the Evangelist: 2 subcats, 9 direct pages

Total categories in tree: **977**. Full list is in crawl state `visited_categories.json`. Per-category sizes: `categories.jsonl`.

The Wikipedia category graph under person-articles (e.g. John Milton, St. Bonaventure, Pope Gregory I) includes paintings, films, sports teams, and knighthoods. Those pages were **walked** (full graph) then **excluded** from the writers table unless they sat in a writer-ish category.

## Method notes

1. Recursively walked every subcategory under Category:Christian_writers. Cycles skipped via a visited-set on category titles.
2. Administrative namespaces excluded (Template, Wikipedia, Portal, File, Module, Help, Draft, User). Main-namespace people and list-of-works/bibliography articles were kept; list pages were harvested for linked people and works, then not emitted as writers.
3. Writers appearing in multiple categories were deduplicated by Wikidata QID (else resolved Wikipedia title); all leaf memberships were retained. JSONL `category_paths` stores up to 8 representative root-to-leaf paths (shortest first); SQLite `writer_categories` stores one row per leaf category.
4. Works came from Wikipedia sections (Works, Selected works, Publications, Bibliography, List of works, books/poems/sermons/hymns/letters, infobox notableworks) plus 22 linked Bibliography-of / List-of-works pages that actually exist, plus Wikidata P50 and P800. Titles were **not** invented; missing stays empty with `works_status=none_listed`.
5. Duplicate works merged by normalized title+year.
6. Rate limits: maxlag=5, modest concurrency on wikitext, retries on 429/503.

## Known gaps

- 0 writers have no Wikidata id (Wikipedia-only).
- 543 writers have no birth/death/floruit dates.
- 306 writers sit in a *stubs* Wikipedia category (short articles; bibliographies often empty).
- Works coverage depends on English Wikipedia bibliography sections and Wikidata P50/P800. Many living authors and stubs list no works. 4804 writers have works=[].
- Non-English Wikipedia bibliographies were not crawled.
- Conventional unlinked titles like “Bibliography of X” were not mass-fetched (most do not exist); only bibliography pages **wikilinked from the writer article** were followed.
- Songwriters’ individual songs appear only if listed as works/hymns on the page or Wikidata.
- SPARQL P50 can attach editions/translations as separate works for prolific authors (e.g. Henryk Sienkiewicz); they were kept when Wikidata listed them, then title+year deduped.
- List-of-writers pages contributed extra candidate titles; non-persons were filtered via Wikidata P31 / biography heuristics.

## Writers with no works found (sample of 80 of 4804)

- A. Arthur Schiller
- A. C. Solomon Raj
- A. de Herz
- A. Donald Macleod
- A. E. Inbanathan
- A. E. P. Albert
- A. James Reimer
- A. Laurence Lyon
- A. M. Cagle
- A. N. Gopal
- Aba I
- Abbott Hall Brisbane
- Aberdeen doctors
- Abraham Calovius
- Abraham Dirk Loman
- Abraham Ecchellensis
- Abraham H. Cannon
- Abraham Hinckelmann
- Abraham Kuenen
- Abraham of Scetes
- Abraham Ruchat
- Abraham Scultetus
- Abraham Trommius
- Abraham von Franckenberg
- Abram Bowman Kolb
- Absalon Pederssøn Beyer
- Abu Bishr Matta ibn Yunus
- Acacius of Beroea
- Acca of Hexham
- Achille Gagliardi
- Achille Vander Heeren
- Ada Jane Blenkhorn
- Adalbert Ebner
- Adam Abramowicz
- Adam Boreel
- Adam Burley
- Adam Burski
- Adam C. Smyth
- Adam Christian Agricola
- Adam Franz Lennig
- Adam Krupski
- Adam of Barking
- Adam of Bockenfield
- Adam of Damerham
- Adam of Usk
- Adam Pulchrae Mulieris
- Adam Reißner
- Adam Russell Taylor
- Adam the Welshman
- Adam Wielomski
- Adelaide
- Adelheid Ruck-Schröder
- Ademar Agostinho Sauthier
- Adolf Keller
- Adolf Köberle
- Adolf Paul Johannes Althaus
- Adolf Wuttke
- Adolfo Pérez Esquivel
- Adolphe Gesché
- Adolphe Monod
- Adrian of Canterbury
- Adriano Caprioli
- Adrianus Johannes Simonis
- Adrien Arcand
- Aeneas of Gaza
- Aerius of Sebaste
- Afeosemime Adogame
- Afonso Mendes
- Agathon
- Agnes Maxwell
- Agostino Bernal
- Agostino Oreggi
- Agricol Lozano
- Agrippa Castor
- Agustín Antolínez
- Agustín de Cazalla
- Ailean a' Ridse MacDhòmhnaill
- Aino Kapewangolo
- AJ Gil
- Al Reimer

Full zero-works list: `/tmp/christian-writers-crawl/state/writers_zero_works.json`

## API errors (summary)

0 errors. See `/tmp/christian-writers-crawl/state/api_errors.jsonl` and `/cursor/stores/self/internal/wikipedia-crawl-log.md`.

## How to resume

State directory: `/tmp/christian-writers-crawl/state`

Phases write checkpoints:
- `visited_categories.json` / `category_queue.json` / `phase1_done.json` — category tree
- `page_memberships.jsonl` — page↔category edges
- `page_meta.jsonl` — Wikipedia extracts and pageprops
- `wikidata_entities.jsonl` — Wikidata records
- `wd_works.jsonl` — SPARQL works
- `wiki_works.jsonl` / `wikitext/` — parsed Wikipedia bibliographies
- `export/` — pre-slim dump; `export-slim/` — published files

Re-run: `python3 /tmp/christian-writers-crawl/crawl.py` (also copied to `/cursor/stores/self/internal/wikipedia_crawler.py`).

The crawler skips completed phases/records. Delete a checkpoint file to redo that phase.

## Deliverable files

- `/cursor/stores/self/docs/christian-writers-database.jsonl`
- `/cursor/stores/self/docs/christian-writers-database.csv`
- `/cursor/stores/self/docs/christian-writers-works.csv`
- `/cursor/stores/self/docs/christian-writers.sqlite`
- `/cursor/stores/self/docs/christian-writers-coverage-report.md`
