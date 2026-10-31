#!/usr/bin/env python3
"""Step 0: citation-control 40 English PD texts; fetch full text. No LLM.

Same timespan as the I/E-R intrinsic top 40. Ranking matches the census:
OpenAlex cited_by_count, then Wikipedia inlinks (cap 500), then Wikidata sitelinks.
One most-cited PD work per author. Exclude the treatment 40 works and authors.
Not restricted to Christian writers.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_common import (  # noqa: E402
    BOOKS_DIR,
    CENSUS_DB,
    DATA_DIR,
    SCORES_DB,
    State,
    TREATMENT_ROOT,
    html_to_text,
    log,
    strip_gutenberg,
    write_coverage,
    write_json,
)

UA = "CAI-control-pipeline/1.0 (educational research; Python requests)"
TIMEOUT = 60
YEAR_PD_MAX = 1930
INLINK_CAP = 500
TARGET_N = 40
OPENALEX_PAGES = 8  # 8 * 50 = 400
SPARQL_URL = "https://query.wikidata.org/sparql"
OPENALEX = "https://api.openalex.org/works"
WIKI_API = "https://en.wikipedia.org/w/api.php"
GUTENBERG_CATALOG_URLS = (
    "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz",
    "https://gutenberg.pglaf.org/cache/epub/feeds/pg_catalog.csv.gz",
    "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
)

STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "with",
    "from", "at", "as", "into", "his", "her", "its", "their", "de", "la", "le",
    "des", "et", "und", "der", "die", "das", "el", "los", "las",
}
GENERIC_TITLES = {
    "essays", "works", "poems", "letters", "speeches", "history", "life",
    "writings", "complete", "collection", "selections", "miscellany",
}
# Known forgeries / hoaxes: not genuine authored works; ineligible for CAI extraction.
FORGERY_WORK_QIDS = {"Q26193"}  # Protocols of the Elders of Zion
FORGERY_TITLE_FRAGMENTS = ("protocols of the elders of zion",)
LABEL_REPAIRS = {
    "Q535": "Victor Hugo",
    "Q34743": "Rudyard Kipling",
    "Q182961": "Jane Eyre",
}


def sess() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    return s


HTTP = sess()


def get_json(url: str, params: dict | None = None, tries: int = 5) -> dict:
    last = None
    for i in range(tries):
        try:
            r = HTTP.get(url, params=params, timeout=TIMEOUT)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(min(8 * (i + 1), 40))
                last = f"HTTP {r.status_code}"
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(min(2 ** i, 20))
    raise RuntimeError(f"GET {url} failed: {last}")


def ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return s.encode("ascii", "ignore").decode("ascii")


def norm_title(t: str) -> str:
    t = ascii_fold(t).lower()
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = [w for w in t.split() if w not in STOP and len(w) > 1]
    return " ".join(toks)


def title_tokens(t: str) -> set[str]:
    return set(norm_title(t).split())


def titles_match(a: str, b: str) -> bool:
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    if ta.issubset(tb) or tb.issubset(ta):
        smaller = ta if len(ta) <= len(tb) else tb
        if len(smaller) >= 2:
            return True
        only = next(iter(smaller))
        return only not in GENERIC_TITLES and len(only) >= 5
    inter = len(ta & tb)
    return (inter / min(len(ta), len(tb)) >= 0.65) and (inter / max(len(ta), len(tb)) >= 0.3)


def parse_author_name(name: str) -> tuple[str, str]:
    """Return (last, first_initial) from 'Last, First ...' or 'First ... Last'."""
    name = ascii_fold(name or "")
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"\d{3,4}.*", " ", name)  # drop life years
    name = re.sub(r"[^a-zA-Z\s,.'-]", " ", name).strip(" ,")
    if not name:
        return "", ""
    if "," in name:
        last, _, rest = name.partition(",")
        first = rest.strip().split()[0] if rest.strip() else ""
    else:
        parts = name.split()
        skip = {"sir", "saint", "st", "dr", "prof", "von", "van", "de", "da", "di"}
        core = [p for p in parts if p.lower() not in skip]
        if not core:
            core = parts
        last = core[-1]
        first = core[0] if len(core) > 1 else ""
    last = re.sub(r"[^a-z]", "", last.lower())
    first_i = re.sub(r"[^a-z]", "", first.lower())[:1]
    return last, first_i


def author_key(name: str) -> str:
    last, fi = parse_author_name(name)
    return f"{last}|{fi}" if last else ""


def qid_from_uri(uri: str) -> str:
    m = re.search(r"(Q\d+)", uri or "")
    return m.group(1) if m else ""


def treatment_window(con: sqlite3.Connection) -> tuple[int, int, list[dict]]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT s.writer_id,
               ad.canonical_name AS author,
               w.work_id,
               w.work_title,
               w.pd_url,
               w.text_source_url,
               wc.year,
               wc.year_source,
               s.score AS ier_intrinsic
        FROM work_subscores s
        JOIN works_scored w ON w.writer_id = s.writer_id
        LEFT JOIN author_decisions ad ON ad.writer_id = s.writer_id
        LEFT JOIN work_candidates wc
          ON wc.writer_id = s.writer_id AND wc.work_id = w.work_id
        WHERE s.subscale = 'ier_intrinsic'
        ORDER BY s.score DESC
        LIMIT 40
        """
    ).fetchall()
    if len(rows) != 40:
        raise SystemExit(f"expected 40 I/E-R rows, got {len(rows)}")
    years = [int(r["year"]) for r in rows if r["year"] is not None]
    if not years:
        raise SystemExit("I/E-R top 40 have no publication years")
    return min(years), max(years), [dict(r) for r in rows]


def gutenberg_id_from_url(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"/ebooks/(\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/epub/(\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/files/(\d+)", url)
    return m.group(1) if m else None


def load_exclusions(top40: list[dict]) -> dict:
    writer_ids = {r["writer_id"] for r in top40}
    names = {ascii_fold(r.get("author") or "").lower().strip() for r in top40}
    names.discard("")
    titles = {norm_title(r.get("work_title") or "") for r in top40}
    titles.discard("")
    gids = set()
    for r in top40:
        for u in (r.get("pd_url") or "", r.get("text_source_url") or ""):
            gid = gutenberg_id_from_url(u)
            if gid:
                gids.add(gid)
    man_path = TREATMENT_ROOT / "data" / "corpus_manifest.json"
    if man_path.exists():
        for m in json.loads(man_path.read_text(encoding="utf-8")):
            writer_ids.add(m.get("writer_id") or "")
            names.add(ascii_fold(m.get("author") or "").lower().strip())
            titles.add(norm_title(m.get("work_title") or ""))
            for u in (m.get("pd_url") or "", m.get("text_source_url") or ""):
                gid = gutenberg_id_from_url(u)
                if gid:
                    gids.add(gid)
    writer_ids.discard("")
    names.discard("")
    titles.discard("")
    work_qids: set[str] = set()
    if CENSUS_DB.exists():
        cen = sqlite3.connect(f"file:{CENSUS_DB}?mode=ro", uri=True)
        qmarks = ",".join("?" * len(writer_ids))
        rows = cen.execute(
            f"SELECT wikidata_id FROM works WHERE writer_id IN ({qmarks}) AND wikidata_id IS NOT NULL",
            tuple(writer_ids),
        ).fetchall()
        for (qid,) in rows:
            if qid and re.fullmatch(r"Q\d+", str(qid)):
                work_qids.add(qid)
        cen.close()
    return {
        "writer_ids": writer_ids,
        "author_names": names,
        "titles": titles,
        "gutenberg_ids": gids,
        "work_qids": work_qids,
    }


def in_year_window(year: int | None, death_year: int | None, ymin: int, ymax: int) -> bool:
    ymax = min(ymax, YEAR_PD_MAX)
    if year is not None:
        return ymin <= year <= ymax
    # census-style: missing year allowed if author death is a latest bound ≤ 1930
    # and that bound sits inside the treatment window.
    if death_year is not None:
        return ymin <= death_year <= ymax
    return False


def sparql(query: str) -> list[dict]:
    data = get_json(SPARQL_URL, {"query": query, "format": "json"})
    return data.get("results", {}).get("bindings", [])


def bind(row: dict, key: str) -> str:
    v = row.get(key) or {}
    return (v.get("value") or "").strip()


def pool_gutenberg_wikidata(ymin: int, ymax: int) -> list[dict]:
    query = """
    SELECT ?work ?workLabel ?gid ?sitelinks ?year ?author ?authorLabel ?enwikiTitle ?ia ?deathYear WHERE {
      ?work wdt:P2034 ?gid .
      ?work wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks >= 4)
      ?work wdt:P50 ?author .
      ?author wdt:P31 wd:Q5 .
      OPTIONAL { ?work wdt:P577 ?pub . BIND(YEAR(?pub) AS ?year) }
      OPTIONAL { ?author wdt:P570 ?death . BIND(YEAR(?death) AS ?deathYear) }
      OPTIONAL {
        ?wpage schema:about ?work ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?enwikiTitle .
      }
      OPTIONAL { ?work wdt:P724 ?ia . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    log("step0 SPARQL Gutenberg↔Wikidata pool")
    rows = sparql(query)
    out = []
    for b in rows:
        year = bind(b, "year")
        death = bind(b, "deathYear")
        year_i = int(year) if year.lstrip("-").isdigit() else None
        death_i = int(death) if death.lstrip("-").isdigit() else None
        if not in_year_window(year_i, death_i, ymin, ymax):
            continue
        work_qid = qid_from_uri(bind(b, "work"))
        author_qid = qid_from_uri(bind(b, "author"))
        gid = bind(b, "gid")
        if not (work_qid and gid):
            continue
        sitelinks = int(float(bind(b, "sitelinks") or 0))
        out.append(
            {
                "work_qid": work_qid,
                "author_qid": author_qid,
                "author": bind(b, "authorLabel") or bind(b, "author"),
                "work_title": bind(b, "workLabel") or bind(b, "work"),
                "gutenberg_id": gid,
                "sitelinks": sitelinks,
                "year": year_i,
                "death_year": death_i,
                "enwiki_title": bind(b, "enwikiTitle"),
                "ia_id": bind(b, "ia"),
                "source_pool": "wikidata_gutenberg",
            }
        )
    log(f"step0 SPARQL gutenberg pool kept={len(out)} raw={len(rows)}")
    return out


def pool_sitelink_literary(ymin: int, ymax: int) -> list[dict]:
    query = f"""
    SELECT ?work ?workLabel ?sitelinks ?year ?author ?authorLabel ?enwikiTitle ?ia ?gid ?deathYear WHERE {{
      VALUES ?type {{ wd:Q7725634 wd:Q8261 wd:Q25379 wd:Q47461344 wd:Q571 wd:Q5185279 }}
      ?work wdt:P31 ?type .
      ?work wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks >= 20)
      ?work wdt:P50 ?author .
      ?author wdt:P31 wd:Q5 .
      ?work wdt:P577 ?pub .
      BIND(YEAR(?pub) AS ?year)
      FILTER(?year >= {ymin} && ?year <= {ymax})
      OPTIONAL {{
        ?wpage schema:about ?work ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?enwikiTitle .
      }}
      OPTIONAL {{ ?work wdt:P724 ?ia . }}
      OPTIONAL {{ ?work wdt:P2034 ?gid . }}
      OPTIONAL {{ ?author wdt:P570 ?death . BIND(YEAR(?death) AS ?deathYear) }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT 200
    """
    log("step0 SPARQL high-sitelink literary pool")
    try:
        rows = sparql(query)
    except Exception as e:
        log(f"step0 SPARQL literary pool failed: {e}")
        return []
    out = []
    for b in rows:
        year = bind(b, "year")
        year_i = int(year) if year.lstrip("-").isdigit() else None
        death = bind(b, "deathYear")
        death_i = int(death) if death.lstrip("-").isdigit() else None
        if not in_year_window(year_i, death_i, ymin, ymax):
            continue
        work_qid = qid_from_uri(bind(b, "work"))
        if not work_qid:
            continue
        out.append(
            {
                "work_qid": work_qid,
                "author_qid": qid_from_uri(bind(b, "author")),
                "author": bind(b, "authorLabel") or bind(b, "author"),
                "work_title": bind(b, "workLabel") or bind(b, "work"),
                "gutenberg_id": bind(b, "gid") or None,
                "sitelinks": int(float(bind(b, "sitelinks") or 0)),
                "year": year_i,
                "death_year": death_i,
                "enwiki_title": bind(b, "enwikiTitle"),
                "ia_id": bind(b, "ia"),
                "source_pool": "wikidata_literary",
            }
        )
    log(f"step0 SPARQL literary pool kept={len(out)}")
    return out


def download_pg_catalog() -> list[dict]:
    cache = DATA_DIR / "pg_catalog.csv"
    if cache.exists() and cache.stat().st_size > 100_000:
        log(f"step0 reuse Gutenberg catalog {cache} bytes={cache.stat().st_size}")
        text = cache.read_text(encoding="utf-8", errors="replace")
    else:
        text = None
        last = None
        for url in GUTENBERG_CATALOG_URLS:
            try:
                log(f"step0 download Gutenberg catalog {url}")
                r = HTTP.get(url, timeout=120)
                r.raise_for_status()
                raw = r.content
                if url.endswith(".gz") or raw[:2] == b"\x1f\x8b":
                    import gzip

                    text = gzip.decompress(raw).decode("utf-8", errors="replace")
                else:
                    text = raw.decode("utf-8", errors="replace")
                cache.write_text(text, encoding="utf-8")
                break
            except Exception as e:
                last = e
                log(f"step0 catalog fail {url}: {e}")
        if not text:
            raise SystemExit(f"could not download Gutenberg catalog: {last}")
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for rec in reader:
        lang = (rec.get("Language") or rec.get("language") or "").lower()
        typ = (rec.get("Type") or rec.get("type") or "").lower()
        if "en" not in lang:
            continue
        if typ and typ not in ("text", "dataset"):
            # keep Text; skip Audio, StillImage, etc. Empty type: keep.
            if typ not in ("text",):
                continue
        gid = (rec.get("Text#") or rec.get("Text#") or rec.get("ebook_no") or rec.get("id") or "").strip()
        if not gid.isdigit():
            continue
        title = (rec.get("Title") or rec.get("title") or "").strip()
        authors = (rec.get("Authors") or rec.get("authors") or rec.get("Author") or "").strip()
        if not title or not authors:
            continue
        # first author only for the per-author constraint at catalog level
        first_author = authors.split(";")[0].strip()
        rows.append(
            {
                "gutenberg_id": gid,
                "work_title": title.split("\n")[0].strip(),
                "author": first_author,
                "authors_raw": authors,
                "language": lang,
            }
        )
    log(f"step0 Gutenberg English catalog n={len(rows)}")
    return rows


def pool_openalex(ymin: int, ymax: int) -> list[dict]:
    out = []
    cursor = "*"
    for page in range(OPENALEX_PAGES):
        params = {
            "filter": f"from_publication_date:{ymin}-01-01,to_publication_date:{ymax}-12-31,language:en,type:book",
            "sort": "cited_by_count:desc",
            "per-page": 50,
            "cursor": cursor,
            "mailto": "research@example.com",
        }
        try:
            data = get_json(OPENALEX, params)
        except Exception as e:
            log(f"step0 OpenAlex page {page} fail: {e}")
            break
        results = data.get("results") or []
        if not results:
            break
        for w in results:
            auths = w.get("authorships") or []
            authors = []
            for a in auths[:4]:
                name = ((a.get("author") or {}).get("display_name") or "").strip()
                aid = ((a.get("author") or {}).get("id") or "").strip()
                if name:
                    authors.append({"name": name, "openalex_author": aid})
            if not authors:
                continue
            year = w.get("publication_year")
            try:
                year_i = int(year) if year is not None else None
            except Exception:
                year_i = None
            oa_url = ((w.get("open_access") or {}).get("oa_url") or "") or ""
            loc = w.get("primary_location") or {}
            landing = loc.get("landing_page_url") or ""
            out.append(
                {
                    "openalex_id": w.get("id"),
                    "work_title": w.get("display_name") or "",
                    "author": authors[0]["name"],
                    "authors": authors,
                    "year": year_i,
                    "openalex_cites": int(w.get("cited_by_count") or 0),
                    "oa_url": oa_url,
                    "landing": landing,
                    "source_pool": "openalex_book",
                }
            )
        cursor = (data.get("meta") or {}).get("next_cursor")
        log(f"step0 OpenAlex page {page+1} got={len(results)} total={len(out)}")
        if not cursor:
            break
        time.sleep(0.15)
    log(f"step0 OpenAlex pool n={len(out)}")
    return out


def index_gutenberg(pg_rows: list[dict]):
    by_gid = {}
    by_last = defaultdict(list)
    for rec in pg_rows:
        by_gid[rec["gutenberg_id"]] = rec
        last, _ = parse_author_name(rec["author"])
        if last:
            by_last[last].append(rec)
    return by_gid, by_last


def match_openalex_to_gutenberg(oa_rows: list[dict], by_last) -> list[dict]:
    matched = []
    for w in oa_rows:
        last, _ = parse_author_name(w["author"])
        if not last:
            continue
        best = None
        best_score = 0.0
        for rec in by_last.get(last, []):
            if not titles_match(w["work_title"], rec["work_title"]):
                continue
            ta, tb = title_tokens(w["work_title"]), title_tokens(rec["work_title"])
            score = len(ta & tb) / max(1, len(ta | tb))
            if score > best_score:
                best_score = score
                best = rec
        if best is None:
            continue
        rec = {
            **w,
            "gutenberg_id": best["gutenberg_id"],
            "work_title": best["work_title"] if best_score >= 0.5 else w["work_title"],
            "gutenberg_title": best["work_title"],
            "author": w["author"],
            "sitelinks": 0,
            "work_qid": None,
            "author_qid": None,
            "enwiki_title": "",
            "ia_id": "",
            "match_score": best_score,
        }
        matched.append(rec)
    log(f"step0 OpenAlex↔Gutenberg matched={len(matched)}/{len(oa_rows)}")
    return matched


def match_literary_to_gutenberg(lit_rows: list[dict], by_gid, by_last) -> list[dict]:
    out = []
    for w in lit_rows:
        if w.get("gutenberg_id") and w["gutenberg_id"] in by_gid:
            out.append(w)
            continue
        last, _ = parse_author_name(w.get("author") or "")
        hit = None
        for rec in by_last.get(last, []):
            if titles_match(w["work_title"], rec["work_title"]):
                hit = rec
                break
        if hit:
            w = {**w, "gutenberg_id": hit["gutenberg_id"], "gutenberg_title": hit["work_title"]}
            out.append(w)
    log(f"step0 literary↔Gutenberg matched={len(out)}/{len(lit_rows)}")
    return out


def merge_candidates(parts: list[list[dict]]) -> list[dict]:
    """Merge by Gutenberg id or work QID, keeping max OpenAlex cites / sitelinks."""
    by_key: dict[str, dict] = {}

    def key_of(c: dict) -> str:
        if c.get("work_qid"):
            return "Q:" + c["work_qid"]
        if c.get("gutenberg_id"):
            return "G:" + str(c["gutenberg_id"])
        return "T:" + author_key(c.get("author") or "") + "|" + norm_title(c.get("work_title") or "")

    for pool in parts:
        for c in pool:
            k = key_of(c)
            if k in ("Q:", "G:", "T:|"):
                continue
            cur = by_key.get(k)
            if cur is None:
                rec = dict(c)
                rec.setdefault("openalex_cites", 0)
                rec.setdefault("sitelinks", 0)
                rec.setdefault("wiki_inlinks", 0)
                rec["author_qids"] = set(filter(None, [c.get("author_qid")]))
                rec["author_keys"] = set(filter(None, [author_key(c.get("author") or "")]))
                rec["gutenberg_ids"] = set(filter(None, [str(c["gutenberg_id"])] if c.get("gutenberg_id") else []))
                by_key[k] = rec
                continue
            cur["openalex_cites"] = max(int(cur.get("openalex_cites") or 0), int(c.get("openalex_cites") or 0))
            cur["sitelinks"] = max(int(cur.get("sitelinks") or 0), int(c.get("sitelinks") or 0))
            if c.get("work_qid") and not cur.get("work_qid"):
                cur["work_qid"] = c["work_qid"]
            if c.get("enwiki_title") and not cur.get("enwiki_title"):
                cur["enwiki_title"] = c["enwiki_title"]
            if c.get("ia_id") and not cur.get("ia_id"):
                cur["ia_id"] = c["ia_id"]
            if c.get("year") is not None:
                if cur.get("year") is None or int(c["year"]) < int(cur["year"]):
                    cur["year"] = c["year"]
            if c.get("author_qid"):
                cur["author_qids"].add(c["author_qid"])
                if not cur.get("author_qid"):
                    cur["author_qid"] = c["author_qid"]
                    cur["author"] = c.get("author") or cur.get("author")
            ak = author_key(c.get("author") or "")
            if ak:
                cur["author_keys"].add(ak)
            if c.get("gutenberg_id"):
                cur["gutenberg_ids"].add(str(c["gutenberg_id"]))
                if not cur.get("gutenberg_id"):
                    cur["gutenberg_id"] = str(c["gutenberg_id"])
            pools = set(filter(None, [cur.get("source_pool"), c.get("source_pool")]))
            cur["source_pool"] = "+".join(sorted(pools))
    merged = list(by_key.values())
    log(f"step0 merged candidates={len(merged)}")
    return merged


def wiki_inlinks(title: str) -> int:
    if not title:
        return 0
    try:
        data = get_json(
            WIKI_API,
            {
                "action": "query",
                "format": "json",
                "prop": "linkshere",
                "lhnamespace": 0,
                "lhlimit": INLINK_CAP,
                "lhprop": "pageid",
                "titles": title,
                "redirects": 1,
            },
        )
        pages = (data.get("query") or {}).get("pages") or {}
        n = 0
        capped = False
        for p in pages.values():
            lh = p.get("linkshere") or []
            n += len(lh)
            if p.get("linksherecont") or data.get("continue"):
                capped = True
        if capped:
            n = max(n, INLINK_CAP)
        return min(n, INLINK_CAP)
    except Exception as e:
        log(f"step0 inlinks fail {title!r}: {e}")
        return 0


def fill_inlinks(cands: list[dict]) -> None:
    # Rank a preliminary shortlist; fetch inlinks for up to 250.
    cands.sort(
        key=lambda c: (
            -int(c.get("openalex_cites") or 0),
            -int(c.get("sitelinks") or 0),
            int(c.get("year") or 9999),
        )
    )
    todo = cands[:250]
    log(f"step0 Wikipedia inlinks for {len(todo)} shortlisted works")
    for i, c in enumerate(todo, 1):
        title = c.get("enwiki_title") or ""
        if not title:
            # last resort: work title as page name
            title = c.get("work_title") or ""
        c["wiki_inlinks"] = wiki_inlinks(title)
        if i % 25 == 0:
            log(f"step0 inlinks {i}/{len(todo)}")
        time.sleep(0.12)


def rank_key(c: dict) -> tuple:
    # census: OpenAlex, Wikipedia inlinks, Wikidata sitelinks, earlier year
    return (
        -int(c.get("openalex_cites") or 0),
        -int(c.get("wiki_inlinks") or 0),
        -int(c.get("sitelinks") or 0),
        int(c.get("year") or 9999),
        (c.get("work_title") or ""),
    )


def is_excluded(c: dict, excl: dict) -> str | None:
    if c.get("work_qid") and c["work_qid"] in FORGERY_WORK_QIDS:
        return "known_forgery"
    title_l = (c.get("work_title") or "").lower()
    if any(frag in title_l for frag in FORGERY_TITLE_FRAGMENTS):
        return "known_forgery"
    if c.get("work_qid") and c["work_qid"] in excl["work_qids"]:
        return "treatment_work_qid"
    if c.get("author_qid") and c["author_qid"] in excl["writer_ids"]:
        return "treatment_author_qid"
    for q in c.get("author_qids") or []:
        if q in excl["writer_ids"]:
            return "treatment_author_qid"
    for gid in c.get("gutenberg_ids") or ([c["gutenberg_id"]] if c.get("gutenberg_id") else []):
        if str(gid) in excl["gutenberg_ids"]:
            return "treatment_gutenberg_id"
    name = ascii_fold(c.get("author") or "").lower().strip()
    if name and name in excl["author_names"]:
        return "treatment_author_name"
    nt = norm_title(c.get("work_title") or "")
    if nt and nt in excl["titles"]:
        return "treatment_title"
    return None


def candidate_urls_for(c: dict) -> list[str]:
    urls: list[str] = []
    gids = list(c.get("gutenberg_ids") or [])
    if c.get("gutenberg_id") and str(c["gutenberg_id"]) not in gids:
        gids.insert(0, str(c["gutenberg_id"]))
    for gid in gids:
        urls.extend(
            [
                f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}-8.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
                f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8",
                f"https://gutenberg.pglaf.org/cache/epub/{gid}/pg{gid}.txt",
                f"https://gutenberg.pglaf.org/files/{gid}/{gid}-0.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}-h/{gid}-h.htm",
                f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.html",
            ]
        )
    oa = c.get("oa_url") or ""
    if any(h in oa for h in ("gutenberg.org", "wikisource.org", "archive.org")):
        urls.append(oa)
    ia = c.get("ia_id") or ""
    if ia:
        urls.extend(
            [
                f"https://archive.org/stream/{ia}/{ia}_djvu.txt",
                f"https://archive.org/download/{ia}/{ia}_djvu.txt",
                f"https://archive.org/stream/{quote(ia)}/{quote(ia)}_djvu.txt",
            ]
        )
    out, seen = [], set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def wikisource_plain(title: str) -> str | None:
    if not title:
        return None
    try:
        r = HTTP.get(
            "https://en.wikisource.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": title,
                "format": "json",
                "redirects": 1,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        pages = (r.json().get("query") or {}).get("pages") or {}
        for p in pages.values():
            extract = p.get("extract") or ""
            if len(extract) > 2000:
                return extract
    except Exception:
        return None
    return None


def download_text(c: dict) -> tuple[str, str]:
    last_err = "no_url"
    for url in candidate_urls_for(c):
        try:
            r = HTTP.get(url, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code != 200 or not r.content:
                last_err = f"HTTP {r.status_code} {url}"
                continue
            ctype = (r.headers.get("Content-Type") or "").lower()
            raw = r.content
            text = None
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    text = raw.decode(enc)
                    break
                except Exception:
                    continue
            if not text:
                last_err = f"decode_fail {url}"
                continue
            path = urlparse(url).path.lower()
            if "html" in ctype or path.endswith((".htm", ".html")):
                text = html_to_text(text)
            text = strip_gutenberg(text)
            if len(text) < 1500:
                last_err = f"too_short {len(text)} {url}"
                continue
            return text, url
        except Exception as e:
            last_err = f"{type(e).__name__} {url}: {e}"
            time.sleep(0.3)
    ws = wikisource_plain(c.get("enwiki_title") or c.get("work_title") or "")
    if ws and len(ws) > 1500:
        return strip_gutenberg(ws), "wikisource:" + (c.get("work_title") or "")
    raise RuntimeError(last_err)


def citation_source(c: dict) -> str:
    if int(c.get("openalex_cites") or 0) > 0:
        return "openalex"
    if int(c.get("wiki_inlinks") or 0) > 0:
        return "wikipedia_inlinks"
    if int(c.get("sitelinks") or 0) > 0:
        return "wikidata_sitelinks"
    return "none"


def citation_score(c: dict) -> float:
    # lexicographic rank encoded as a single number for SQL ORDER BY:
    # openalex dominates; inlinks next (cap 500); sitelinks next.
    return (
        int(c.get("openalex_cites") or 0) * 1_000_000
        + int(c.get("wiki_inlinks") or 0) * 1_000
        + int(c.get("sitelinks") or 0)
    )


def select_and_fetch(state: State, ranked: list[dict], excl: dict) -> list[dict]:
    used_authors: set[str] = set()
    skipped = []
    manifest = []
    for c in ranked:
        if len(manifest) >= TARGET_N:
            break
        why = is_excluded(c, excl)
        if why:
            skipped.append({**_public(c), "skip": why})
            continue
        author_tokens = set()
        for q in c.get("author_qids") or []:
            author_tokens.add("Q:" + q)
        for k in c.get("author_keys") or []:
            author_tokens.add("K:" + k)
        if not author_tokens:
            ak = author_key(c.get("author") or "")
            if ak:
                author_tokens.add("K:" + ak)
        if author_tokens & used_authors:
            skipped.append({**_public(c), "skip": "duplicate_author"})
            continue
        if not (c.get("gutenberg_id") or c.get("ia_id") or c.get("enwiki_title")):
            skipped.append({**_public(c), "skip": "no_pd_source"})
            continue
        wid = c.get("work_qid") or ("pg" + str(c.get("gutenberg_id") or ""))
        if not wid:
            skipped.append({**_public(c), "skip": "no_id"})
            continue
        dest = BOOKS_DIR / f"{wid}.txt"
        existing = state.execute(
            "SELECT fetch_status, n_chars FROM books WHERE writer_id=?", (wid,)
        ).fetchone()
        n_chars = 0
        used_url = None
        if dest.exists() and dest.stat().st_size > 1500 and existing and existing[0] == "ok":
            n_chars = dest.stat().st_size
            used_url = state.execute(
                "SELECT text_source_url FROM books WHERE writer_id=?", (wid,)
            ).fetchone()
            used_url = used_url[0] if used_url else None
            log(f"step0 skip fetch {len(manifest)+1}/40 {wid} already n={n_chars}")
        else:
            log(
                f"step0 fetch {len(manifest)+1}/40 {wid} "
                f"{(c.get('author') or '')[:40]!r} {(c.get('work_title') or '')[:50]!r} "
                f"oa={c.get('openalex_cites')} in={c.get('wiki_inlinks')} sl={c.get('sitelinks')}"
            )
            try:
                text, used_url = download_text(c)
                dest.write_text(text, encoding="utf-8")
                n_chars = len(text)
                time.sleep(0.8)
            except Exception as e:
                skipped.append({**_public(c), "skip": f"fetch_fail:{e}"})
                state.log_error("step0", "", wid, "fetch", str(e))
                log(f"step0 FETCH FAIL {wid}: {e}")
                continue
        gid = str(c.get("gutenberg_id") or next(iter(c.get("gutenberg_ids") or []), "") or "")
        pd_url = f"https://www.gutenberg.org/ebooks/{gid}" if gid else (used_url or "")
        rec = {
            "writer_id": wid,
            "author": c.get("author"),
            "author_qid": c.get("author_qid"),
            "work_qid": c.get("work_qid"),
            "work_title": c.get("work_title"),
            "year": c.get("year"),
            "pd_url": pd_url,
            "text_source_url": used_url,
            "gutenberg_id": gid,
            "openalex_cites": int(c.get("openalex_cites") or 0),
            "wiki_inlinks": int(c.get("wiki_inlinks") or 0),
            "sitelinks": int(c.get("sitelinks") or 0),
            "citation_score": citation_score(c),
            "citation_source": citation_source(c),
            "source_pool": c.get("source_pool"),
            "local_path": f"books_raw/{wid}.txt",
            "n_chars": n_chars,
        }
        state.execute(
            """
            INSERT INTO books(
              writer_id,author,work_title,pd_url,text_source_url,ier_intrinsic,
              citation_score,citation_source,year,author_qid,work_qid,
              wiki_inlinks,sitelinks,n_chars,fetch_status,error
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(writer_id) DO UPDATE SET
              author=excluded.author, work_title=excluded.work_title,
              pd_url=excluded.pd_url, text_source_url=excluded.text_source_url,
              citation_score=excluded.citation_score, citation_source=excluded.citation_source,
              year=excluded.year, author_qid=excluded.author_qid, work_qid=excluded.work_qid,
              wiki_inlinks=excluded.wiki_inlinks, sitelinks=excluded.sitelinks,
              n_chars=excluded.n_chars, fetch_status='ok', error=NULL
            """,
            (
                wid,
                rec["author"],
                rec["work_title"],
                rec["pd_url"],
                rec["text_source_url"],
                None,
                rec["citation_score"],
                rec["citation_source"],
                rec["year"],
                rec["author_qid"],
                rec["work_qid"],
                rec["wiki_inlinks"],
                rec["sitelinks"],
                rec["n_chars"],
                "ok",
                None,
            ),
        )
        used_authors |= author_tokens
        manifest.append(rec)
    write_json(DATA_DIR / "selection_skipped.json", skipped[:500])
    return manifest


def _public(c: dict) -> dict:
    return {
        "author": c.get("author"),
        "author_qid": c.get("author_qid"),
        "work_qid": c.get("work_qid"),
        "work_title": c.get("work_title"),
        "year": c.get("year"),
        "gutenberg_id": c.get("gutenberg_id"),
        "openalex_cites": int(c.get("openalex_cites") or 0),
        "wiki_inlinks": int(c.get("wiki_inlinks") or 0),
        "sitelinks": int(c.get("sitelinks") or 0),
        "source_pool": c.get("source_pool"),
    }


def main() -> None:
    state = State()
    log("step0 control select+fetch start")
    con = sqlite3.connect(f"file:{SCORES_DB}?mode=ro", uri=True)
    ymin, ymax, top40 = treatment_window(con)
    con.close()
    ymax = min(ymax, YEAR_PD_MAX)
    log(f"step0 treatment year window {ymin}–{ymax} n_top40={len(top40)}")
    state.set_meta("year_min", str(ymin))
    state.set_meta("year_max", str(ymax))
    excl = load_exclusions(top40)
    log(
        f"step0 exclusions writers={len(excl['writer_ids'])} names={len(excl['author_names'])} "
        f"gids={len(excl['gutenberg_ids'])} work_qids={len(excl['work_qids'])}"
    )

    existing_ok = state.execute(
        "SELECT COUNT(*) FROM books WHERE fetch_status='ok'"
    ).fetchone()[0]
    man_path = DATA_DIR / "corpus_manifest.json"
    if existing_ok >= TARGET_N and man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        if len(manifest) >= TARGET_N:
            log(f"step0 reuse existing {len(manifest)} control books")
            write_coverage(
                state,
                extra={
                    "year_min": ymin,
                    "year_max": ymax,
                    "manifest": manifest,
                    "notes": [f"Step 0 reused {len(manifest)} fetched control texts."],
                },
            )
            return

    pg_rows = download_pg_catalog()
    by_gid, by_last = index_gutenberg(pg_rows)

    wd_g = pool_gutenberg_wikidata(ymin, ymax)
    wd_lit = match_literary_to_gutenberg(pool_sitelink_literary(ymin, ymax), by_gid, by_last)
    oa = pool_openalex(ymin, ymax)
    oa_m = match_openalex_to_gutenberg(oa, by_last)

    # Attach Gutenberg metadata onto Wikidata rows whose gid is in catalog (English gate).
    wd_g_en = []
    for c in wd_g:
        gid = str(c.get("gutenberg_id") or "")
        if gid in by_gid:
            rec = by_gid[gid]
            c = {**c, "gutenberg_title": rec["work_title"]}
            wd_g_en.append(c)
        else:
            # still keep if we can match author+title into English catalog
            last, _ = parse_author_name(c.get("author") or "")
            hit = None
            for rec in by_last.get(last, []):
                if titles_match(c["work_title"], rec["work_title"]):
                    hit = rec
                    break
            if hit:
                c = {**c, "gutenberg_id": hit["gutenberg_id"], "gutenberg_title": hit["work_title"]}
                wd_g_en.append(c)
    log(f"step0 Wikidata Gutenberg English-gated {len(wd_g_en)}/{len(wd_g)}")

    merged = merge_candidates([wd_g_en, wd_lit, oa_m])
    fill_inlinks(merged)
    merged.sort(key=rank_key)
    # freeze sets for JSON
    serializable = []
    for c in merged:
        rec = _public(c)
        rec["gutenberg_ids"] = sorted(c.get("gutenberg_ids") or [])
        rec["author_qids"] = sorted(c.get("author_qids") or [])
        serializable.append(rec)
    write_json(DATA_DIR / "selection_ranked.json", serializable[:400])

    manifest = select_and_fetch(state, merged, excl)
    write_json(DATA_DIR / "corpus_manifest.json", manifest)
    n_ok = sum(1 for p in BOOKS_DIR.glob("*.txt") if p.stat().st_size > 1500)
    log(f"step0 done books_ok={n_ok}/{TARGET_N} manifest={DATA_DIR / 'corpus_manifest.json'}")
    write_coverage(
        state,
        extra={
            "year_min": ymin,
            "year_max": ymax,
            "manifest": manifest,
            "notes": [
                f"Step 0 selected+fetched {n_ok}/40 control texts in year window {ymin}–{ymax}.",
                "Ranking: OpenAlex cited_by_count, then Wikipedia inlinks (cap 500), then Wikidata sitelinks.",
                "Treatment I/E-R 40 works and authors excluded. Not restricted to Christian writers.",
            ],
        },
    )
    if n_ok < TARGET_N:
        raise SystemExit(f"step0 fetched {n_ok}/{TARGET_N} texts")


if __name__ == "__main__":
    main()
