#!/usr/bin/env python3
"""Step 0: I/E-R intrinsic top 40, fetch Gutenberg/Wikisource texts. No LLM."""
from __future__ import annotations

import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_common import (  # noqa: E402
    BOOKS_DIR,
    DATA_DIR,
    SCORES_DB,
    State,
    html_to_text,
    log,
    strip_gutenberg,
    write_coverage,
    write_json,
)

UA = "CAI-pipeline/1.0 (educational research; Python requests)"
TIMEOUT = 60


def top40(con: sqlite3.Connection) -> list[dict]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT s.writer_id,
               ad.canonical_name AS author,
               w.work_title,
               w.pd_url,
               w.text_source_url,
               w.pd_source,
               s.score AS ier_intrinsic
        FROM work_subscores s
        JOIN works_scored w ON w.writer_id = s.writer_id
        LEFT JOIN author_decisions ad ON ad.writer_id = s.writer_id
        WHERE s.subscale = 'ier_intrinsic'
        ORDER BY s.score DESC
        LIMIT 40
        """
    ).fetchall()
    if len(rows) != 40:
        raise SystemExit(f"expected 40 I/E-R rows, got {len(rows)}")
    return [dict(r) for r in rows]


def gutenberg_id(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"/ebooks/(\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/(\d{2,})(?:[-/._]|$)", url)
    return m.group(1) if m else None


def candidate_urls(row: dict) -> list[str]:
    urls: list[str] = []
    primary = row.get("text_source_url") or row.get("pd_url") or ""
    gid = gutenberg_id(primary) or gutenberg_id(row.get("pd_url") or "")
    if gid:
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
    if primary:
        urls.append(primary)
    # unique preserve order
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def wikisource_plain(title: str) -> str | None:
    # last-resort: MediaWiki parse of a title-like path
    if not title:
        return None
    api = "https://en.wikisource.org/w/api.php"
    try:
        r = requests.get(
            api,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": title,
                "format": "json",
                "redirects": 1,
            },
            headers={"User-Agent": UA},
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


def download_text(row: dict) -> tuple[str, str]:
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept": "text/plain, text/html;q=0.9, */*;q=0.5"})
    last_err = "no_url"
    for url in candidate_urls(row):
        try:
            r = sess.get(url, timeout=TIMEOUT, allow_redirects=True)
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
            time.sleep(0.4)
    ws = wikisource_plain(row.get("work_title") or "")
    if ws and len(ws) > 1500:
        return strip_gutenberg(ws), "wikisource:" + (row.get("work_title") or "")
    raise RuntimeError(last_err)


def main() -> None:
    state = State()
    log("step0 select+fetch start")
    con = sqlite3.connect(f"file:{SCORES_DB}?mode=ro", uri=True)
    rows = top40(con)
    con.close()
    manifest = []
    for i, row in enumerate(rows, 1):
        wid = row["writer_id"]
        dest = BOOKS_DIR / f"{wid}.txt"
        existing = state.execute("SELECT fetch_status, n_chars FROM books WHERE writer_id=?", (wid,)).fetchone()
        if dest.exists() and dest.stat().st_size > 1500 and existing and existing[0] == "ok":
            n_chars = dest.stat().st_size
            log(f"step0 skip {i}/40 {wid} already fetched n={n_chars}")
        else:
            log(f"step0 fetch {i}/40 {wid} {(row.get('author') or '')[:40]!r}")
            try:
                text, used = download_text(row)
                dest.write_text(text, encoding="utf-8")
                n_chars = len(text)
                state.execute(
                    """
                    INSERT INTO books(writer_id,author,work_title,pd_url,text_source_url,ier_intrinsic,n_chars,fetch_status,error)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(writer_id) DO UPDATE SET
                      author=excluded.author, work_title=excluded.work_title,
                      pd_url=excluded.pd_url, text_source_url=excluded.text_source_url,
                      ier_intrinsic=excluded.ier_intrinsic, n_chars=excluded.n_chars,
                      fetch_status='ok', error=NULL
                    """,
                    (
                        wid,
                        row.get("author"),
                        row.get("work_title"),
                        row.get("pd_url"),
                        used,
                        row.get("ier_intrinsic"),
                        n_chars,
                        "ok",
                        None,
                    ),
                )
                time.sleep(0.8)
            except Exception as e:
                state.execute(
                    """
                    INSERT INTO books(writer_id,author,work_title,pd_url,text_source_url,ier_intrinsic,fetch_status,error)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(writer_id) DO UPDATE SET fetch_status='error', error=excluded.error
                    """,
                    (
                        wid,
                        row.get("author"),
                        row.get("work_title"),
                        row.get("pd_url"),
                        row.get("text_source_url"),
                        row.get("ier_intrinsic"),
                        "error",
                        str(e)[:800],
                    ),
                )
                state.log_error("step0", "", wid, "fetch", str(e))
                n_chars = 0
                used = row.get("text_source_url")
        manifest.append(
            {
                "writer_id": wid,
                "author": row.get("author"),
                "work_title": row.get("work_title"),
                "pd_url": row.get("pd_url"),
                "text_source_url": (existing and None) or row.get("text_source_url"),
                "ier_intrinsic": row.get("ier_intrinsic"),
                "local_path": f"books_raw/{wid}.txt",
                "n_chars": dest.stat().st_size if dest.exists() else n_chars,
            }
        )
        # fill text_source_url from DB if we stored the actually-used URL
        used_row = state.execute(
            "SELECT text_source_url, n_chars FROM books WHERE writer_id=?", (wid,)
        ).fetchone()
        if used_row:
            manifest[-1]["text_source_url"] = used_row[0]
            manifest[-1]["n_chars"] = used_row[1]

    write_json(DATA_DIR / "corpus_manifest.json", manifest)
    n_ok = sum(1 for p in BOOKS_DIR.glob("*.txt") if p.stat().st_size > 1500)
    log(f"step0 done books_ok={n_ok}/40 manifest={DATA_DIR / 'corpus_manifest.json'}")
    if n_ok < 40:
        log("step0 WARNING incomplete fetches; continuing with what we have")
    write_coverage(state, extra={"notes": [f"Step 0 fetched {n_ok}/40 texts into books_raw/."]})
    if n_ok == 0:
        raise SystemExit("step0 fetched 0 texts")


if __name__ == "__main__":
    main()
