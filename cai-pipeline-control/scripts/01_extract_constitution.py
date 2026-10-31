#!/usr/bin/env python3
"""Step 1: corpus-only constitutional extraction + MiniLM near-duplicate merge."""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_common import (  # noqa: E402
    BOOKS_DIR,
    CHUNK_MAX_CHARS,
    DATA_DIR,
    EXTRACT_PROMPT,
    EXTRACT_SCHEMA,
    LLMClient,
    MODEL_ID,
    State,
    chunk_text,
    log,
    write_coverage,
    write_json,
    write_jsonl,
)

_EMB_LOCK = threading.Lock()
_EMB_MODEL = None


def minilm():
    global _EMB_MODEL
    with _EMB_LOCK:
        if _EMB_MODEL is None:
            from sentence_transformers import SentenceTransformer

            _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _EMB_MODEL


def load_book(writer_id: str) -> str:
    p = BOOKS_DIR / f"{writer_id}.txt"
    return p.read_text(encoding="utf-8", errors="replace")


def merge_parts(parts: list[dict], book_id: str, author: str) -> dict:
    principles = []
    prompts = []
    seen_p, seen_q = set(), set()
    for part in parts:
        for pr in part.get("principles") or []:
            if not isinstance(pr, dict):
                continue
            thesis = (pr.get("source_thesis") or "").strip()
            cr = (pr.get("critique_request") or "").strip()
            rr = (pr.get("revision_request") or "").strip()
            if not (thesis and cr and rr):
                continue
            key = " ".join(cr.lower().split())
            if key in seen_p:
                continue
            seen_p.add(key)
            principles.append(
                {"source_thesis": thesis, "critique_request": cr, "revision_request": rr}
            )
        for q in part.get("elicited_prompts") or []:
            if not isinstance(q, str):
                continue
            s = " ".join(q.split())
            if len(s) < 12:
                continue
            k = s.lower()
            if k in seen_q:
                continue
            seen_q.add(k)
            prompts.append(s)
    if len(prompts) > 50:
        prompts = diversify_prompts(prompts, 50)
    return {
        "book_id": book_id,
        "author": author,
        "principles": principles,
        "elicited_prompts": prompts,
    }


def diversify_prompts(prompts: list[str], k: int) -> list[str]:
    try:
        model = minilm()
        emb = model.encode(prompts, normalize_embeddings=True, show_progress_bar=False)
        selected = [0]
        while len(selected) < k and len(selected) < len(prompts):
            sel = np.array(emb[selected])
            # maximin: farthest from current set
            sims = sel @ emb.T
            nearest = sims.max(axis=0)
            nearest[selected] = 1.1
            nxt = int(np.argmin(nearest))
            selected.append(nxt)
        return [prompts[i] for i in selected]
    except Exception:
        step = max(1, len(prompts) // k)
        return prompts[::step][:k]


def extract_book(pool: LLMClient, state: State, writer_id: str, author: str, title: str) -> dict:
    extract_path = DATA_DIR / "extracts" / f"{writer_id}.json"
    if extract_path.exists():
        data = json.loads(extract_path.read_text(encoding="utf-8"))
        if (
            data.get("model") == MODEL_ID
            and data.get("elicited_prompts") is not None
            and data.get("principles") is not None
        ):
            log(f"step1 skip {writer_id} cached extract model={MODEL_ID}")
            return data
    text = load_book(writer_id)
    chunks = chunk_text(text, max_chars=CHUNK_MAX_CHARS)
    n_chunks = len(chunks)
    parts: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        if n_chunks == 1:
            header = (
                f"Book id: {writer_id}\nAuthor: {author}\nWork title: {title}\n\n"
                f"TEXT:\n{chunk}\n"
            )
            user = EXTRACT_PROMPT + "\n" + header
        else:
            header = (
                f"Book id: {writer_id}\nAuthor: {author}\nWork title: {title}\n"
                f"This excerpt is portion {i} of {n_chunks} of one work. "
                f"Extract only from this excerpt. Do not invent material this excerpt does not support. "
                f"Produce as many distinct first-person dilemmas as this excerpt actually addresses "
                f"(target about {max(8, 50 // n_chunks + 4)}, not more than the excerpt supports).\n\n"
                f"TEXT:\n{chunk}\n"
            )
            user = EXTRACT_PROMPT + "\n" + header
        ref = f"{writer_id}:chunk{i}/{n_chunks}"
        log(f"step1 extract {ref} chars={len(chunk)}")
        data = pool.generate_json(
            user,
            EXTRACT_SCHEMA,
            step="step1",
            ref=ref,
            max_output_tokens=16384,
            temperature=0.4,
        )
        if not isinstance(data, dict):
            data = {}
        data.setdefault("book_id", writer_id)
        data.setdefault("author", author)
        data.setdefault("principles", [])
        data.setdefault("elicited_prompts", [])
        parts.append(data)

    merged = merge_parts(parts, writer_id, author)
    # If a single-chunk book came in well under 50 and the text is long enough to support more,
    # one follow-up elicitation on the same book (no other books, no outside topics).
    if n_chunks == 1 and len(merged["elicited_prompts"]) < 40 and len(text) > 8000:
        already = merged["elicited_prompts"]
        follow = (
            EXTRACT_PROMPT
            + f"\nBook id: {writer_id}\nAuthor: {author}\nWork title: {title}\n"
            + "Additional dilemmas already extracted (do not repeat):\n"
            + "\n".join(f"- {p}" for p in already[:40])
            + "\n\nTEXT:\n"
            + text[:60_000]
            + "\n"
        )
        try:
            more = pool.generate_json(
                follow,
                EXTRACT_SCHEMA,
                step="step1",
                ref=f"{writer_id}:followup",
                max_output_tokens=16384,
                temperature=0.5,
            )
            merged = merge_parts([merged, more if isinstance(more, dict) else {}], writer_id, author)
        except Exception as e:
            state.log_error("step1", MODEL_ID, f"{writer_id}:followup", type(e).__name__, str(e))
    merged["model"] = MODEL_ID
    extract_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merged


def cluster_principles(raw: list[dict]) -> list[dict]:
    items = []
    for rec in raw:
        bid = rec.get("book_id")
        for pr in rec.get("principles") or []:
            items.append({**pr, "source_books": [bid]})
    if len(items) <= 1:
        return items
    from sklearn.cluster import AgglomerativeClustering

    model = minilm()
    texts = [
        (it.get("source_thesis") or "")
        + " "
        + (it.get("critique_request") or "")
        for it in items
    ]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=0.28,
    )
    labels = clustering.fit_predict(emb)
    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    out = []
    for lab, idxs in clusters.items():
        vecs = emb[idxs]
        centroid = vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = vecs @ centroid
        best = idxs[int(np.argmax(sims))]
        books = []
        for i in idxs:
            for b in items[i].get("source_books") or []:
                if b and b not in books:
                    books.append(b)
        rec = {
            "source_thesis": items[best]["source_thesis"],
            "critique_request": items[best]["critique_request"],
            "revision_request": items[best]["revision_request"],
            "source_books": books,
            "cluster_size": len(idxs),
        }
        out.append(rec)
    out.sort(key=lambda r: (-r["cluster_size"], r["source_thesis"][:80]))
    return out


def export_raw(state: State) -> list[dict]:
    rows = state.execute(
        "SELECT writer_id, author FROM books WHERE extract_status='ok' ORDER BY COALESCE(citation_score,0) DESC"
    ).fetchall()
    raw = []
    for wid, author in rows:
        p = DATA_DIR / "extracts" / f"{wid}.json"
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec.get("model") != MODEL_ID:
                continue
            rec.setdefault("book_id", wid)
            rec.setdefault("author", author)
            raw.append(rec)
    write_jsonl(DATA_DIR / "raw_principles.jsonl", raw)
    return raw


def main() -> None:
    state = State()
    pool = LLMClient(state)
    books = state.execute(
        """
        SELECT writer_id, author, work_title FROM books
        WHERE fetch_status='ok'
        ORDER BY COALESCE(citation_score,0) DESC
        """
    ).fetchall()
    if not books:
        raise SystemExit("step1: no fetched books; run 00_select_and_fetch.py first")
    log(f"step1 start n_books={len(books)}")
    todo = []
    for i, (wid, author, title) in enumerate(books, 1):
        row = state.execute(
            "SELECT extract_status, extract_model FROM books WHERE writer_id=?", (wid,)
        ).fetchone()
        if (
            row
            and row[0] == "ok"
            and row[1] == MODEL_ID
            and (DATA_DIR / "extracts" / f"{wid}.json").exists()
        ):
            log(f"step1 {i}/{len(books)} {wid} already extracted model={MODEL_ID}")
            continue
        todo.append((i, wid, author or "", title or ""))

    def _one(item):
        i, wid, author, title = item
        try:
            merged = extract_book(pool, state, wid, author, title)
            state.execute(
                """
                UPDATE books SET extract_status='ok', n_principles=?, n_prompts=?, n_chunks=?,
                  error=NULL, extract_model=?
                WHERE writer_id=?
                """,
                (
                    len(merged.get("principles") or []),
                    len(merged.get("elicited_prompts") or []),
                    len(chunk_text(load_book(wid))),
                    MODEL_ID,
                    wid,
                ),
            )
            log(
                f"step1 {i}/{len(books)} {wid} principles={len(merged['principles'])} "
                f"prompts={len(merged['elicited_prompts'])}"
            )
            return wid
        except Exception as e:
            state.execute(
                "UPDATE books SET extract_status='error', error=? WHERE writer_id=?",
                (str(e)[:800], wid),
            )
            state.log_error("step1", MODEL_ID, wid, type(e).__name__, str(e))
            log(f"step1 FAIL {wid}: {e}")
            raise

    if todo:
        workers = 4
        log(f"step1 extracting {len(todo)} books workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, item) for item in todo]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    pass
                write_coverage(state)

    if not todo:
        const_path = DATA_DIR / "constitution.json"
        if const_path.exists():
            existing = json.loads(const_path.read_text(encoding="utf-8"))
            if existing.get("model") == MODEL_ID and existing.get("principles"):
                n_ok = state.execute(
                    "SELECT COUNT(*) FROM books WHERE extract_status='ok' AND COALESCE(extract_model,'')=?",
                    (MODEL_ID,),
                ).fetchone()[0]
                log(f"step1 skip cluster extracted={n_ok} constitution={len(existing['principles'])}")
                state.set_meta("constitution_n", str(len(existing["principles"])))
                write_coverage(
                    state,
                    extra={
                        "constitution_n": len(existing["principles"]),
                        "notes": [f"Step 1 reused clustered constitution size {len(existing['principles'])}."],
                    },
                )
                if n_ok == 0:
                    raise SystemExit("step1 extracted 0 books")
                return

    raw = export_raw(state)
    constitution = cluster_principles(raw)
    write_json(
        DATA_DIR / "constitution.json",
        {"model": MODEL_ID, "thinking": "disabled", "principles": constitution},
    )
    state.set_meta("constitution_n", str(len(constitution)))
    n_ok = state.execute(
        "SELECT COUNT(*) FROM books WHERE extract_status='ok' AND COALESCE(extract_model,'')=?",
        (MODEL_ID,),
    ).fetchone()[0]
    log(f"step1 done extracted={n_ok} constitution={len(constitution)}")
    write_coverage(
        state,
        extra={
            "constitution_n": len(constitution),
            "notes": [
                f"Step 1 extracted {n_ok} books; clustered constitution size {len(constitution)}.",
                "Near-duplicates merged with sentence-transformers/all-MiniLM-L6-v2 + AgglomerativeClustering (cosine, average, threshold 0.28).",
            ],
        },
    )
    if n_ok == 0:
        raise SystemExit("step1 extracted 0 books")


if __name__ == "__main__":
    main()
