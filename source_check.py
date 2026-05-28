import argparse
import json
import math
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

# This command must not consume GPU memory. These environment flags are set
# before project imports so incidental ML library imports stay on CPU/silent.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SOURCE_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:source|sources|citation|citations|reference|references|"
    r"file|filename|document|doc|page|section|chunk|"
    r"\ucd9c\ucc98|\uadfc\uac70|\ud30c\uc77c\uba85|\ubb38\uc11c|\ud398\uc774\uc9c0|\ucabd|"
    r"\uc139\uc158|\uccad\ud06c|\uc81c\ubaa9|\uae30\uad00|\uc608\uc0b0|\uacf5\uace0\uc77c|"
    r"\uc785\ucc30\uae30\uac04|\ud575\uc2ec\ub0b4\uc6a9|\ubb38\uc11cid)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{2,}|[\uac00-\ud7a3]{2,}")
CHUNK_ID_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_.-]*_\d{4,}\b")
NOT_FOUND_RE = re.compile(
    r"(\ucc3e\uc744 \uc218 \uc5c6|\ud655\uc778\ud560 \uc218 \uc5c6|"
    r"\uad00\ub828\ub41c \uc815\ubcf4\ub97c \ucc3e\uc9c0 \ubabb|"
    r"\uad00\ub828 \uc815\ubcf4\uac00 \uc5c6|\uadfc\uac70\uac00 \uc5c6)",
    re.IGNORECASE,
)
PAGE_RE_LIST = [
    re.compile(r"(?:p\.?|page|\ud398\uc774\uc9c0|\ucabd)\s*[:.]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:p\.?|page|\ud398\uc774\uc9c0|\ucabd)\b", re.IGNORECASE),
]

DATE_RE = re.compile(
    r"\d{4}\s*(?:[-./]|\ub144)\s*\d{1,2}\s*(?:[-./]|\uc6d4)\s*\d{1,2}\s*(?:\uc77c)?"
)
NUMBER_UNIT_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:"
    r"\uc6d0|\ucc9c\uc6d0|\ubc31\ub9cc\uc6d0|\ub9cc\uc6d0|\uc5b5\uc6d0|"
    r"\uc77c|\uac1c\uc6d4|\ub144|\uba85|\uac74|%|"
    r"won|krw|days?|months?|years?"
    r")?",
    re.IGNORECASE,
)
REQ_ID_RE = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b")

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "about",
}


def normalize_text(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: Any) -> str:
    return re.sub(r"\s+", "", normalize_text(text).lower())


def tokenize(text: Any) -> set[str]:
    tokens = {m.group(0).lower() for m in TOKEN_RE.finditer(normalize_text(text))}
    return {token for token in tokens if token not in STOPWORDS}


def normalize_value(value: str) -> str:
    value = compact_text(value)
    value = value.replace(",", "")
    value = value.replace(" ", "")
    return value


def extract_values(text: Any) -> list[str]:
    normalized = normalize_text(text)
    values: list[str] = []

    for regex in (DATE_RE, REQ_ID_RE, NUMBER_UNIT_RE):
        for match in regex.finditer(normalized):
            value = normalize_value(match.group(0))
            if value and value not in values:
                values.append(value)

    return values


def split_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for raw_line in normalize_text(answer).replace(". ", ".\n").splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw_line).strip()
        if not line or SOURCE_LINE_RE.match(line):
            continue

        parts = re.split(r"(?<=[.!?])\s+|(?<=[\ub2e4\uc694\ub2c8\uae4c])\s+", line)
        for part in parts:
            part = part.strip(" -\t")
            part = CHUNK_ID_RE.sub("", part)
            part = re.sub(r"\[\s*\]", "", part).strip(" -\t[]")
            if len(part) >= 8 and not SOURCE_LINE_RE.match(part):
                claims.append(part)

    # Keep order while removing duplicates.
    seen = set()
    unique_claims = []
    for claim in claims:
        key = compact_text(claim)
        if key not in seen:
            seen.add(key)
            unique_claims.append(claim)

    return unique_claims


def flatten_chunk(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", item)
    return {
        "id": item.get("id") or metadata.get("chunk_id"),
        "chunk_id": metadata.get("chunk_id") or item.get("id"),
        "doc_id": metadata.get("doc_id"),
        "file_name": metadata.get("file_name"),
        "title": metadata.get("title"),
        "page_number": metadata.get("page_number"),
        "section_title": metadata.get("section_title"),
        "requirement_id": metadata.get("requirement_id"),
        "content": metadata.get("content") or item.get("content") or "",
        "metadata": metadata,
    }


def load_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [flatten_chunk(item) for item in raw]


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_collection_name(config_path: Path, collection_name: str | None, embed_provider: str | None) -> str:
    if collection_name:
        return collection_name

    config = load_config(config_path)
    provider = embed_provider or config.get("providers", {}).get("embedding")
    collections = config.get("collection_name", {})

    if provider and provider in collections:
        return collections[provider]

    if collections:
        return next(iter(collections.values()))

    raise SystemExit("컬렉션명을 찾을 수 없습니다. --collection 또는 config.yaml의 collection_name을 설정하세요.")


def load_chunks_from_db(collection_name: str, batch_size: int, limit: int | None = None) -> list[dict[str, Any]]:
    from src.vector_db.vectordb import client

    chunks: list[dict[str, Any]] = []
    offset = None

    while True:
        request_limit = batch_size
        if limit is not None:
            remaining = limit - len(chunks)
            if remaining <= 0:
                break
            request_limit = min(request_limit, remaining)

        records, offset = client.scroll(
            collection_name=collection_name,
            limit=request_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = dict(record.payload or {})
            payload.setdefault("point_id", str(record.id))
            chunks.append(flatten_chunk(payload))

        if offset is None or not records:
            break

    return chunks


def find_pages(answer: str) -> list[int]:
    pages: list[int] = []
    for regex in PAGE_RE_LIST:
        for match in regex.finditer(answer):
            page = int(match.group(1))
            if page not in pages:
                pages.append(page)
    return pages


def find_chunk_ids(answer: str, chunks: list[dict[str, Any]]) -> list[str]:
    valid_ids = {str(chunk.get("chunk_id") or "") for chunk in chunks}
    found = []
    for match in CHUNK_ID_RE.finditer(answer):
        chunk_id = match.group(0)
        if chunk_id in valid_ids and chunk_id not in found:
            found.append(chunk_id)
    return found


def find_file_names(answer: str, chunks: list[dict[str, Any]]) -> list[str]:
    answer_compact = compact_text(answer)
    names = sorted(
        {str(chunk.get("file_name") or "") for chunk in chunks if chunk.get("file_name")},
        key=len,
        reverse=True,
    )

    found: list[str] = []
    for name in names:
        if compact_text(name) in answer_compact and name not in found:
            found.append(name)

    return found


def extract_citations(answer: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "chunk_ids": find_chunk_ids(answer, chunks),
        "file_names": find_file_names(answer, chunks),
        "pages": find_pages(answer),
    }


def page_matches(chunk_page: Any, cited_pages: Iterable[int]) -> bool:
    if chunk_page is None:
        return False

    try:
        page = int(chunk_page)
    except (TypeError, ValueError):
        return False

    cited = set(cited_pages)
    return page in cited or (page + 1) in cited


def source_candidates(citations: dict[str, Any], chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    chunk_ids = set(citations["chunk_ids"])
    file_names = set(citations["file_names"])
    pages = citations["pages"]

    if chunk_ids:
        candidates = [chunk for chunk in chunks if chunk.get("chunk_id") in chunk_ids]
        return candidates, "chunk_id"

    if file_names and pages:
        candidates = [
            chunk
            for chunk in chunks
            if chunk.get("file_name") in file_names and page_matches(chunk.get("page_number"), pages)
        ]
        if candidates:
            return candidates, "file_page"

    if file_names:
        candidates = [chunk for chunk in chunks if chunk.get("file_name") in file_names]
        return candidates, "file_only"

    return [], "missing"


def lexical_score(query: str, content: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    content_tokens = tokenize(content)
    overlap = query_tokens & content_tokens
    return len(overlap) / len(query_tokens)


def value_score(claim_values: list[str], content: str) -> tuple[float | None, list[str]]:
    if not claim_values:
        return None, []

    content_compact = normalize_value(content)
    missing = [value for value in claim_values if value not in content_compact]
    return (len(claim_values) - len(missing)) / len(claim_values), missing


def best_evidence(claim: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = -math.inf

    for chunk in candidates:
        content = chunk_evidence_text(chunk)
        score = lexical_score(claim, content)
        values, missing = value_score(extract_values(claim), content)
        if values is not None:
            score = 0.45 * score + 0.55 * values

        if score > best_score:
            best_score = score
            best = dict(chunk)
            best["_match_score"] = round(score, 4)
            best["_missing_values"] = missing

    return best


def lexical_candidates(claim: str, chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    scored = []
    for chunk in chunks:
        content = chunk_evidence_text(chunk)
        score = lexical_score(claim, content)
        values, _ = value_score(extract_values(claim), content)
        if values is not None:
            score = 0.45 * score + 0.55 * values
        if score > 0:
            candidate = dict(chunk)
            candidate["_candidate_score"] = round(score, 4)
            scored.append(candidate)

    scored.sort(key=lambda item: item["_candidate_score"], reverse=True)
    return scored[:limit]


def snippet(text: Any, limit: int = 280) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def chunk_evidence_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    fields = [
        ("doc_id", chunk.get("doc_id") or metadata.get("doc_id")),
        ("chunk_id", chunk.get("chunk_id") or metadata.get("chunk_id")),
        ("file_name", chunk.get("file_name") or metadata.get("file_name")),
        ("title", chunk.get("title") or metadata.get("title")),
        ("organization", metadata.get("organization")),
        ("budget", metadata.get("budget")),
        ("announcement_date", metadata.get("announcement_date")),
        ("bid_start", metadata.get("bid_start")),
        ("bid_deadline", metadata.get("bid_deadline")),
        ("page_number", chunk.get("page_number") or metadata.get("page_number")),
        ("section_title", chunk.get("section_title") or metadata.get("section_title")),
        ("requirement_id", chunk.get("requirement_id") or metadata.get("requirement_id")),
        ("content", chunk.get("content") or metadata.get("content")),
    ]

    lines = []
    for label, value in fields:
        text = value_to_text(value)
        if text:
            lines.append(f"{label}: {text}")

    return "\n".join(lines)


def is_not_found_answer(answer: str) -> bool:
    return bool(NOT_FOUND_RE.search(normalize_text(answer)))


def check_claim(
    claim: str,
    chunks: list[dict[str, Any]],
    cited_candidates: list[dict[str, Any]],
    source_mode: str,
    fallback_top_k: int,
) -> dict[str, Any]:
    claim_values = extract_values(claim)

    if source_mode == "missing":
        fallback = lexical_candidates(claim, chunks, fallback_top_k)
        evidence = fallback[0] if fallback else None
        return {
            "claim": claim,
            "status": "missing_citation",
            "reason": "No explicit source was found in the answer.",
            "claim_values": claim_values,
            "best_candidate": evidence_to_output(evidence),
        }

    if not cited_candidates:
        return {
            "claim": claim,
            "status": "bad_source_location",
            "reason": "The cited source could not be mapped to a stored chunk.",
            "claim_values": claim_values,
            "best_candidate": None,
        }

    evidence = best_evidence(claim, cited_candidates)
    if not evidence:
        return {
            "claim": claim,
            "status": "unclear",
            "reason": "No source text was available for the cited location.",
            "claim_values": claim_values,
            "best_evidence": None,
        }

    match_score = float(evidence.get("_match_score", 0.0))
    missing_values = evidence.get("_missing_values", [])

    if missing_values:
        status = "unclear"
        reason = "Some exact values in the claim were not found in the cited source."
    elif match_score >= 0.55:
        status = "clear"
        reason = "The cited source location exists and supports the claim."
    elif match_score >= 0.25:
        status = "partial"
        reason = "The cited source is related, but the support is weak or indirect."
    else:
        status = "unclear"
        reason = "The cited source does not sufficiently support the claim."

    return {
        "claim": claim,
        "status": status,
        "reason": reason,
        "match_score": match_score,
        "claim_values": claim_values,
        "missing_values": missing_values,
        "best_evidence": evidence_to_output(evidence),
    }


def evidence_to_output(evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if not evidence:
        return None

    return {
        "chunk_id": evidence.get("chunk_id"),
        "doc_id": evidence.get("doc_id"),
        "file_name": evidence.get("file_name"),
        "page_number": evidence.get("page_number"),
        "section_title": evidence.get("section_title"),
        "requirement_id": evidence.get("requirement_id"),
        "score": evidence.get("_match_score", evidence.get("_candidate_score")),
        "snippet": snippet(evidence.get("content")),
    }


def overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {check.get("status") for check in checks}
    if not checks:
        return "no_claims"
    if "bad_source_location" in statuses or "missing_citation" in statuses or "unclear" in statuses:
        return "needs_review"
    if "partial" in statuses:
        return "partial"
    return "clear"


def verify_answer(answer: str, chunks: list[dict[str, Any]], fallback_top_k: int) -> dict[str, Any]:
    claims = split_claims(answer)
    citations = extract_citations(answer, chunks)

    if is_not_found_answer(answer) and not citations["chunk_ids"] and not citations["file_names"]:
        return {
            "overall_status": "not_found_response",
            "source_mode": "not_found_response",
            "citations": citations,
            "claim_count": 0,
            "checks": [],
        }

    candidates, source_mode = source_candidates(citations, chunks)

    checks = [
        check_claim(
            claim=claim,
            chunks=chunks,
            cited_candidates=candidates,
            source_mode=source_mode,
            fallback_top_k=fallback_top_k,
        )
        for claim in claims
    ]

    return {
        "overall_status": overall_status(checks),
        "source_mode": source_mode,
        "citations": citations,
        "claim_count": len(claims),
        "checks": checks,
    }


def read_answer(args: argparse.Namespace) -> str:
    if args.answer:
        return args.answer
    if args.answer_file:
        return Path(args.answer_file).read_text(encoding="utf-8")
    raise SystemExit("Use --answer or --answer-file.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify answer citations from Qdrant payloads without DB conversion."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--collection", type=str, help="Qdrant 컬렉션명. 생략하면 config.yaml에서 읽습니다.")
    parser.add_argument("--embed-provider", type=str, help="config.yaml의 collection_name 선택용 provider.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="DB에서 읽을 최대 point 수. 디버깅용입니다.")
    parser.add_argument("--chunks", type=Path, help="선택 사항: DB 대신 기존 JSON 청크 파일을 읽습니다.")
    parser.add_argument("--answer", type=str)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--fallback-top-k", type=int, default=5)

    args = parser.parse_args()

    if args.chunks:
        chunks = load_chunks(args.chunks)
        source_label = f"json:{args.chunks}"
    else:
        collection_name = resolve_collection_name(
            config_path=args.config,
            collection_name=args.collection,
            embed_provider=args.embed_provider,
        )
        chunks = load_chunks_from_db(
            collection_name=collection_name,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        source_label = f"qdrant:{collection_name}"

    if not chunks:
        raise SystemExit(f"검증할 청크를 찾을 수 없습니다. source={source_label}")

    answer = read_answer(args)
    result = verify_answer(answer, chunks, args.fallback_top_k)
    result["source"] = source_label
    result["chunk_count"] = len(chunks)

    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(result_json)


if __name__ == "__main__":
    main()
