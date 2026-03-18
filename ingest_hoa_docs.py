#!/usr/bin/env python3
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent

if len(sys.argv) > 1:
    HOA_NAME = sys.argv[1].strip().lower()
else:
    HOA_NAME = "milestone"

DATASET_DIR = BASE_DIR / "data" / HOA_NAME
DOCS_DIR = DATASET_DIR / "hoa_docs"
PROCESSED_DIR = DATASET_DIR / "processed"
INDEX_DIR = DATASET_DIR / "index"
OCR_DIR = DATASET_DIR / "ocr_cache"

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
FAISS_PATH = INDEX_DIR / "hoa_index.faiss"
META_PATH = INDEX_DIR / "hoa_index_meta.json"

print(f"[config] HOA_NAME={HOA_NAME}")
print(f"[config] DOCS_DIR={DOCS_DIR}")
print(f"[config] INDEX_DIR={INDEX_DIR}")

EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_TARGET_CHARS = 1100
CHUNK_OVERLAP_CHARS = 180

USE_OCRMYPDF_FALLBACK = True

FORCE_OCR_FILES = {
    "P.I.C.A.E.-Covenant6641909.1.pdf",
}

SKIP_FILES = {
    # "Milestone-HOA-Exhibit-A-Supplements_v2s.pdf",
}

STOPWORDS = {
    "the", "and", "for", "that", "with", "shall", "this", "from", "are", "not",
    "any", "all", "may", "have", "has", "but", "its", "was", "were", "their",
    "such", "into", "than", "then", "hereof", "thereof", "within", "which",
    "upon", "each", "lot", "association", "owner", "owners", "section", "article",
    "declaration", "bylaws", "property", "common", "area", "board", "members",
    "will", "said", "been", "being", "also", "other", "under", "more", "these",
    "those", "them", "they", "his", "her", "its", "our", "you", "your", "than",
    "had", "can", "could", "should", "would", "where", "when", "what", "who",
}

IMPORTANT_TERMS = {
    "parking", "vehicle", "vehicles", "commercial", "garage", "street", "lease",
    "leasing", "tenant", "rent", "rental", "assessment", "assessments", "lien",
    "meeting", "members", "board", "trash", "garbage", "recycling", "pets",
    "architectural", "approval", "committee", "nuisance", "rule", "rules",
    "restriction", "restrictions", "fine", "fines", "dues", "maintenance",
}


def clean_text_basic(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\uf0b7", " ")
    text = text.replace("•", " ")
    text = text.replace("·", " ")
    text = text.replace("\\", " ")
    text = text.replace(" ", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-")

    # Join hyphenated line breaks before flattening whitespace
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove long numeric stamps / recording ids that are often noise
    text = re.sub(r"\b\d{6,}\b", " ", text)

    return text.strip()


def split_lines_preserve_structure(text: str) -> list[str]:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    return [line.strip() for line in text.split("\n")]


def looks_like_heading(line: str) -> bool:
    if not line:
        return False

    line = re.sub(r"\s+", " ", line).strip()
    low = line.lower()

    if len(line) > 160:
        return False

    if re.match(r"^(article|section)\b", low):
        return True

    if re.match(r"^\d+(\.\d+){1,4}\b", line):
        return True

    if line.endswith(":") and len(line) < 100:
        return True

    alpha = sum(ch.isalpha() for ch in line)
    upper = sum(ch.isupper() for ch in line if ch.isalpha())

    if alpha >= 6 and upper / max(alpha, 1) > 0.75 and len(line) < 100:
        return True

    title_words = line.split()
    if 1 <= len(title_words) <= 10 and alpha >= 6:
        if all(w[:1].isupper() or w.lower() in {"and", "of", "the", "for", "to"} for w in title_words):
            return True

    return False


def extract_page_heading_candidates(raw_text: str) -> list[str]:
    headings = []
    for line in split_lines_preserve_structure(raw_text):
        line = normalize_line(line)
        if not line or is_noise_line(line):
            continue
        if looks_like_heading(line):
            headings.append(line)
    return headings


def pick_best_heading(headings: list[str]) -> str | None:
    if not headings:
        return None

    def heading_score(h: str) -> tuple[int, int]:
        score = 0
        low = h.lower()
        if low.startswith("section"):
            score += 5
        if low.startswith("article"):
            score += 4
        if re.match(r"^\d+(\.\d+){1,4}\b", h):
            score += 4
        if ":" in h:
            score += 1
        if len(h) <= 80:
            score += 1
        return (score, -len(h))

    return sorted(headings, key=heading_score, reverse=True)[0]

def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line


def is_noise_line(line: str) -> bool:
    if not line:
        return True

    low = line.lower().strip()

    if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", low):
        return True
    if re.fullmatch(r"\d+", low):
        return True
    if re.fullmatch(r"[ivxlcdm]+", low):
        return True
    if re.fullmatch(r"[_\-. ]{3,}", line):
        return True
    if re.fullmatch(r"[a-z]{0,3}\d*[a-z\d\- ]{0,6}", low) and len(low) <= 8:
        return True

    alpha = sum(ch.isalpha() for ch in line)
    digit = sum(ch.isdigit() for ch in line)
    total = max(len(line), 1)

    if len(line) < 3:
        return True
    if alpha == 0 and digit > 0:
        return True
    if alpha / total < 0.20 and len(line) < 40:
        return True

    return False


def remove_noise_lines(text: str) -> str:
    lines = [normalize_line(x) for x in text.split("\n")]
    cleaned = []

    for line in lines:
        if is_noise_line(line):
            continue

        # Remove repeated "Page X of Y" fragments embedded in longer lines
        line = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def merge_broken_words(text: str) -> str:
    # Fix OCR-spaced words like "pa rking" or "res trictions" cautiously
    def repl(match):
        a, b = match.group(1), match.group(2)
        merged = a + b
        if len(a) <= 3 and len(b) <= 8:
            return merged
        return match.group(0)

    text = re.sub(r"\b([A-Za-z]{1,3})\s+([A-Za-z]{2,8})\b", repl, text)

    # Fix all-caps spaced letter runs like H O A
    text = re.sub(r"\b([A-Z])\s+([A-Z])\s+([A-Z])\b", r"\1\2\3", text)

    return text


def clean_text(text: str) -> str:
    text = clean_text_basic(text)
    text = remove_noise_lines(text)
    text = merge_broken_words(text)

    # Flatten leftover line breaks into prose-friendly spaces
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def guess_doc_type(filename: str) -> str:
    name = filename.lower()
    if "bylaws" in name:
        return "bylaws"
    if "declaration" in name or "covenant" in name or "restrictions" in name:
        return "ccrs"
    if "policy" in name:
        return "policy"
    if "welcome" in name:
        return "welcome"
    if "conduct" in name:
        return "board_policy"
    if "exhibit" in name:
        return "exhibit"
    return "other"


def quality_metrics(text: str) -> dict:
    text = text or ""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", text)
    total_chars = max(len(text), 1)

    alpha = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    weird = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}'\"-/%$&:@#") for ch in text)

    alpha_ratio = alpha / total_chars
    digit_ratio = digits / total_chars
    weird_ratio = weird / total_chars
    word_count = len(words)

    short_words = sum(1 for w in words if len(w) <= 2)
    short_word_ratio = short_words / max(word_count, 1)

    unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)

    important_hits = sum(1 for w in words if w.lower() in IMPORTANT_TERMS)

    score = 1.0
    score -= max(0.0, 0.45 - alpha_ratio) * 2.2
    score -= max(0.0, weird_ratio - 0.10) * 3.5
    score -= max(0.0, short_word_ratio - 0.35) * 1.5
    score -= 0.15 if word_count < 20 else 0.0
    score -= 0.10 if unique_ratio < 0.35 else 0.0
    score += min(0.20, important_hits * 0.02)

    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "word_count": word_count,
        "alpha_ratio": round(alpha_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "weird_ratio": round(weird_ratio, 4),
        "short_word_ratio": round(short_word_ratio, 4),
        "unique_ratio": round(unique_ratio, 4),
        "important_hits": important_hits,
    }


def is_low_quality_text(text: str) -> bool:
    m = quality_metrics(text)
    return (
        m["word_count"] < 12
        or m["score"] < 0.38
        or m["alpha_ratio"] < 0.40
        or m["weird_ratio"] > 0.22
    )


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    counts = Counter(words)

    ranked = sorted(
        counts.items(),
        key=lambda kv: (
            kv[1] + (3 if kv[0] in IMPORTANT_TERMS else 0),
            len(kv[0]),
        ),
        reverse=True,
    )
    return [w for w, _ in ranked[:limit]]


def sentence_aware_chunks(text: str, target_chars: int = CHUNK_TARGET_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?;])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    chunks = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
            continue

        if len(current) + 1 + len(sent) <= target_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            if overlap_chars > 0:
                overlap = current[-overlap_chars:].strip()
                current = (overlap + " " + sent).strip()
            else:
                current = sent

    if current.strip():
        chunks.append(current.strip())

    # Fallback if any chunk is still too large
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= target_chars * 1.35:
            final_chunks.append(chunk)
            continue

        start = 0
        while start < len(chunk):
            end = min(start + target_chars, len(chunk))
            if end < len(chunk):
                boundary = chunk.rfind(". ", start, end)
                if boundary != -1 and boundary > start + 300:
                    end = boundary + 1
            piece = chunk[start:end].strip()
            if piece:
                final_chunks.append(piece)
            if end >= len(chunk):
                break
            start = max(end - overlap_chars, start + 1)

    return final_chunks


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    results = []

    last_heading = None

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        heading_candidates = extract_page_heading_candidates(raw_text)
        page_heading = pick_best_heading(heading_candidates) or last_heading

        if page_heading:
            last_heading = page_heading

        results.append(
            {
                "source_file": pdf_path.name,
                "page": page_num,
                "doc_type": guess_doc_type(pdf_path.name),
                "raw_text": raw_text,
                "text": cleaned,
                "section_title": page_heading,
                "heading_candidates": heading_candidates,
            }
        )

    return results


def pdf_needs_ocr(pages: list[dict]) -> bool:
    non_empty_pages = [p for p in pages if p["text"].strip()]
    if not non_empty_pages:
        return True

    low_quality_count = sum(is_low_quality_text(p["text"]) for p in non_empty_pages)
    ratio = low_quality_count / max(len(non_empty_pages), 1)
    return ratio >= 0.5


def is_ocrmypdf_available() -> bool:
    return shutil.which("ocrmypdf") is not None


def ocr_pdf(input_pdf: Path) -> Path | None:
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    output_pdf = OCR_DIR / input_pdf.name

    if output_pdf.exists():
        print(f"[ocr] using cached OCR PDF: {output_pdf.name}")
        return output_pdf

    if not is_ocrmypdf_available():
        print("[ocr] ocrmypdf not found on PATH; skipping OCR fallback")
        return None

    cmd = [
        "ocrmypdf",
        "--force-ocr",
        "--rotate-pages",
        "--deskew",
        str(input_pdf),
        str(output_pdf),
    ]

    print(f"[ocr] running OCR on: {input_pdf.name}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[ocr] wrote OCR PDF: {output_pdf}")
        return output_pdf
    except subprocess.CalledProcessError as e:
        print(f"[ocr] failed for {input_pdf.name}: {e}")
        return None


def extract_best_pdf_pages(pdf_path: Path) -> tuple[list[dict], bool]:
    pages = extract_pdf_pages(pdf_path)
    force_ocr = pdf_path.name in FORCE_OCR_FILES

    if not USE_OCRMYPDF_FALLBACK:
        return pages, False

    if not force_ocr and not pdf_needs_ocr(pages):
        return pages, False

    print(f"[ocr] low-quality extraction detected in {pdf_path.name}")
    ocr_pdf_path = ocr_pdf(pdf_path)
    if not ocr_pdf_path:
        return pages, False

    ocr_pages = extract_pdf_pages(ocr_pdf_path)

    original_non_empty = [p for p in pages if p["text"].strip()]
    ocr_non_empty = [p for p in ocr_pages if p["text"].strip()]

    original_bad = sum(is_low_quality_text(p["text"]) for p in original_non_empty)
    ocr_bad = sum(is_low_quality_text(p["text"]) for p in ocr_non_empty)

    if not ocr_non_empty:
        return pages, False

    if force_ocr or ocr_bad < original_bad:
        print(f"[ocr] using OCR text for {pdf_path.name}")
        for page in ocr_pages:
            page["source_file"] = pdf_path.name
            page["doc_type"] = guess_doc_type(pdf_path.name)
        return ocr_pages, True

    print(f"[ocr] OCR did not improve {pdf_path.name}; using original extraction")
    return pages, False


def build_chunks() -> list[dict]:
    all_chunks: list[dict] = []

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    txt_files = sorted(DOCS_DIR.glob("*.txt"))

    if not pdf_files and not txt_files:
        raise FileNotFoundError(f"No PDF or TXT files found in {DOCS_DIR}")

    for path in pdf_files:
        if path.name in SKIP_FILES:
            print(f"[skip] skipped file: {path.name}")
            continue

        print(f"[ingest] reading PDF: {path.name}")
        pages, used_ocr = extract_best_pdf_pages(path)
        if used_ocr:
            print(f"[ingest] OCR fallback used for: {path.name}")

        page_count = 0
        warned_pages = 0
        chunk_count = 0
        skipped_chunks = 0

        for page_data in pages:
            page_text = page_data["text"]
            if not page_text.strip():
                continue

            page_count += 1
            page_quality = quality_metrics(page_text)

            if is_low_quality_text(page_text):
                warned_pages += 1
                print(
                    f"[warn] low-quality page text: {path.name} page {page_data['page']} "
                    f"(score={page_quality['score']:.3f})"
                )

            page_chunks = sentence_aware_chunks(page_text)

            for idx, chunk_text in enumerate(page_chunks, start=1):
                chunk_quality = quality_metrics(chunk_text)

                if is_low_quality_text(chunk_text):
                    skipped_chunks += 1
                    print(
                        f"[skip] low-quality chunk: {path.name} "
                        f"page {page_data['page']} chunk {idx} "
                        f"(score={chunk_quality['score']:.3f})"
                    )
                    continue

                keywords = extract_keywords(chunk_text)

                all_chunks.append(
                    {
                        "id": f"{path.name}::p{page_data['page']}::c{idx}",
                        "source_file": path.name,
                        "page": page_data["page"],
                        "doc_type": page_data["doc_type"],
                        "chunk_index": idx,
                        "text": chunk_text,
                        "section_title": page_data.get("section_title"),
                        "heading_candidates": page_data.get("heading_candidates", []),
                        "keywords": keywords,
                        "quality_score": chunk_quality["score"],
                        "page_quality_score": page_quality["score"],
                        "word_count": chunk_quality["word_count"],
                        "important_hits": chunk_quality["important_hits"],
                        "used_ocr": used_ocr,
                    }
                )
                chunk_count += 1

        print(
            f"[ingest] kept pages={page_count}, warned pages={warned_pages}, "
            f"kept chunks={chunk_count}, skipped low-quality chunks={skipped_chunks}"
        )

    for path in txt_files:
        if path.name in SKIP_FILES:
            print(f"[skip] skipped file: {path.name}")
            continue

        print(f"[ingest] reading TXT: {path.name}")
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            print(f"[warn] no text found in {path.name}")
            continue

        doc_type = guess_doc_type(path.name)
        chunk_count = 0
        skipped_chunks = 0

        for idx, chunk_text in enumerate(sentence_aware_chunks(text), start=1):
            chunk_quality = quality_metrics(chunk_text)

            if is_low_quality_text(chunk_text):
                skipped_chunks += 1
                print(
                    f"[skip] low-quality chunk: {path.name} chunk {idx} "
                    f"(score={chunk_quality['score']:.3f})"
                )
                continue

            keywords = extract_keywords(chunk_text)

            all_chunks.append(
                {
                    "id": f"{path.name}::c{idx}",
                    "source_file": path.name,
                    "page": None,
                    "doc_type": doc_type,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "keywords": keywords,
                    "quality_score": chunk_quality["score"],
                    "page_quality_score": chunk_quality["score"],
                    "word_count": chunk_quality["word_count"],
                    "important_hits": chunk_quality["important_hits"],
                    "used_ocr": False,
                    "section_title": None,
                    "heading_candidates": [],
                }
            )
            chunk_count += 1

        print(
            f"[ingest] kept chunks={chunk_count}, "
            f"skipped low-quality chunks={skipped_chunks}"
        )

    return all_chunks


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = build_chunks()
    if not all_chunks:
        raise RuntimeError("No chunks produced from source documents.")

    print(f"[ingest] total chunks: {len(all_chunks)}")
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    matrix = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(FAISS_PATH))
    META_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

    avg_quality = sum(c["quality_score"] for c in all_chunks) / max(len(all_chunks), 1)
    avg_words = sum(c["word_count"] for c in all_chunks) / max(len(all_chunks), 1)

    print(f"[stats] avg chunk quality score: {avg_quality:.3f}")
    print(f"[stats] avg chunk word count:   {avg_words:.1f}")
    print(f"[done] wrote chunks to: {CHUNKS_PATH}")
    print(f"[done] wrote index to:  {FAISS_PATH}")
    print(f"[done] wrote meta to:   {META_PATH}")


if __name__ == "__main__":
    main()