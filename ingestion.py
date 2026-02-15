"""
Document Ingestion Module
-------------------------
Responsibilities:
- Extract text from PDF files (page-aware)
- Clean raw text
- Chunk text into overlapping, semantically coherent units
- Attach metadata required for retrieval and citation
"""

from typing import List, Dict
from pypdf import PdfReader
import re
import nltk
from nltk.tokenize import sent_tokenize

# Ensure tokenizer is available
nltk.download("punkt", quiet=True)

# -----------------------------
# Configuration (tunable knobs)
# -----------------------------

CHUNK_SIZE = 500-600       # approx words per chunk
CHUNK_OVERLAP = 100     # approx words overlap


# -----------------------------
# PDF Text Extraction
# -----------------------------

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF file page by page.

    Returns:
        List of dicts with keys:
        - page: int
        - text: str
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception:
            text = None

        if text and text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text: str) -> str:
    """
    Perform minimal cleaning while preserving semantic meaning.
    """
    text = re.sub(r"\s+", " ", text)   # normalize whitespace
    return text.strip()


# -----------------------------
# Chunking with Overlap
# -----------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into sentence-aware chunks with overlap.

    Args:
        text: Cleaned text
        chunk_size: target size in words
        overlap: overlapping words between chunks

    Returns:
        List of chunk strings
    """
    sentences = sent_tokenize(text)
    chunks = []

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        # If adding this sentence exceeds chunk size, finalize chunk
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))

            # Apply overlap
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = [" ".join(overlap_words)]
            current_length = len(overlap_words)

        current_chunk.append(sentence)
        current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -----------------------------
# Full Ingestion Pipeline
# -----------------------------

def ingest_pdf(pdf_path: str) -> List[Dict]:
    """
    End-to-end ingestion pipeline.

    Input:
        pdf_path: path to PDF document

    Output:
        List of chunks with metadata:
        {
            "doc_id": str,
            "page": int,
            "chunk_id": int,
            "text": str
        }
    """
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        raise ValueError("No extractable text found in PDF.")

    all_chunks = []

    for page_data in pages:
        cleaned_text = clean_text(page_data["text"])
        chunks = chunk_text(cleaned_text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": pdf_path,
                "page": page_data["page"],
                "chunk_id": idx,
                "text": chunk
            })

    return all_chunks


# -----------------------------
# Optional Local Test
# -----------------------------

if __name__ == "__main__":
    # Simple manual test (adjust path if needed)
    test_pdf = "data/sample_logistics.pdf"
    chunks = ingest_pdf(test_pdf)

    print(f"Total chunks created: {len(chunks)}\n")
    print("Sample chunk:\n")
    print(chunks[0])
