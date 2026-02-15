from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> list:
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages
import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    return text.strip()
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list:
    sentences = sent_tokenize(text)
    chunks = []

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))

            # overlap handling
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = [" ".join(overlap_words)]
            current_length = len(overlap_words)

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
def ingest_pdf(pdf_path: str) -> list:
    pages = extract_text_from_pdf(pdf_path)
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
if __name__ == "__main__":
    chunks = ingest_pdf("sample_logistics.pdf")

    print(f"Total chunks: {len(chunks)}\n")
    print("Sample chunk:\n")
    print(chunks[0])
