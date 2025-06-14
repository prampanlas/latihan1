print("Mulai membangun index vektor...")

import os
import pickle
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss

# Konfigurasi
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBEDDING_MODEL = "intfloat/e5-small-v2"

INPUT_FILE = Path("processed/text_chunks.pkl")
OUTPUT_INDEX = Path("processed/vectorstore.faiss")
OUTPUT_METADATA = Path("processed/metadata.pkl")


def split_into_chunks(paragraphs, chunk_size, overlap):
    chunks = []
    idx = 0
    for item in paragraphs:
        file_name = item["file_name"]
        texts = item["chunks"]

        joined = " ".join(texts)
        words = joined.split()

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append({
                "file_name": file_name,
                "text": chunk,
                "chunk_id": idx,
                "start_word": start,
                "end_word": end
            })
            idx += 1
            start += chunk_size - overlap
    return chunks


def main():
    print("🔍 Memuat teks dari:", INPUT_FILE)
    with open(INPUT_FILE, "rb") as f:
        paragraphs = pickle.load(f)

    print("✂️ Melakukan chunking teks...")
    chunks = split_into_chunks(paragraphs, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"📦 Total chunk: {len(chunks)}")
    texts = [chunk["text"] for chunk in chunks]

    print("🔗 Memuat model embedding:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("📊 Membuat embedding...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("🧠 Membuat index FAISS...")
    dim = embeddings.shape[1]
    print(f"✅ Ukuran dimensi embedding: {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    OUTPUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUTPUT_INDEX))

    with open(OUTPUT_METADATA, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Index dan metadata selesai dibuat.")
    print(f"📁 Index: {OUTPUT_INDEX}")
    print(f"📁 Metadata: {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
