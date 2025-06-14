import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import numpy as np

# Konfigurasi
VECTOR_INDEX_PATH = Path("processed/vectorstore.faiss")
METADATA_PATH = Path("processed/metadata.pkl")
EMBEDDING_MODEL = "intfloat/e5-small-v2"
LLM_MODEL_PATH = "models/nous-hermes-2.gguf"
TOP_K = 5

# Prompt template dasar
PROMPT_TEMPLATE = """
Jawablah pertanyaan berdasarkan konteks berikut.
Jika tidak tahu, jawab: 'Saya tidak tahu berdasarkan dokumen ini.'

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""

def load_faiss_index(index_path):
    index = faiss.read_index(str(index_path))
    return index

def load_metadata(metadata_path):
    with open(metadata_path, "rb") as f:
        return pickle.load(f)

def get_top_k_chunks(query, model, index, metadata, k):
    embedding = model.encode([query], convert_to_numpy=True)
    _, I = index.search(embedding, k)
    return [metadata[i]["text"] for i in I[0]]

def main():
    # Load vectorstore dan metadata
    print("🔍 Memuat index dan metadata...")
    index = load_faiss_index(VECTOR_INDEX_PATH)
    metadata = pickle.load(open(METADATA_PATH, "rb"))

    # Load model embedding
    print("🔗 Memuat model embedding...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load LLM lokal
    print("🧠 Memuat model LLM lokal...")
    llm = Llama(model_path=str(LLM_MODEL_PATH), n_ctx=4096, n_threads=4, n_gpu_layers=35)

    print("✅ Sistem siap. Ketik pertanyaanmu.")
    while True:
        question = input("\n❓ Pertanyaan: ")
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Keluar.")
            break

        top_chunks = get_top_k_chunks(question, embed_model, index, metadata, TOP_K)
        context = "\n---\n".join(top_chunks)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        print("📝 Menjawab...")
        output = llm(prompt, max_tokens=512, stop=["\n\n"])
        print("\n💬 Jawaban:")
        print(output["choices"][0]["text"].strip())

if __name__ == "__main__":
    main()