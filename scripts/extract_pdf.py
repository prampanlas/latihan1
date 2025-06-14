print("📥 Mulai ekstraksi teks dari file PDF...")

import os
import pickle
import json
from pathlib import Path
from tqdm import tqdm

from helper import extract_text_blocks, summarize_page_stats

# Lokasi input/output
INPUT_DIR = Path("data/pdf/")
OUTPUT_DIR = Path("processed/")
OUTPUT_FILE = OUTPUT_DIR / "text_chunks.pkl"

def main():
    all_chunks = []
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ Tidak ada PDF ditemukan di folder data/pdf/")
        return

    for pdf_file in tqdm(pdf_files, desc="Memproses PDF"):
        try:
            paras, page_stats = extract_text_blocks(pdf_file)

            if paras and len(" ".join(paras)) > 500:
                all_chunks.append({
                    "file_name": pdf_file.name,
                    "chunks": paras
                })
            else:
                print(f"⚠️ File {pdf_file.name} memiliki konten sangat sedikit. Dilewati.")

            # Simpan statistik halaman
            stats_path = OUTPUT_DIR / f"{pdf_file.stem}_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(page_stats, f, indent=2)

            summarize_page_stats(pdf_file.name, len(paras), page_stats)

        except Exception as e:
            print(f"❌ Gagal memproses {pdf_file.name}: {e}")

    # Simpan hasil ekstraksi ke file pickle
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n✅ Selesai. {len(all_chunks)} file diproses. Disimpan di {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

