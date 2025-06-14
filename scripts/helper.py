print("Loading helper functions...")

import fitz  # PyMuPDF

def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    page_stats = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        local_paragraphs = []
        total_chars = 0

        for block in blocks:
            text = block[4].strip()
            total_chars += len(text)
            if len(text) > 30:
                local_paragraphs.append(text)

        paragraphs.extend(local_paragraphs)

        page_stats.append({
            "page": page_num + 1,
            "char_count": total_chars,
            "paragraphs": len(local_paragraphs)
        })

    return paragraphs, page_stats

def summarize_page_stats(file_name, para_count, stats):
    total_chars_all = sum(p["char_count"] for p in stats)
    empty_pages = [p["page"] for p in stats if p["char_count"] < 50]

    print(f"📄 {file_name}: {len(stats)} halaman, {total_chars_all} karakter, {para_count} paragraf.")
    if empty_pages:
        print(f"⚠️ Halaman minim teks: {empty_pages}")
