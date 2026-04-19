"""
Script Semi-Otomatis: Mencari relevant_chunks untuk setiap pertanyaan.

Cara kerja:
1. Memuat chunks.json dan questions.json
2. Untuk setiap pertanyaan, mengekstrak kata kunci dari ground_truth_answer
3. Mencocokkan kata kunci dengan isi setiap chunk
4. Mengurutkan berdasarkan skor kecocokan
5. Menyimpan top-N chunk sebagai relevant_chunks di questions.json

Output: questions.json yang sudah ditambahkan field "relevant_chunks"

PENTING: Hasil ini perlu DIVERIFIKASI secara manual!
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

# === Konfigurasi ===
CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
OUTPUT_PATH = Path(__file__).parent / "questions.json"  # overwrite
TOP_N = 5  # Ambil top-N chunk per pertanyaan
MIN_SCORE = 3  # Minimal skor kecocokan (jumlah kata kunci yang cocok)

# Kata-kata umum yang diabaikan (stopwords Bahasa Indonesia sederhana)
STOPWORDS = {
    "yang", "dan", "di", "dari", "untuk", "dengan", "pada", "ke",
    "ini", "itu", "atau", "juga", "adalah", "dalam", "tidak",
    "dapat", "akan", "telah", "sudah", "oleh", "serta", "bahwa",
    "bagi", "secara", "sebagai", "agar", "jika", "maka", "melalui",
    "setiap", "sesuai", "harus", "wajib", "yaitu", "antara",
    "berupa", "lain", "saat", "lebih", "ada", "tersebut", "maupun",
    "baik", "namun", "masih", "bisa", "dua", "satu", "tiga",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "1", "2",
    "3", "4", "5", "6", "7", "8", "9", "10", "dst", "dll",
    "the", "and", "for", "of", "to", "in", "with",
}


def extract_keywords(text: str) -> list[str]:
    """Ekstrak kata kunci dari teks (buang stopwords, ambil kata >= 3 huruf)."""
    # Lowercase dan split
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Filter: buang stopwords dan kata pendek
    keywords = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    return keywords


def score_chunk(chunk_content: str, keywords: list[str]) -> int:
    """Hitung skor kecocokan antara chunk dan kata kunci."""
    content_lower = chunk_content.lower()
    score = 0
    for kw in keywords:
        if kw in content_lower:
            score += 1
    return score


def find_relevant_chunks(chunks: list, question: dict, top_n: int = TOP_N) -> list[dict]:
    """Cari chunk yang paling relevan untuk sebuah pertanyaan."""
    gt = question.get("ground_truth_answer", "")
    q_text = question.get("question", "")

    # Gabungkan kata kunci dari ground truth + pertanyaan
    keywords = extract_keywords(gt)
    q_keywords = extract_keywords(q_text)

    # Prioritaskan kata kunci dari ground truth, tambahkan dari pertanyaan
    all_keywords = keywords + [k for k in q_keywords if k not in keywords]

    # Hitung unique keywords untuk normalisasi
    unique_keywords = list(set(all_keywords))

    # Skor setiap chunk
    scored = []
    for chunk in chunks:
        content = chunk.get("content", "")
        s = score_chunk(content, unique_keywords)
        if s >= MIN_SCORE:
            scored.append({
                "id": chunk["id"],
                "score": s,
                "max_score": len(unique_keywords),
                "percentage": round(s / len(unique_keywords) * 100, 1) if unique_keywords else 0,
                "preview": content[:150].replace("\n", " "),
                "source": chunk.get("metadata", {}).get("source", "unknown"),
            })

    # Urutkan berdasarkan skor (tertinggi dulu)
    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_n]


def main():
    print("=" * 70)
    print("PENCARIAN RELEVANT CHUNKS (Semi-Otomatis)")
    print("=" * 70)

    # Load data
    print(f"\nMemuat chunks dari: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  → {len(chunks)} chunks dimuat")

    print(f"Memuat pertanyaan dari: {QUESTIONS_PATH}")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    print(f"  → {len(questions)} pertanyaan dimuat")

    # Proses setiap pertanyaan
    print(f"\nMencari relevant chunks (top {TOP_N}, min score {MIN_SCORE})...")
    print("-" * 70)

    total_found = 0
    results_summary = []

    for i, q in enumerate(questions):
        q_id = q["id"]
        q_text = q["question"]

        candidates = find_relevant_chunks(chunks, q, TOP_N)

        # Simpan hanya ID ke relevant_chunks
        q["relevant_chunks"] = [c["id"] for c in candidates]

        # Tampilkan progress
        print(f"\n[{q_id}] {q_text[:70]}...")
        print(f"  Ditemukan {len(candidates)} chunk relevan:")
        for j, c in enumerate(candidates):
            print(f"    {j+1}. [{c['score']}/{c['max_score']} = {c['percentage']}%] {c['id']}")
            print(f"       Sumber: {c['source']}")
            print(f"       Preview: {c['preview'][:100]}...")

        total_found += len(candidates)
        results_summary.append({
            "id": q_id,
            "found": len(candidates),
        })

    # Simpan hasil
    print("\n" + "=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    print(f"Total pertanyaan: {len(questions)}")
    print(f"Total chunk relevan ditemukan: {total_found}")
    print(f"Rata-rata chunk per pertanyaan: {total_found / len(questions):.1f}")

    # Pertanyaan tanpa chunk
    empty = [r for r in results_summary if r["found"] == 0]
    if empty:
        print(f"\n⚠️  {len(empty)} pertanyaan TANPA chunk relevan:")
        for r in empty:
            print(f"    - {r['id']}")

    # Simpan
    print(f"\nMenyimpan hasil ke: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("✓ Selesai!")

    print(f"\n⚠️  PENTING: Verifikasi hasil relevant_chunks di {OUTPUT_PATH}")
    print("   Buka file dan periksa apakah chunk yang dipilih memang relevan.")


if __name__ == "__main__":
    main()
