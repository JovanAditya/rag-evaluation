# 📊 Evaluasi RAG: Baseline vs Hybrid

Modul evaluasi untuk membandingkan performa **RAG Baseline** (vector search) dengan **RAG Hybrid** (BM25 + vector + reranking).

## 📁 Struktur

```
evaluation/
├── config.yaml           # Konfigurasi evaluasi
├── questions.json        # Dataset pertanyaan + ground truth
├── run_evaluation.py     # Script utama evaluasi
├── generate_report.py    # Generate laporan & visualisasi
├── results/              # Hasil evaluasi (auto-generated)
└── README.md
```

## 🚀 Cara Penggunaan

### 1. Install Dependencies

```bash
# Dari root rag-deploy/
conda activate academic-rag

# Install tambahan untuk visualisasi
pip install matplotlib pyyaml
```

### 2. Set Environment Variables

```bash
export DATA_PATH=/path/to/rag-deploy/data
export INDEX_CACHE_DIR=/path/to/rag-deploy/data/cache
export CHROMA_PERSIST_DIRECTORY=/path/to/rag-deploy/data/chroma_db
```

### 3. Jalankan Evaluasi

```bash
cd evaluation/

# Full evaluation (20 pertanyaan × 3 run)
python run_evaluation.py

# Quick test (5 pertanyaan × 1 run)
python run_evaluation.py --limit 5 --runs 1

# Custom config
python run_evaluation.py --config config.yaml --output results/
```

### 4. Generate Laporan

```bash
python generate_report.py --input results/evaluation_YYYYMMDD_HHMMSS.json
```

Output:
- `results/report.html` - Laporan HTML lengkap
- `results/time_comparison.png` - Grafik perbandingan waktu
- `results/quality_comparison.png` - Grafik perbandingan kualitas
- `results/radar_comparison.png` - Radar chart multi-dimensi
- `results/per_question_time.png` - Perbandingan per pertanyaan

## 📋 Metrik Evaluasi

| Kategori | Metrik | Deskripsi |
|----------|--------|-----------|
| Retrieval | Avg Relevance Score | Rata-rata skor relevansi dokumen |
| Retrieval | Documents Retrieved | Jumlah dokumen yang diambil |
| Quality | Confidence | Skor kepercayaan sistem |
| Performance | Retrieval Time | Waktu pencarian dokumen |
| Performance | Generation Time | Waktu generate jawaban |
| Performance | Total Time | Waktu total per query |

## 📝 Kustomisasi Pertanyaan

Edit `questions.json` untuk menambah/mengubah pertanyaan:

```json
{
  "id": "Q021",
  "question": "Pertanyaan baru?",
  "ground_truth_answer": "Jawaban referensi",
  "category": "prosedur",
  "difficulty": "medium"
}
```
