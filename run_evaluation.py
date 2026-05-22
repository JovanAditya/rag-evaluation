"""
RAG Evaluation Script - Perbandingan Baseline vs Hybrid.

Menjalankan evaluasi terhadap kedua pipeline RAG menggunakan
dataset pertanyaan akademik dan mengumpulkan metrik performa.

Metrik yang dihitung:
  - Retrieval: MRR, Precision@K, Recall@K
  - RAGAS (opsional): Faithfulness, Answer Relevancy, Context Precision
  - Operasional: waktu retrieval/generasi, jumlah dokumen, dll.
  - Statistik: Mean, Std, Shapiro-Wilk, t-Test/Wilcoxon
"""

import os
import sys
import json
import time
import yaml
import logging
import argparse
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add rag-model to path (use root rag-model, not rag-api submodule)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAG_MODEL_DIR = PROJECT_ROOT / "rag-model"
sys.path.insert(0, str(RAG_MODEL_DIR))

# Load .env dari rag-model directory (prioritas konfigurasi generator)
try:
    from dotenv import load_dotenv
    # Load .env dari rag-model
    model_env = RAG_MODEL_DIR / ".env"
    if model_env.exists():
        load_dotenv(model_env)
        print(f"✅ Loaded environment from: {model_env}")
    else:
        # Fallback ke root jika tidak ada
        load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    # Fallback: baca .env secara manual dari rag-model
    env_path = RAG_MODEL_DIR / ".env"
    if not env_path.exists():
        env_path = PROJECT_ROOT / ".env"
        
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
        print(f"✅ Loaded environment manually from: {env_path}")

from rag_model.core.config import RAGConfig, RetrievalConfig, IndexConfig
from rag_model.core.pipeline import AcademicRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Konfigurasi retry untuk error 503/overloaded/rate-limit
MAX_RETRIES = 100   # Diperbanyak agar tidak di-skip jika Gemini High Demand
INITIAL_WAIT = 15   # Detik awal tunggu sebelum retry
MAX_WAIT = 300      # Maksimum waktu tunggu (detik) (5 menit)
BACKOFF_FACTOR = 2  # Faktor pengali exponential backoff

# Pattern error yang bisa di-retry (503, 429, overloaded, dsb)
RETRYABLE_PATTERNS = [
    r"503",
    r"429",
    r"overloaded",
    r"high demand",
    r"rate.?limit",
    r"resource.?exhausted",
    r"service.?unavailable",
    r"too many requests",
    r"quota",
    r"capacity",
    r"temporarily",
    r"try again",
    r"server.?busy",
    r"RESOURCE_EXHAUSTED",
    r"ResourceExhausted",
]


def is_retryable_error(error_msg: str) -> bool:
    """
    Cek apakah error termasuk yang bisa di-retry.

    Mendeteksi: HTTP 503/429, Gemini overloaded/high demand,
    rate limit, resource exhausted, dsb.
    """
    if not error_msg:
        return False
    for pattern in RETRYABLE_PATTERNS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return True
    return False


def is_retryable_result(result: Dict[str, Any]) -> bool:
    """
    Cek apakah result evaluasi menunjukkan error yang bisa di-retry.

    Selain cek field 'error', juga cek apakah answer mengandung
    pesan error dari LLM generator (karena Gemini error di-catch
    dan dikembalikan sebagai answer dengan success=False).

    Kasus yang di-handle:
    - pipeline.query() berhasil (no exception) tapi Gemini return 503
    - _generate_gemini() menangkap error dan return {success: False, answer: pesan_error}
    - Dari sisi evaluator, result tetap success=True karena no exception
    - Retrieval berhasil (sources_count > 0), hanya generation yang gagal
    """
    # Cek error field
    if result.get("error") and is_retryable_error(str(result["error"])):
        return True

    # Cek answer yang sebenarnya pesan error dari LLM
    # Ini terjadi karena _generate_gemini() catch exception dan return
    # user-friendly message sebagai answer, bukan raise exception
    answer = result.get("answer", "")
    error_answer_patterns = [
        "server AI sedang sibuk",
        "kuota layanan AI",
        "tidak dapat terhubung",
        "terjadi kesalahan saat memproses",
    ]
    for pattern in error_answer_patterns:
        if pattern in answer:
            return True

    return False


# ============================================================
# Utility Functions
# ============================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load evaluation configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_questions(questions_path: str = "questions.json") -> List[Dict]:
    """Load evaluation questions."""
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def create_rag_pipeline(
    pipeline_type: str,
    config_overrides: Dict[str, Any],
    paths_config: Dict[str, str]
) -> AcademicRAG:
    """Create a RAG pipeline with specific configuration."""
    rag_config = RAGConfig()

    # Override pipeline settings
    rag_config.retrieval.pipeline_type = pipeline_type
    rag_config.retrieval.use_reranking = config_overrides.get("use_reranking", False)
    rag_config.retrieval.max_results = config_overrides.get("max_results", 5)

    if "bm25_weight" in config_overrides:
        rag_config.retrieval.bm25_weight = config_overrides["bm25_weight"]
    if "vector_weight" in config_overrides:
        rag_config.retrieval.vector_weight = config_overrides["vector_weight"]

    # Override paths
    rag_config.index.chroma_dir = paths_config.get("chroma_dir", "../data/chroma_db")
    rag_config.index.cache_dir = paths_config.get("cache_dir", "../data/cache")

    # Create pipeline in research mode
    return AcademicRAG(
        config=rag_config,
        research_mode=True,
        response_format="full"
    )


# ============================================================
# Retrieval Metrics: MRR, Precision@K, Recall@K
# ============================================================

def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 5
) -> Dict[str, float]:
    """
    Hitung metrik retrieval standar.

    Args:
        retrieved_ids: ID chunk yang di-retrieve oleh pipeline (urut berdasarkan ranking)
        relevant_ids: ID chunk yang relevan (ground truth)
        k: Jumlah top-K untuk Precision@K dan Recall@K

    Returns:
        Dict berisi mrr, precision_at_k, recall_at_k
    """
    if not relevant_ids:
        return {"mrr": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}

    relevant_set = set(relevant_ids)

    # MRR: 1/rank dokumen relevan pertama
    mrr = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_set:
            mrr = 1.0 / (i + 1)
            break

    # Precision@K: dokumen relevan dalam top-K / K
    relevant_in_topk = sum(1 for rid in retrieved_ids[:k] if rid in relevant_set)
    precision_at_k = relevant_in_topk / k if k > 0 else 0.0

    # Recall@K: dokumen relevan ditemukan / total dokumen relevan
    recall_at_k = relevant_in_topk / len(relevant_set) if relevant_set else 0.0

    return {
        "mrr": round(mrr, 4),
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
    }


def extract_retrieved_chunk_ids(sources: List[Dict]) -> List[str]:
    """
    Ekstrak ID chunk dari hasil retrieval pipeline.
    Mencoba beberapa field yang mungkin berisi chunk ID.
    """
    ids = []
    for s in sources:
        # Coba berbagai field yang mungkin
        chunk_id = (
            s.get("id")
            or s.get("chunk_id")
            or s.get("metadata", {}).get("id")
            or s.get("metadata", {}).get("chunk_id")
            or ""
        )
        if chunk_id:
            ids.append(chunk_id)
    return ids


# ============================================================
# RAGAS Metrics (Opsional)
# ============================================================

def compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    ragas_config: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """
    Hitung metrik RAGAS menggunakan LLM-as-a-judge.

    Mendukung Ollama (lokal) dan Gemini (API).
    Return None jika RAGAS tidak tersedia.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )
        from datasets import Dataset
    except ImportError:
        logger.warning("Library 'ragas' atau 'datasets' tidak tersedia. Lewati metrik RAGAS.")
        return None

    try:
        # Konfigurasi Provider (default fallback jika belum diubah di config)
        llm_provider = ragas_config.get("llm_provider", ragas_config.get("judge_model", "ollama"))
        embed_provider = ragas_config.get("embed_provider", ragas_config.get("judge_model", "ollama"))
        
        llm = None
        embeddings = None
        
        # 1. SETUP LLM JUDGE
        if llm_provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                ollama_model = ragas_config.get("ollama_model", "llama3.2:latest")
                llm = ChatOllama(model=ollama_model, format="json", temperature=0)
                logger.info(f"  RAGAS LLM: Ollama ({ollama_model})")
            except ImportError:
                logger.warning("langchain-ollama tidak tersedia. Install: pip install langchain-ollama")
                return None
        elif llm_provider == "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    logger.warning("GEMINI_API_KEY tidak ditemukan di environment.")
                    return None
                gemini_model = ragas_config.get("gemini_model", "gemini-2.5-flash")
                llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=api_key)
                logger.info(f"  RAGAS LLM: Gemini ({gemini_model})")
            except ImportError:
                logger.warning("langchain-google-genai tidak tersedia. Install: pip install langchain-google-genai")
                return None
                
        # 2. SETUP EMBEDDINGS
        if embed_provider == "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings
                ollama_embed_model = ragas_config.get("ollama_embed_model", "nomic-embed-text")
                embeddings = OllamaEmbeddings(model=ollama_embed_model)
                logger.info(f"  RAGAS Embeddings: Ollama ({ollama_embed_model})")
            except ImportError:
                logger.warning("langchain-ollama tidak tersedia untuk embeddings.")
                return None
        elif embed_provider == "gemini":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    logger.warning("GEMINI_API_KEY tidak ditemukan di environment.")
                    return None
                gemini_embed_model = ragas_config.get("gemini_embed_model", "gemini-embedding-001")
                embeddings = GoogleGenerativeAIEmbeddings(model=gemini_embed_model, google_api_key=api_key)
                logger.info(f"  RAGAS Embeddings: Gemini ({gemini_embed_model})")
            except Exception as e:
                logger.warning(f"Gagal memuat Gemini Embeddings: {e}")
                return None

        # Buat dataset untuk RAGAS
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        # Evaluasi dengan LLM yang dipilih
        eval_kwargs = {
            "dataset": dataset,
            "metrics": [faithfulness, answer_relevancy, context_precision],
        }
        if llm:
            eval_kwargs["llm"] = llm
        if embeddings:
            eval_kwargs["embeddings"] = embeddings

        result = evaluate(**eval_kwargs)

        # Helper untuk ekstrak dan rata-rata skor amankan dari nilai NaN/List
        def get_score(metric_name):
            try:
                # `result` bisa berupa dict/Dataset tergantung versi RAGAS
                val = result[metric_name]
                if isinstance(val, (list, tuple, np.ndarray)):
                    # Jika berupa list, ambil elemen pertama atau rata-rata
                    val = np.nanmean(val) if len(val) > 0 else 0.0
                
                # Cek secara spesifik nilai NaN
                if np.isnan(float(val)): return 0.0
                return float(val)
            except Exception:
                return 0.0

        return {
            "faithfulness": round(get_score("faithfulness"), 4),
            "answer_relevancy": round(get_score("answer_relevancy"), 4),
            "context_precision": round(get_score("context_precision"), 4),
        }

    except Exception as e:
        import traceback
        logger.error("=== ERROR RAGAS ===")
        logger.error(traceback.format_exc())
        return None


# ============================================================
# Evaluation Core
# ============================================================

def _evaluate_single_attempt(
    pipeline: AcademicRAG,
    question: Dict[str, Any],
    pipeline_name: str,
    run_number: int,
    k: int = 5,
    enable_ragas: bool = False,
    ragas_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Single attempt to evaluate a question (tanpa retry)."""
    q_text = question["question"]
    q_id = question["id"]
    relevant_chunks = question.get("relevant_chunks", [])
    ground_truth = question.get("ground_truth_answer", "")

    try:
        start_time = time.time()
        result = pipeline.query(question=q_text, include_metrics=True)
        total_time = time.time() - start_time

        # Extract data from result
        metadata = result.get("metadata", {})
        sources = result.get("sources", [])
        answer = result.get("answer", "")

        # Average relevance score (dari ChromaDB atau Reranker)
        # Gunakan 'is not None' karena skor 0.0 adalah nilai valid tapi dianggap Falsy di Python
        scores = [s.get("score", 0) for s in sources if s.get("score") is not None]
        avg_score = sum(scores) / len(scores) if scores else 0

        # --- Metrik Retrieval: MRR, P@K, R@K ---
        retrieved_ids = extract_retrieved_chunk_ids(sources)
        retrieval_metrics = compute_retrieval_metrics(retrieved_ids, relevant_chunks, k)

        # --- Konteks untuk RAGAS ---
        contexts = [s.get("content", s.get("text", "")) for s in sources]

        # --- Cek apakah answer sebenarnya pesan error dari LLM ---
        # Gemini 503/overloaded: _generate_gemini() catch exception dan
        # return {success: False, answer: pesan_error} tanpa raise.
        # Dari sisi pipeline.query(), ini terlihat sukses (no exception).
        llm_error_patterns = [
            "server AI sedang sibuk",
            "kuota layanan AI",
            "tidak dapat terhubung",
            "terjadi kesalahan saat memproses",
        ]
        is_llm_error = any(p in answer for p in llm_error_patterns)

        if is_llm_error:
            logger.warning(f"  [{pipeline_name}] LLM error terdeteksi dalam answer: {answer[:80]}...")

        # --- RAGAS Metrics (opsional) - SKIP jika answer error ---
        ragas_result = None
        if enable_ragas and ragas_config and not is_llm_error:
            ragas_result = compute_ragas_metrics(
                question=q_text,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                ragas_config=ragas_config,
            )
        elif is_llm_error and enable_ragas:
            logger.info(f"  [{pipeline_name}] RAGAS dilewati karena LLM error")

        return {
            "question_id": q_id,
            "pipeline": pipeline_name,
            "run": run_number,
            "success": not is_llm_error,  # False jika answer adalah pesan error LLM
            "answer": answer,
            "answer_length": len(answer),
            "confidence": result.get("confidence", 0),
            "metrics": {
                # Metrik operasional
                "retrieval_time": metadata.get("retrieval_time", 0),
                "generation_time": metadata.get("generation_time", 0),
                "total_time": metadata.get("total_time", total_time),
                "documents_retrieved": metadata.get("documents_retrieved", len(sources)),
                "context_length": metadata.get("context_length", 0),
                "avg_relevance_score": avg_score,
                # Metrik Retrieval
                "mrr": retrieval_metrics["mrr"],
                "precision_at_k": retrieval_metrics["precision_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
            },
            "ragas": ragas_result,  # None jika tidak diaktifkan atau LLM error
            "sources_count": len(sources),
            "sources": sources[:3],
            "retrieved_chunk_ids": retrieved_ids,
        }

    except Exception as e:
        logger.error(f"  [{pipeline_name}] Error: {e}")
        return {
            "question_id": q_id,
            "pipeline": pipeline_name,
            "run": run_number,
            "success": False,
            "error": str(e),
            "metrics": {
                "retrieval_time": 0,
                "generation_time": 0,
                "total_time": 0,
                "documents_retrieved": 0,
                "context_length": 0,
                "avg_relevance_score": 0,
                "mrr": 0,
                "precision_at_k": 0,
                "recall_at_k": 0,
            },
            "ragas": None,
        }


def evaluate_single_question(
    pipeline: AcademicRAG,
    question: Dict[str, Any],
    pipeline_name: str,
    run_number: int,
    k: int = 5,
    enable_ragas: bool = False,
    ragas_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single question with automatic retry for 503/overloaded errors.

    Menangani error dari Gemini API:
    - HTTP 503 Service Unavailable
    - HTTP 429 Too Many Requests / Rate Limit
    - RESOURCE_EXHAUSTED (quota/high demand)
    - Server overloaded / temporarily unavailable

    Menggunakan exponential backoff: 15s -> 30s -> 60s -> 120s -> 120s
    """
    q_text = question["question"]
    q_id = question["id"]

    logger.info(f"  [{pipeline_name}] Run {run_number} - {q_id}: {q_text[:60]}...")

    for attempt in range(1, MAX_RETRIES + 1):
        result = _evaluate_single_attempt(
            pipeline, question, pipeline_name, run_number,
            k=k, enable_ragas=enable_ragas, ragas_config=ragas_config,
        )

        # Sukses — langsung return
        if result.get("success") and not is_retryable_result(result):
            if attempt > 1:
                logger.info(f"  [{pipeline_name}] ✓ Berhasil setelah {attempt} percobaan")
            return result

        # Cek apakah error-nya bisa di-retry
        error_msg = result.get("error", "") or result.get("answer", "")
        if not is_retryable_error(error_msg) and not is_retryable_result(result):
            # Error bukan tipe yang bisa di-retry, langsung return
            logger.warning(f"  [{pipeline_name}] Error tidak bisa di-retry: {error_msg[:100]}")
            return result

        # Error bisa di-retry — hitung waktu tunggu
        if attempt < MAX_RETRIES:
            wait_time = min(INITIAL_WAIT * (BACKOFF_FACTOR ** (attempt - 1)), MAX_WAIT)
            logger.warning(
                f"  [{pipeline_name}] ⚠ Error 503/overloaded terdeteksi "
                f"(percobaan {attempt}/{MAX_RETRIES}). "
                f"Menunggu {wait_time:.0f} detik sebelum retry..."
            )
            logger.warning(f"  [{pipeline_name}]   Detail: {error_msg[:150]}")
            time.sleep(wait_time)
        else:
            # Sudah habis semua retry
            logger.error(
                f"  [{pipeline_name}] ✗ Gagal setelah {MAX_RETRIES} percobaan. "
                f"Error terakhir: {error_msg[:150]}"
            )
            # Tandai bahwa ini gagal setelah retry
            result["retries_exhausted"] = True
            result["total_attempts"] = MAX_RETRIES
            return result

    return result  # Fallback (seharusnya tidak tercapai)


# ============================================================
# Statistical Analysis
# ============================================================

def statistical_analysis(
    baseline_scores: List[float],
    advanced_scores: List[float],
    metric_name: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Analisis statistik:
    1. Statistik deskriptif (Mean, Std)
    2. Uji normalitas (Shapiro-Wilk)
    3. Uji hipotesis (Paired t-Test atau Wilcoxon)

    Args:
        baseline_scores: Skor dari pipeline baseline (per pertanyaan)
        advanced_scores: Skor dari pipeline advanced (per pertanyaan)
        metric_name: Nama metrik yang diuji
        alpha: Taraf signifikansi (default 0.05)

    Returns:
        Dict berisi hasil analisis lengkap
    """
    from scipy import stats

    baseline_arr = np.array(baseline_scores)
    advanced_arr = np.array(advanced_scores)

    result = {
        "metric": metric_name,
        "n_samples": len(baseline_scores),
        "descriptive": {
            "baseline": {
                "mean": round(float(np.mean(baseline_arr)), 4),
                "std": round(float(np.std(baseline_arr, ddof=1)), 4),
                "min": round(float(np.min(baseline_arr)), 4),
                "max": round(float(np.max(baseline_arr)), 4),
            },
            "advanced": {
                "mean": round(float(np.mean(advanced_arr)), 4),
                "std": round(float(np.std(advanced_arr, ddof=1)), 4),
                "min": round(float(np.min(advanced_arr)), 4),
                "max": round(float(np.max(advanced_arr)), 4),
            },
        },
    }

    # Uji Normalitas (Shapiro-Wilk) - butuh minimal 3 sampel
    if len(baseline_scores) >= 3:
        sw_baseline_stat, sw_baseline_p = stats.shapiro(baseline_arr)
        sw_advanced_stat, sw_advanced_p = stats.shapiro(advanced_arr)

        baseline_normal = sw_baseline_p > alpha
        advanced_normal = sw_advanced_p > alpha
        both_normal = baseline_normal and advanced_normal

        result["normality"] = {
            "test": "Shapiro-Wilk",
            "alpha": alpha,
            "baseline": {
                "statistic": round(float(sw_baseline_stat), 4),
                "p_value": round(float(sw_baseline_p), 4),
                "is_normal": baseline_normal,
            },
            "advanced": {
                "statistic": round(float(sw_advanced_stat), 4),
                "p_value": round(float(sw_advanced_p), 4),
                "is_normal": advanced_normal,
            },
            "both_normal": both_normal,
        }

        # Uji Hipotesis
        try:
            if both_normal:
                # Paired t-Test (parametrik)
                t_stat, p_value = stats.ttest_rel(baseline_arr, advanced_arr)
                result["hypothesis_test"] = {
                    "test": "Paired Sample t-Test",
                    "reason": "Data berdistribusi normal",
                    "statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_value), 6),
                    "alpha": alpha,
                    "significant": float(p_value) < alpha,
                    "conclusion": (
                        f"H0 DITOLAK (p={p_value:.6f} < α={alpha}): "
                        "Terdapat perbedaan signifikan antara Baseline dan Advanced."
                        if float(p_value) < alpha else
                        f"H0 DITERIMA (p={p_value:.6f} >= α={alpha}): "
                        "Tidak terdapat perbedaan signifikan."
                    ),
                }
            else:
                # Wilcoxon Signed-Rank Test (non-parametrik)
                w_stat, p_value = stats.wilcoxon(baseline_arr, advanced_arr)
                result["hypothesis_test"] = {
                    "test": "Wilcoxon Signed-Rank Test",
                    "reason": "Data tidak berdistribusi normal",
                    "statistic": round(float(w_stat), 4),
                    "p_value": round(float(p_value), 6),
                    "alpha": alpha,
                    "significant": float(p_value) < alpha,
                    "conclusion": (
                        f"H0 DITOLAK (p={p_value:.6f} < α={alpha}): "
                        "Terdapat perbedaan signifikan antara Baseline dan Advanced."
                        if float(p_value) < alpha else
                        f"H0 DITERIMA (p={p_value:.6f} >= α={alpha}): "
                        "Tidak terdapat perbedaan signifikan."
                    ),
                }
        except Exception as e:
            result["hypothesis_test"] = {
                "error": str(e),
                "note": "Uji hipotesis gagal (mungkin data identik atau terlalu sedikit)."
            }
    else:
        result["normality"] = {"error": "Sampel terlalu sedikit (< 3) untuk Shapiro-Wilk."}
        result["hypothesis_test"] = {"error": "Tidak bisa diuji — sampel terlalu sedikit."}

    return result


def run_statistical_tests(
    all_results: List[Dict],
    pipelines_config: Dict,
) -> List[Dict[str, Any]]:
    """
    Jalankan uji statistik untuk setiap metrik utama.
    Membandingkan pipeline 'baseline' vs 'hybrid'.
    """
    pipeline_names = list(pipelines_config.keys())
    if len(pipeline_names) < 2:
        logger.warning("Perlu minimal 2 pipeline untuk uji statistik.")
        return []

    baseline_name = pipeline_names[0]  # 'baseline'
    advanced_name = pipeline_names[1]  # 'hybrid'

    # Kumpulkan skor per pertanyaan (rata-rata jika multi-run)
    metrics_to_test = ["mrr", "precision_at_k", "recall_at_k"]

    # Jika RAGAS tersedia, tambahkan
    sample = next((r for r in all_results if r.get("ragas")), None)
    if sample:
        metrics_to_test.extend(["faithfulness", "answer_relevancy", "context_precision"])

    stat_results = []

    for metric_key in metrics_to_test:
        # Ambil skor per pertanyaan (rata-rata antar run)
        baseline_per_q = _get_per_question_scores(all_results, baseline_name, metric_key)
        advanced_per_q = _get_per_question_scores(all_results, advanced_name, metric_key)

        if len(baseline_per_q) == len(advanced_per_q) and len(baseline_per_q) > 0:
            result = statistical_analysis(baseline_per_q, advanced_per_q, metric_key)
            stat_results.append(result)
        else:
            logger.warning(f"Ukuran sampel tidak cocok untuk {metric_key}: "
                           f"baseline={len(baseline_per_q)}, advanced={len(advanced_per_q)}")

    return stat_results


def _get_per_question_scores(
    all_results: List[Dict],
    pipeline_name: str,
    metric_key: str
) -> List[float]:
    """Ambil skor rata-rata per pertanyaan untuk pipeline tertentu."""
    from collections import defaultdict

    # Kelompokkan per question_id
    scores_by_q = defaultdict(list)
    for r in all_results:
        if r["pipeline"] == pipeline_name and r.get("success", False):
            # Cek di metrics atau ragas
            score = r.get("metrics", {}).get(metric_key)
            if score is None and r.get("ragas"):
                score = r["ragas"].get(metric_key)
            if score is not None:
                scores_by_q[r["question_id"]].append(score)

    # Rata-ratakan per pertanyaan
    per_q_scores = []
    for q_id in sorted(scores_by_q.keys()):
        vals = scores_by_q[q_id]
        per_q_scores.append(float(np.mean(vals)))

    return per_q_scores


# ============================================================
# Main Evaluation Runner
# ============================================================

def run_evaluation(config: Dict[str, Any], questions: List[Dict]) -> Dict[str, Any]:
    """Run full evaluation across both pipelines."""
    eval_config = config["evaluation"]
    paths_config = config.get("paths", {})
    num_runs = eval_config.get("num_runs", 1)
    pipelines_config = eval_config["pipelines"]
    k = eval_config.get("k", 5)  # Top-K untuk Precision@K dan Recall@K

    # RAGAS config
    ragas_config = eval_config.get("ragas", {})
    enable_ragas = ragas_config.get("enabled", False)

    results = {
        "metadata": {
            "experiment_name": eval_config["name"],
            "description": eval_config["description"],
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "num_runs": num_runs,
            "k": k,
            "pipelines": list(pipelines_config.keys()),
            "ragas_enabled": enable_ragas,
        },
        "results": [],
        "summary": {},
        "statistical_tests": [],
    }

    # Initialize pipelines
    logger.info("=" * 60)
    logger.info("Inisialisasi Pipeline")
    logger.info("=" * 60)

    pipelines = {}
    for name, p_config in pipelines_config.items():
        logger.info(f"Inisialisasi pipeline: {name}")
        try:
            pipelines[name] = create_rag_pipeline(
                pipeline_type=p_config["pipeline_type"],
                config_overrides=p_config,
                paths_config=paths_config,
            )
            logger.info(f"  ✓ Pipeline '{name}' berhasil diinisialisasi")
        except Exception as e:
            logger.error(f"  ✗ Gagal menginisialisasi '{name}': {e}")
            return results

    # Run evaluation
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Memulai Evaluasi ({len(questions)} pertanyaan × {num_runs} run × {len(pipelines)} pipeline)")
    logger.info(f"K = {k} | RAGAS = {'ON' if enable_ragas else 'OFF'}")
    logger.info("=" * 60)

    all_results = []

    for run_num in range(1, num_runs + 1):
        logger.info(f"\n--- Run {run_num}/{num_runs} ---")

        for q_idx, question in enumerate(questions, 1):
            logger.info(f"\nPertanyaan {q_idx}/{len(questions)}: {question['id']}")

            for pipeline_name, pipeline in pipelines.items():
                result = evaluate_single_question(
                    pipeline, question, pipeline_name, run_num,
                    k=k,
                    enable_ragas=enable_ragas,
                    ragas_config=ragas_config,
                )
                all_results.append(result)
                
                # Kasih jeda antar request hanya jika menggunakan Gemini (cloud API)
                # Ollama berjalan lokal sehingga tidak perlu rate limit protection
                llm_provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
                if llm_provider == "gemini":
                    if result.get("retries_exhausted"):
                        sleep_time = 30  # Jeda lebih lama setelah retry habis
                        logger.warning(f"  [Sleep] Menunggu {sleep_time} detik (setelah retry habis)...\n")
                    else:
                        sleep_time = 5
                        logger.info(f"  [Sleep] Menunggu {sleep_time} detik (Gemini Rate Limit Protection)...\n")
                    time.sleep(sleep_time)

    results["results"] = all_results

    # Generate summary
    results["summary"] = generate_summary(all_results, pipelines_config)

    # Statistical tests
    logger.info("\n" + "=" * 60)
    logger.info("Menjalankan Uji Statistik")
    logger.info("=" * 60)
    results["statistical_tests"] = run_statistical_tests(all_results, pipelines_config)

    return results


# ============================================================
# Summary & Output
# ============================================================

def generate_summary(
    all_results: List[Dict], pipelines_config: Dict
) -> Dict[str, Any]:
    """Generate summary statistics from evaluation results."""
    summary = {}

    for pipeline_name in pipelines_config:
        pipeline_results = [r for r in all_results if r["pipeline"] == pipeline_name]
        successful = [r for r in pipeline_results if r.get("success", False)]

        if not successful:
            summary[pipeline_name] = {"error": "No successful results"}
            continue

        # Aggregate metrics
        metrics = {
            "total_questions": len(pipeline_results),
            "successful": len(successful),
            "success_rate": len(successful) / len(pipeline_results),
            # Metrik operasional
            "avg_retrieval_time": np.mean([r["metrics"]["retrieval_time"] for r in successful]),
            "avg_generation_time": np.mean([r["metrics"]["generation_time"] for r in successful]),
            "avg_total_time": np.mean([r["metrics"]["total_time"] for r in successful]),
            "avg_documents_retrieved": np.mean([r["metrics"]["documents_retrieved"] for r in successful]),
            "avg_context_length": np.mean([r["metrics"]["context_length"] for r in successful]),
            "avg_relevance_score": np.mean([r["metrics"]["avg_relevance_score"] for r in successful]),
            "avg_confidence": np.mean([r.get("confidence", 0) for r in successful]),
            "avg_answer_length": np.mean([r.get("answer_length", 0) for r in successful]),
            # Metrik Retrieval
            "avg_mrr": np.mean([r["metrics"]["mrr"] for r in successful]),
            "avg_precision_at_k": np.mean([r["metrics"]["precision_at_k"] for r in successful]),
            "avg_recall_at_k": np.mean([r["metrics"]["recall_at_k"] for r in successful]),
        }

        # RAGAS metrics (jika ada)
        ragas_results = [r["ragas"] for r in successful if r.get("ragas")]
        if ragas_results:
            metrics["avg_faithfulness"] = np.mean([r["faithfulness"] for r in ragas_results])
            metrics["avg_answer_relevancy"] = np.mean([r["answer_relevancy"] for r in ragas_results])
            metrics["avg_context_precision"] = np.mean([r["context_precision"] for r in ragas_results])

        # Round all float values
        for k, v in metrics.items():
            if isinstance(v, (float, np.floating)):
                metrics[k] = round(float(v), 4)

        summary[pipeline_name] = metrics

    return summary


def save_results(results: Dict, output_dir: str = "results"):
    """Save evaluation results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"\nHasil disimpan ke: {filepath}")
    return filepath


def print_summary(results: Dict):
    """Print evaluation summary to console."""
    summary = results.get("summary", {})

    print("\n" + "=" * 70)
    print("RINGKASAN EVALUASI")
    print("=" * 70)
    print(f"Eksperimen: {results['metadata']['experiment_name']}")
    print(f"Waktu: {results['metadata']['timestamp']}")
    print(f"Pertanyaan: {results['metadata']['num_questions']}")
    print(f"Pengulangan: {results['metadata']['num_runs']}")
    print(f"K: {results['metadata'].get('k', 5)}")
    print(f"RAGAS: {'Aktif' if results['metadata'].get('ragas_enabled') else 'Nonaktif'}")
    print()

    # Header
    header = f"{'Metrik':<30} "
    for pipeline in summary:
        header += f"{'│ ' + pipeline:<20}"
    print(header)
    print("─" * len(header))

    # Metric rows - grouped
    metric_groups = {
        "── Metrik Retrieval ──": {
            "avg_mrr": "MRR",
            "avg_precision_at_k": "Precision@K",
            "avg_recall_at_k": "Recall@K",
        },
        "── Metrik RAGAS ──": {
            "avg_faithfulness": "Faithfulness",
            "avg_answer_relevancy": "Answer Relevancy",
            "avg_context_precision": "Context Precision",
        },
        "── Metrik Operasional ──": {
            "success_rate": "Success Rate",
            "avg_retrieval_time": "Avg Retrieval Time (s)",
            "avg_generation_time": "Avg Generation Time (s)",
            "avg_total_time": "Avg Total Time (s)",
            "avg_relevance_score": "Avg Relevance Score",
            "avg_confidence": "Avg Confidence",
            "avg_answer_length": "Avg Answer Length",
        },
    }

    for group_name, metrics in metric_groups.items():
        # Check if any metric in this group exists
        has_data = any(
            metric_key in summary.get(list(summary.keys())[0], {})
            for metric_key in metrics
        )
        if not has_data:
            continue

        print(f"\n{group_name}")
        for metric_key, label in metrics.items():
            row = f"{label:<30} "
            for pipeline_name in summary:
                value = summary[pipeline_name].get(metric_key, "N/A")
                if isinstance(value, float):
                    if metric_key == "success_rate":
                        row += f"│ {value:.1%}            "
                    elif "time" in metric_key:
                        row += f"│ {value:.4f}           "
                    else:
                        row += f"│ {value:.4f}           "
                else:
                    row += f"│ {str(value):<18}"
            print(row)

    print("\n" + "=" * 70)

    # Print statistical test results
    stat_tests = results.get("statistical_tests", [])
    if stat_tests:
        print("\n" + "=" * 70)
        print("UJI STATISTIK")
        print("=" * 70)

        for test in stat_tests:
            metric = test.get("metric", "?")
            desc = test.get("descriptive", {})
            norm = test.get("normality", {})
            hyp = test.get("hypothesis_test", {})

            print(f"\n--- {metric.upper()} ---")

            # Deskriptif
            b = desc.get("baseline", {})
            a = desc.get("advanced", {})
            print(f"  Baseline:  Mean={b.get('mean', '?')}, Std={b.get('std', '?')}")
            print(f"  Advanced:  Mean={a.get('mean', '?')}, Std={a.get('std', '?')}")

            # Normalitas
            if "error" not in norm:
                bn = norm.get("baseline", {})
                an = norm.get("advanced", {})
                print(f"  Shapiro-Wilk: Baseline p={bn.get('p_value', '?')} "
                      f"({'Normal' if bn.get('is_normal') else 'Tidak Normal'})")
                print(f"                Advanced p={an.get('p_value', '?')} "
                      f"({'Normal' if an.get('is_normal') else 'Tidak Normal'})")

            # Hipotesis
            if "error" not in hyp:
                print(f"  Uji: {hyp.get('test', '?')}")
                print(f"  p-value: {hyp.get('p_value', '?')}")
                sig = "✓ SIGNIFIKAN" if hyp.get("significant") else "✗ Tidak Signifikan"
                print(f"  Hasil: {sig}")
                print(f"  Kesimpulan: {hyp.get('conclusion', '')}")

        print("\n" + "=" * 70)


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi Perbandingan RAG Baseline vs Hybrid"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path ke file konfigurasi"
    )
    parser.add_argument(
        "--questions", default="questions.json", help="Path ke dataset pertanyaan"
    )
    parser.add_argument(
        "--output", default="results", help="Direktori output hasil"
    )
    parser.add_argument(
        "--runs", type=int, default=None, help="Override jumlah pengulangan"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Batasi jumlah pertanyaan (untuk testing)"
    )
    parser.add_argument(
        "--ragas", action="store_true", help="Aktifkan metrik RAGAS (butuh API key)"
    )
    args = parser.parse_args()

    # Change to script directory
    os.chdir(SCRIPT_DIR)

    # Load config & questions
    logger.info("Memuat konfigurasi...")
    config = load_config(args.config)
    questions = load_questions(args.questions)

    # Apply overrides
    if args.runs:
        config["evaluation"]["num_runs"] = args.runs
    if args.limit:
        questions = questions[: args.limit]
        logger.info(f"Dibatasi ke {args.limit} pertanyaan")
    if args.ragas:
        config["evaluation"]["ragas"]["enabled"] = True

    # Run evaluation
    results = run_evaluation(config, questions)

    # Save & print
    filepath = save_results(results, args.output)
    print_summary(results)

    print(f"\nUntuk generate laporan visual, jalankan:")
    print(f"  python generate_report.py --input {filepath}")


if __name__ == "__main__":
    main()
