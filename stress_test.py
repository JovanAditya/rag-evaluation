"""
Stress Test: Run all questions from questions.json through the RAG pipeline
and save raw answers for manual audit.
"""
import sys
import json
import time
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/jovan/umb/rag-deploy")
RAG_MODEL_DIR = PROJECT_ROOT / "rag-model"
sys.path.insert(0, str(RAG_MODEL_DIR))

# Load .env
try:
    from dotenv import load_dotenv
    env_path = RAG_MODEL_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from rag_model.core.pipeline import AcademicRAG

def main():
    # Load questions
    with open(SCRIPT_DIR / "questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    
    print(f"Total questions: {len(questions)}")
    print("Initializing RAG pipeline...")
    
    rag = AcademicRAG()
    
    results = []
    failures = []
    
    for i, q in enumerate(questions):
        qid = q["id"]
        qtext = q["question"]
        gt = q.get("ground_truth_answer", "")
        cat = q.get("category", "")
        
        print(f"\n[{i+1}/{len(questions)}] {qid}: {qtext[:80]}...")
        
        try:
            start = time.time()
            result = rag.query(qtext, pipeline_type="advanced", max_results=5)
            elapsed = time.time() - start
            
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            top_source_id = sources[0]["id"] if sources else "N/A"
            
            entry = {
                "id": qid,
                "category": cat,
                "question": qtext,
                "ground_truth": gt,
                "answer": answer,
                "top_source": top_source_id,
                "sources_count": len(sources),
                "time": round(elapsed, 2),
            }
            results.append(entry)
            
            # Quick check for potential failures
            answer_lower = answer.lower()
            is_fail = any(p in answer_lower for p in [
                "maaf", "tidak tersedia", "tidak ditemukan", 
                "tidak disebutkan", "belum tersedia",
                "tidak dapat ditemukan", "tidak ada informasi"
            ])
            
            if is_fail:
                failures.append(entry)
                print(f"  ❌ POTENTIAL FAIL: {answer[:120]}...")
            else:
                print(f"  ✅ OK ({elapsed:.1f}s): {answer[:120]}...")
                
        except Exception as e:
            entry = {
                "id": qid,
                "category": cat,
                "question": qtext,
                "ground_truth": gt,
                "answer": f"ERROR: {str(e)}",
                "top_source": "N/A",
                "sources_count": 0,
                "time": 0,
            }
            results.append(entry)
            failures.append(entry)
            print(f"  ❌ ERROR: {e}")
    
    # Save results
    output_path = SCRIPT_DIR / "stress_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(results),
            "failures": len(failures),
            "success_rate": f"{(len(results)-len(failures))/len(results)*100:.1f}%",
            "results": results,
            "failure_ids": [f["id"] for f in failures],
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"STRESS TEST COMPLETE")
    print(f"Total: {len(results)} | Failures: {len(failures)} | Success Rate: {(len(results)-len(failures))/len(results)*100:.1f}%")
    if failures:
        print(f"\nFailed IDs: {[f['id'] for f in failures]}")
        for f in failures:
            print(f"  - {f['id']}: {f['answer'][:100]}...")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
