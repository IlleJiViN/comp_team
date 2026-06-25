import os
import sys
import json
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from google import genai

# Configure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Configurations
ES_URL = "http://localhost:9200"
INDEX_NAME = "spotsync_chunks"
GCP_PROJECT = "spotsync-500217"
GCP_LOCATION = "us-central1"

# Define Golden Queries and their Expected target names/keywords
GOLDEN_TESTS = [
    {
        "query": "합정역 근처 맛있는 양념 돼지갈비 고기집",
        "expected_keywords": ["갈비", "양화정", "합정숯불갈비"]
    },
    {
        "query": "화양동 신나게 노래 부르고 스트레스 푸는 코인 노래방",
        "expected_keywords": ["파파노래방", "노래연습장", "노래방"]
    },
    {
        "query": "동교동 카야잼 커피번 디저트 맛집 아늑한 카페",
        "expected_keywords": ["트렌", "로티보이", "카페"]
    },
    {
        "query": "망원동 오징어 초무침 골뱅이 무침 빈대떡 파는 포장마차 감성 요리주점",
        "expected_keywords": ["나들목", "빈대떡", "포장마차"]
    },
    {
        "query": "공부하기 좋은 조용하고 분위기 편안한 홍대 스터디 카페",
        "expected_keywords": ["스터디카페", "독서실", "공간"]
    }
]

def get_vertex_embedding(text: str) -> np.ndarray:
    """Generates 768-dimensional embedding using Google Vertex AI text-embedding-004."""
    google_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    res = google_client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    if res and res.embeddings:
        return np.array(res.embeddings[0].values, dtype=np.float32)
    raise RuntimeError("Failed to generate embedding from Vertex AI")

def calculate_mrr_and_success(results, expected_keywords):
    """Calculates Reciprocal Rank and Success metrics for a search result list."""
    for rank, hit in enumerate(results):
        name = hit['_source'].get('name', '').lower()
        category = hit['_source'].get('category', '').lower()
        text_val = hit['_source'].get('text', '').lower()
        
        # Match if any of the expected keywords are in the place name or text
        matched = any(kw.lower() in name or kw.lower() in text_val for kw in expected_keywords)
        if matched:
            reciprocal_rank = 1.0 / (rank + 1)
            return reciprocal_rank, rank + 1
            
    return 0.0, None

def run_sweep():
    print("="*80)
    print("      SpotSync AI - Hybrid Search Ratio Sweep Optimizer")
    print("="*80)

    # 1. Connect to ES
    es = Elasticsearch(ES_URL, request_timeout=30)
    try:
        info = es.info()
        print(f"[INFO] Connected to Elasticsearch version {info['version']['number']}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Elasticsearch: {e}")
        return

    # 2. Pre-generate embeddings for Golden Queries to speed up the sweep
    print("\n[1/3] Generating Google text-embedding-004 vectors for golden queries...")
    query_embeddings = {}
    for test in GOLDEN_TESTS:
        q = test["query"]
        try:
            print(f"  - Embedding: '{q}'...")
            query_embeddings[q] = get_vertex_embedding(q)
        except Exception as e:
            print(f"  - [ERROR] Failed to embed '{q}': {e}. Using dummy embedding.")
            query_embeddings[q] = np.zeros(768)

    # 3. Sweep KNN weights (boost) from 0.0 to 1.0
    print("\n[2/3] Executing hybrid search sweeps across weights...")
    
    # We will sweep the knn boost weight (which represents the vector score weighting)
    # in ES, BM25 query and KNN are combined.
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_results = []

    for w in weights:
        print(f"  - Sweeping Semantic Weight: {w:.1f} (BM25 Weight: {1.0 - w:.1f})")
        mrr_sum = 0.0
        success_at_5 = 0
        success_at_10 = 0
        latencies = []
        
        for test in GOLDEN_TESTS:
            q = test["query"]
            expected = test["expected_keywords"]
            q_emb = query_embeddings[q]
            
            # Formulate hybrid query
            # We assign KNN boost = w, and BM25 text query boost = 1.0 - w
            # If w is 0.0, we execute pure BM25. If w is 1.0, we execute pure KNN.
            
            body = {}
            if w > 0.0:
                body["knn"] = {
                    "field": "embedding",
                    "query_vector": q_emb.tolist(),
                    "k": 20,
                    "num_candidates": 100,
                    "boost": w
                }
            
            # BM25 portion
            should_clauses = [
                {"match": {"text": {"query": q, "boost": (1.0 - w) * 3.0}}},
                {"match": {"name": {"query": q, "boost": (1.0 - w) * 5.0}}}
            ]
            body["query"] = {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 0
                }
            }
            body["size"] = 20
            
            t_start = time.perf_counter()
            try:
                res = es.search(index=INDEX_NAME, body=body)
                latency = (time.perf_counter() - t_start) * 1000
                latencies.append(latency)
                
                hits = res['hits']['hits']
                rr, rank = calculate_mrr_and_success(hits, expected)
                
                mrr_sum += rr
                if rank and rank <= 5: success_at_5 += 1
                if rank and rank <= 10: success_at_10 += 1
                
            except Exception as e:
                print(f"    [ERROR] Query failed for weight {w}: {e}")
                latencies.append(0)

        avg_mrr = mrr_sum / len(GOLDEN_TESTS)
        pct_s5 = (success_at_5 / len(GOLDEN_TESTS)) * 100
        pct_s10 = (success_at_10 / len(GOLDEN_TESTS)) * 100
        avg_lat = sum(latencies) / len(latencies)
        
        sweep_results.append({
            "weight": w,
            "mrr": avg_mrr,
            "success_5": pct_s5,
            "success_10": pct_s10,
            "latency": avg_lat
        })
        print(f"    => Avg MRR: {avg_mrr:.4f} | Success@5: {pct_s5:.1f}% | Latency: {avg_lat:.1f}ms")

    # 4. Generate Report
    print("\n[3/3] Generating optimization report...")
    df_results = pd.DataFrame(sweep_results)
    optimal_idx = df_results['mrr'].idxmax()
    optimal_weight = df_results.loc[optimal_idx, 'weight']
    
    print("\n" + "="*50)
    print("🏆 OPTIMAL HYBRID SEARCH CONFIGURATION")
    print(f"  - Optimal Semantic Weight (KNN): {optimal_weight:.1f}")
    print(f"  - Optimal Keyword Weight (BM25): {1.0 - optimal_weight:.1f}")
    print(f"  - Max Mean Reciprocal Rank (MRR): {df_results.loc[optimal_idx, 'mrr']:.4f}")
    print(f"  - Success@5: {df_results.loc[optimal_idx, 'success_5']:.1f}%")
    print(f"  - Avg Latency: {df_results.loc[optimal_idx, 'latency']:.1f} ms")
    print("="*50 + "\n")

    # Save to a Markdown report file
    report_content = f"""# 📈 SpotSync AI - Hybrid Search Ratio Sweep Optimization Report

This report summarizes the sweep of hybrid search weight parameters (KNN vector search vs. BM25 keyword search) evaluated against golden test queries using **Google Vertex AI `text-embedding-004` (768-dim)** and Elasticsearch.

## 📊 Evaluation Metrics Sweep Results

| Semantic Weight (KNN) | Keyword Weight (BM25) | Mean Reciprocal Rank (MRR) | Success@5 (%) | Success@10 (%) | Avg Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |\n"""
    
    for r in sweep_results:
        is_opt = " ⭐ **(Optimal)**" if r["weight"] == optimal_weight else ""
        report_content += f"| {r['weight']:.1f}{is_opt} | {1.0 - r['weight']:.1f} | {r['mrr']:.4f} | {r['success_5']:.1f}% | {r['success_10']:.1f}% | {r['latency']:.1f} ms |\n"
        
    report_content += f"""
## 🏆 Conclusion & Recommendations
- **Optimal Weight Configuration**: We highly recommend setting the Elasticsearch KNN query boost to **`{optimal_weight:.1f}`** and the BM25 query text match boost to **`{1.0 - optimal_weight:.1f}`**.
- This hybrid balanced setting achieves a Mean Reciprocal Rank (MRR) of **`{df_results.loc[optimal_idx, 'mrr']:.4f}`** and successfully retrieves the correct target places inside the **Top 5** results **`{df_results.loc[optimal_idx, 'success_5']:.1f}%`** of the time.
"""

    report_path = "data/hybrid_sweep_results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[SUCCESS] Saved detailed sweep report to {report_path}")

if __name__ == "__main__":
    import time
    run_sweep()
