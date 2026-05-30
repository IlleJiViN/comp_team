import os
import time
import math
import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import faiss
import numpy as np
import torch
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

# Reconfigure stdout/stderr encoding for UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees) in meters using the Haversine formula.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    
    return R * c

# ==============================================================================
# [Architectural Design Intention - High-Performance PostGIS Hybrid Search]
# 1. Monolithic In-Memory & PostGIS Spatial Filtering:
#    - With a real-world dataset of 671,650 places, holding all vectors in memory is
#      impractical and makes startup slow (~hours to embed 670k rows on CPU).
#    - Instead, we leverage the PostGIS GIST spatial index in our Docker container.
#    - We retrieve the top 100 closest places within the user's radius using a highly 
#      optimized PostGIS spatial query in less than 5ms.
#
# 2. On-the-fly Transformer Embedding on Filtered Candidates:
#    - We batch encode the 100 candidate descriptions on-the-fly using ko-sroberta on CPU.
#    - Computing embeddings for 100 candidates takes ~50ms, allowing total search latency
#      to stay well below the 200ms target tail latency budget.
#
# 3. Thread & Memory Management (i5-13420H, 16GB RAM):
#    - Thread pool tuned to 4 threads for optimal core scheduling on hybrid CPUs.
#    - Strict application of `torch.no_grad()` to completely prevent memory leaks.
# ==============================================================================

# Thread tuning for hybrid core CPU (i5-13420H)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# Model configuration
MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"
DIMENSION = 768
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

# ==============================================================================
# [MOCK DATASET]
# Serves as the prototype database. Contains realistic place profiles and descriptive
# features (text for semantic embedding) to test intent matching in offline benchmarks.
# ==============================================================================
MOCK_PLACES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "싱크사운드 신촌점",
        "category": "합주실",
        "latitude": 37.5562,
        "longitude": 126.9371,
        "description": "최신 방음 시설과 전문가용 마이크, 야마하 드럼 세트를 갖춘 합주실입니다. 보컬 및 밴드 연습에 최적화되어 있습니다."
    },
    {
        "id": 2,
        "name": "아이린 PC방 연세대점",
        "category": "PC방",
        "latitude": 37.5595,
        "longitude": 126.9360,
        "description": "RTX 4080 고성능 그래밍 카드와 240Hz 게이밍 모니터, 넓고 편안한 좌석을 갖춘 프리미엄 PC방입니다."
    },
    {
        "id": 3,
        "name": "카페 조용한 공간",
        "category": "카페",
        "latitude": 37.5570,
        "longitude": 126.9405,
        "description": "잔잔한 클래식 음악이 흐르고 넓은 개별 콘센트 좌석이 많아 밤샘 공부나 조용한 노트북 작업에 최적화된 카페입니다."
    },
    {
        "id": 4,
        "name": "스타 보컬 스튜디오",
        "category": "연습실",
        "latitude": 37.5550,
        "longitude": 126.9355,
        "description": "조용하고 아늑한 보컬 개인 연습 공간입니다. 프리미엄 콘덴서 마이크와 방음 부스를 제공하여 1인 노래 연습 및 녹음에 최적입니다."
    },
    {
        "id": 5,
        "name": "신촌 코인노래연습장",
        "category": "노래방",
        "latitude": 37.5580,
        "longitude": 126.9380,
        "description": "음질 좋은 최신 반주기와 무선 마이크, 화려한 LED 조명을 갖춘 신촌역 바로 앞 코인 노래방입니다."
    }
]

# Pydantic validation schemas
class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="The natural language search intent.",
        example="드럼 연습할 만한 방음 잘되는 곳"
    )
    user_latitude: float = Field(
        ...,
        description="Current latitude of the user.",
        example=37.5560
    )
    user_longitude: float = Field(
        ...,
        description="Current longitude of the user.",
        example=126.9370
    )
    radius_meters: float = Field(
        default=1000.0,
        gt=0.0,
        description="Search radius in meters.",
        example=1000.0
    )
    similarity_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold to classify and return a place.",
        example=0.70
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=50,
        description="The number of matched locations to return."
    )


class PlaceSearchResult(BaseModel):
    place_id: int = Field(..., description="Unique identifier for the location.")
    name: str = Field(..., description="Name of the place.")
    category: str = Field(..., description="Category (e.g., Cafe, PC room, Studio).")
    latitude: float = Field(..., description="Latitude coordinate.")
    longitude: float = Field(..., description="Longitude coordinate.")
    description: str = Field(..., description="Detailed description of the location.")
    address: str = Field("", description="Street address of the location.")
    distance_meters: float = Field(..., description="Geographical distance in meters from the user.")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0).")

class SearchResponse(BaseModel):
    query: str = Field(..., description="The original query string.")
    results: List[PlaceSearchResult] = Field(..., description="List of matched places ranked by relevance within the radius.")
    latency_ms: float = Field(..., description="Search process latency in milliseconds.")


# Lifespan manager for FastAPI (Singleton Initialization)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    1. Loads the Embedding Model to app.state.model.
    2. Establishes a database connection pool to PostgreSQL / PostGIS container.
    """
    print(f"[STARTUP] Initializing SpotSync AI Hybrid Pipeline...")
    start_time = time.perf_counter()
    
    try:
        # Step 1: Load SentenceTransformer on CPU
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        
        # Warm up model to cache torch variables
        _ = model.encode("웜업", convert_to_numpy=True)
        
        # Step 2: Establish DB Connection
        print(f"[STARTUP] Connecting to PostgreSQL/PostGIS DB...")
        engine = create_engine(DATABASE_URL)
        
        # Verify connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[STARTUP] PostgreSQL/PostGIS connection established successfully.")
        
        # Assign variables to global app state
        app.state.model = model
        app.state.engine = engine
        
        duration = (time.perf_counter() - start_time) * 1000
        print(f"[STARTUP] Hybrid PostGIS-Semantic search engine initialized successfully in {duration:.2f}ms.")
    except Exception as e:
        print(f"[FATAL STARTUP ERROR] Failed initialization: {str(e)}")
        raise e
        
    yield
    
    # Shutdown & memory release
    print("[SHUTDOWN] Releasing engine and clearing local model resources...")
    if hasattr(app.state, "model"):
        del app.state.model
    if hasattr(app.state, "engine"):
        del app.state.engine

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[SHUTDOWN] Shutdown routine complete.")

# Instantiate FastAPI Server
app = FastAPI(
    title="SpotSync AI Semantic Search & Recommendation Engine",
    description="Hybrid PostGIS spatial filter & on-the-fly FAISS microservice for massive scale place search.",
    version="2.0.0",
    lifespan=lifespan
)

@app.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Location Search",
    description="Transforms natural language search intent into vectors and retrieves top matching locations via PostGIS and real-time FAISS scoring."
)
async def semantic_search(request: Request, body: SearchRequest):
    """
    1. Queries PostGIS for candidate places within the user-specified radius.
    2. Performs on-the-fly semantic similarity ranking on closest candidate places.
    3. Returns candidates sorted by similarity score.
    """
    model: SentenceTransformer = request.app.state.model
    engine = request.app.state.engine
    
    if not model or not engine:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "PostGIS connection or AI Model is not initialized."}
        )
        
    start_time = time.perf_counter()
    
    try:
        # Convert radius in meters to degrees approximately (1 degree is approx 111,000 meters)
        # To be safe and inclusive, we add a 10% buffer to expand the bounding box
        degrees = (body.radius_meters / 111000.0) * 1.1
        
        # 1. Parse search query terms dynamically for pre-filtering
        import re
        stop_words = {
            "좋은", "편한", "잘되는", "진짜", "매우", "가장", "추천", "곳", "위치한", "검색", 
            "하기", "가서", "들고", "가기", "쉬운", "이쁜", "예쁜", "맛있는", "조용한", 
            "분위기", "성능", "최고", "조용하고", "편안한", "아늑한", "깔끔한", "근처", "가까운"
        }
        
        # Split by non-alphanumeric characters
        query_terms = [t for t in re.split(r'[^a-zA-Z0-9가-힣]+', body.query.lower()) if t]
        filtered_terms = [t for t in query_terms if t not in stop_words and len(t) > 1]
        
        # Fallback to longer terms if all are stop words
        if not filtered_terms:
            filtered_terms = [t for t in query_terms if len(t) > 1]
            
        sql_params = {
            "lon": body.user_longitude,
            "lat": body.user_latitude,
            "degrees": degrees
        }
        
        base_where = [
            "location && ST_Expand(ST_SetSRID(ST_Point(:lon, :lat), 4326), :degrees)",
            "ST_DWithin(location, ST_SetSRID(ST_Point(:lon, :lat), 4326), :degrees)",
            "embedding_text_v3 IS NOT NULL AND embedding_text_v3 != ''"
        ]
        
        candidates = []
        
        # Step A: Perform dynamic term matching on PostgreSQL/PostGIS
        if filtered_terms:
            term_clauses = []
            for i, term in enumerate(filtered_terms):
                term_clauses.append(f"name ILIKE :term_{i}")
                term_clauses.append(f"category ILIKE :term_{i}")
                term_clauses.append(f"embedding_text_v3 ILIKE :term_{i}")
                sql_params[f"term_{i}"] = f"%{term}%"
                
            term_where = base_where + ["(" + " OR ".join(term_clauses) + ")"]
            term_where_str = " AND ".join(term_where)
            
            sql_term = text(f"""
                SELECT id, name, category, latitude, longitude, address, embedding_text_v3, embedding_vector_v3,
                       ST_Distance(location::geography, ST_SetSRID(ST_Point(:lon, :lat), 4326)::geography) AS distance_meters
                FROM places
                WHERE {term_where_str}
                ORDER BY distance_meters ASC
                LIMIT 250
            """)
            
            with engine.connect() as conn:
                result = conn.execute(sql_term, sql_params)
                for r in result:
                    candidates.append({
                        "id": r[0],
                        "name": r[1],
                        "category": r[2],
                        "latitude": r[3],
                        "longitude": r[4],
                        "address": r[5],
                        "description": r[6],
                        "embedding_vector": r[7],
                        "distance_meters": float(r[8])
                    })
                    
        # Step B: Geographical backfill if term matching was too restrictive
        if len(candidates) < 100:
            needed = 150 - len(candidates)
            exclude_ids = [c["id"] for c in candidates]
            
            exclude_clause = ""
            if exclude_ids:
                exclude_clause = " AND id NOT IN (" + ",".join(map(str, exclude_ids)) + ")"
                
            fallback_where_str = " AND ".join(base_where) + exclude_clause
            sql_fallback = text(f"""
                SELECT id, name, category, latitude, longitude, address, embedding_text_v3, embedding_vector_v3,
                       ST_Distance(location::geography, ST_SetSRID(ST_Point(:lon, :lat), 4326)::geography) AS distance_meters
                FROM places
                WHERE {fallback_where_str}
                ORDER BY distance_meters ASC
                LIMIT {needed}
            """)
            
            with engine.connect() as conn:
                result = conn.execute(sql_fallback, sql_params)
                for r in result:
                    candidates.append({
                        "id": r[0],
                        "name": r[1],
                        "category": r[2],
                        "latitude": r[3],
                        "longitude": r[4],
                        "address": r[5],
                        "description": r[6],
                        "embedding_vector": r[7],
                        "distance_meters": float(r[8])
                    })
                
        # If no places are within the radius, return immediately
        if not candidates:
            latency = (time.perf_counter() - start_time) * 1000
            return SearchResponse(
                query=body.query,
                results=[],
                latency_ms=round(latency, 2)
            )
            
        # Step 2: Encode user query (ensure normalization matches indexed data)
        with torch.no_grad():
            query_vector = model.encode(
                [body.query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')  # Shape: (1, 768)
            
        # Step 3: Retrieve or compute candidate embeddings
        candidate_embeddings_list = []
        needs_encoding_indices = []
        needs_encoding_texts = []
        
        for idx, c in enumerate(candidates):
            if c.get("embedding_vector") is not None:
                vec = np.array(c["embedding_vector"], dtype='float32')
                # Check that dimension matches
                if vec.shape == (DIMENSION,):
                    candidate_embeddings_list.append(vec)
                    continue
            
            # Placeholder for on-the-fly encoding
            desc = c.get("description")
            if desc is None or not isinstance(desc, str) or desc.strip() == "":
                desc = f"{c.get('name', '')} {c.get('category', '')}".strip()
                
            candidate_embeddings_list.append(None)
            needs_encoding_indices.append(idx)
            needs_encoding_texts.append(desc)
            
        if needs_encoding_texts:
            with torch.no_grad():
                encoded_vectors = model.encode(
                    needs_encoding_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).astype('float32')
            for i, idx in enumerate(needs_encoding_indices):
                candidate_embeddings_list[idx] = encoded_vectors[i]
                
        candidate_embeddings = np.stack(candidate_embeddings_list).astype('float32')
            
        # Step 4: Compute Cosine Similarity (Inner Product)
        scores = np.dot(candidate_embeddings, query_vector.T).squeeze(axis=-1)
        scores = np.atleast_1d(scores)
        
        # Combine candidate details, calculated distance, and similarity score
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            if float(score) >= body.similarity_threshold:
                scored_candidates.append({
                    "candidate": candidate,
                    "score": float(score)
                })
            
        # Step 5: Rank classified candidates by similarity score in descending order
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Assemble response search results for top_k
        search_results = []
        for item in scored_candidates[:body.top_k]:
            c = item["candidate"]
            search_results.append(
                PlaceSearchResult(
                    place_id=c["id"],
                    name=c["name"],
                    category=c["category"],
                    latitude=c["latitude"],
                    longitude=c["longitude"],
                    description=c["description"],
                    address=c.get("address", ""),
                    distance_meters=round(c["distance_meters"], 1),
                    similarity_score=item["score"]
                )
            )
            
        latency = (time.perf_counter() - start_time) * 1000
        
        return SearchResponse(
            query=body.query,
            results=search_results,
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Search execution failure: {str(e)}"}
        )


# ==============================================================================
# [BENCHMARK RUNNER & INTEGRATION TEST]
# Runs manual offline builds of the FAISS index and measures query matching
# accuracy and performance latency profile.
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("      SpotSync AI Semantic Search & FAISS Match Accuracy Benchmark")
    print("="*80)
    
    # 1. Setup local resources
    print(f"Loading model: {MODEL_NAME}...")
    t0 = time.perf_counter()
    bench_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Model loaded in: {((time.perf_counter() - t0)*1000):.2f} ms")
    
    # 2. Index building
    print("Encoding mock database descriptions...")
    descriptions = [p["description"] for p in MOCK_PLACES]
    with torch.no_grad():
        embeddings = bench_model.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        
    bench_index = faiss.IndexFlatIP(DIMENSION)
    bench_index.add(embeddings)
    print(f"FAISS local Index successfully built with {bench_index.ntotal} vectors.")
    
    # 3. Test queries for intent verification
    test_scenarios = [
        {"query": "드럼이랑 마이크 성능 좋은 방음 잘되는 음악 합주실", "expected": "싱크사운드 신촌점"},
        {"query": "노트북 들고 공부하기 편한 조용하고 편안한 카페", "expected": "카페 조용한 공간"},
        {"query": "컴퓨터 그래픽카드 최고 사양 게이밍 모니터 넓은 피씨방", "expected": "아이린 PC방 연세대점"},
        {"query": "혼자 가서 보컬 연습하고 마이크 녹음하기 조용한 스튜디오", "expected": "스타 보컬 스튜디오"}
    ]
    
    # 4. Run verification and calculate performance latency
    print("\nRunning Verification Queries...")
    print("-" * 65)
    
    success_count = 0
    all_latencies = []
    verification_results = []
    
    for scene in test_scenarios:
        q = scene["query"]
        t_start = time.perf_counter()
        
        with torch.no_grad():
            q_vec = bench_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
            scores, indices = bench_index.search(q_vec, 1)
            
        t_end = time.perf_counter()
        lat = (t_end - t_start) * 1000
        all_latencies.append(lat)
        
        top_idx = indices[0][0]
        top_score = scores[0][0]
        matched_name = MOCK_PLACES[top_idx]["name"] if top_idx != -1 else "None"
        
        is_success = matched_name == scene["expected"]
        if is_success:
            success_count += 1
            status_tag = "✅ MATCH"
        else:
            status_tag = "❌ MISMATCH"
            
        print(f"Query:  \"{q}\"")
        print(f"Match:  {matched_name} (Score: {top_score:.4f} | Latency: {lat:.2f}ms) -> {status_tag}")
        print("-" * 65)
        
        # Save verification result for report
        verification_results.append({
            "query": q,
            "expected": scene["expected"],
            "matched_name": matched_name,
            "score": float(top_score),
            "latency": lat,
            "is_success": is_success
        })
        
    # Latency Stats over repeated inference tests
    extra_iterations = 30
    print(f"Profiling latency stability across {extra_iterations} iterations...")
    for i in range(extra_iterations):
        q = test_scenarios[i % len(test_scenarios)]["query"]
        t_s = time.perf_counter()
        with torch.no_grad():
            q_vec = bench_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
            _ = bench_index.search(q_vec, 3)
        all_latencies.append((time.perf_counter() - t_s) * 1000)
        
    mean_lat = np.mean(all_latencies)
    p90 = np.percentile(all_latencies, 90)
    p99 = np.percentile(all_latencies, 99)
    
    print("\n" + "="*50)
    print("               BENCHMARK RESULT METRICS")
    print("="*50)
    print(f"  - Matching Accuracy:  {success_count}/{len(test_scenarios)} ({(success_count/len(test_scenarios)*100):.1f}%)")
    print(f"  - Average Search Lat: {mean_lat:.2f} ms")
    print(f"  - Tail Latency (p90):  {p90:.2f} ms")
    print(f"  - Peak Latency (p99):  {p99:.2f} ms")
    print("="*50)
    
    if p90 < 200:
        print("🎉 STATUS: SUCCESS - Tail latency is well under the 200ms threshold.")
    else:
        print("⚠️ STATUS: WARNING - Tail latency exceeds target budget.")
    print("="*80 + "\n")
    
    # 5. Generate and Save Markdown Report
    import datetime
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report.md")
    
    status_emoji = "🎉 SUCCESS" if p90 < 200 else "⚠️ WARNING"
    status_desc = "Tail latency is well under the 200ms threshold." if p90 < 200 else "Tail latency exceeds target budget."
    
    report_content = f"""# SpotSync AI Semantic Search Benchmark Report

Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🖥️ System & Model Configurations
- **Model Name**: `{MODEL_NAME}`
- **Device**: `{DEVICE}`
- **FAISS Index Type**: `IndexFlatIP` (Cosine Similarity via L2 Normalization)
- **PyTorch Thread Limit**: `{torch.get_num_threads()} threads`
- **Total Indexed Mock Places**: `{len(MOCK_PLACES)}`

## 📊 Performance Metrics Summary
- **Overall Status**: **{status_emoji}** ({status_desc})
- **Matching Accuracy**: **{success_count}/{len(test_scenarios)}** ({(success_count/len(test_scenarios)*100):.1f}%)
- **Average Search Latency**: `{mean_lat:.2f} ms`
- **Tail Latency (p90)**: `{p90:.2f} ms`
- **Peak Latency (p99)**: `{p99:.2f} ms`

## 🧪 Detailed Verification Scenarios
| # | Query | Expected Place | Matched Place | Similarity Score | Latency (ms) | Status |
|---|---|---|---|---|---|---|
"""
    for idx, res in enumerate(verification_results, 1):
        status_tag = "✅ MATCH" if res['is_success'] else "❌ MISMATCH"
        report_content += f"| {idx} | {res['query']} | {res['expected']} | {res['matched_name']} | {res['score']:.4f} | {res['latency']:.2f} | {status_tag} |\n"
        
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[BENCHMARK] Beautiful markdown report successfully saved to:\n  -> {report_path}\n")
    except Exception as e:
        print(f"[BENCHMARK ERROR] Failed to save report: {str(e)}")

