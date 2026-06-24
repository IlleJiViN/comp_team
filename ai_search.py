import os
import time
import math
import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from categories import CATEGORY_DESCRIPTIONS

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
# [Architectural Design v4 - Decomposed Category Similarity Matching]
#
# Problem with v3 (Single Sentence Embedding):
#   - Entire place description (name + category hierarchy + category description,
#     200~300 chars) was compressed into a single 768-dim vector.
#   - This causes semantic dilution: query-irrelevant noise in the text
#     degrades similarity scores and ranking discriminability.
#
# v4 Solution (Decomposed Category Similarity):
#   1. Pre-cache category prototype vectors at startup (~50 categories).
#   2. At search time, match query against category prototypes first.
#   3. Use matched categories as a PostGIS filter (WHERE category IN ...).
#   4. Compute name similarity for fine-grained ranking within categories.
#   5. Final score = α × category_sim + β × name_sim (α=0.6, β=0.4).
#
# Benefits:
#   - Eliminates semantic dilution by comparing short, focused text units.
#   - DB queries become more targeted (only relevant categories).
#   - No DB schema changes or re-indexing required.
# ==============================================================================

# Thread tuning for hybrid core CPU (i5-13420H)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# Model configuration
MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"
DIMENSION = 768
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/spotsync")

# Scoring weights
ALPHA_CATEGORY = 0.7   # Weight for category similarity
BETA_NAME = 0.3        # Weight for name similarity

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
    category_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity to consider a category relevant.",
        example=0.45
    )
    similarity_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum final combined score threshold to return a place.",
        example=0.40
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
    similarity_score: float = Field(..., description="Combined similarity score (0.0 to 1.0).")
    category_score: float = Field(0.0, description="Category prototype similarity score.")
    name_score: float = Field(0.0, description="Place name similarity score.")

class SearchResponse(BaseModel):
    query: str = Field(..., description="The original query string.")
    matched_categories: List[str] = Field(default=[], description="Categories that matched the query intent.")
    results: List[PlaceSearchResult] = Field(..., description="List of matched places ranked by relevance within the radius.")
    latency_ms: float = Field(..., description="Search process latency in milliseconds.")


# Lifespan manager for FastAPI (Singleton Initialization)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    1. Loads the Embedding Model to app.state.model.
    2. Pre-caches category prototype vectors for decomposed similarity matching.
    3. Establishes a database connection pool to PostgreSQL / PostGIS container.
    """
    print(f"[STARTUP] Initializing SpotSync AI v4 (Decomposed Category Similarity)...")
    start_time = time.perf_counter()
    
    try:
        # Step 1: Load SentenceTransformer on CPU
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        
        # Warm up model to cache torch variables
        _ = model.encode("웜업", convert_to_numpy=True)
        
        # Step 2: Pre-cache category prototype vectors
        print(f"[STARTUP] Encoding {len(CATEGORY_DESCRIPTIONS)} category prototypes...")
        category_names = list(CATEGORY_DESCRIPTIONS.keys())
        category_texts = list(CATEGORY_DESCRIPTIONS.values())
        
        with torch.no_grad():
            category_vectors = model.encode(
                category_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')  # Shape: (num_categories, 768)
        
        print(f"[STARTUP] Category prototype matrix cached: {category_vectors.shape}")
        
        # Step 3: Establish DB Connection
        print(f"[STARTUP] Connecting to PostgreSQL/PostGIS DB...")
        engine = create_engine(DATABASE_URL)
        
        # Verify connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[STARTUP] PostgreSQL/PostGIS connection established successfully.")
        
        # Step 4: Initialize in-memory vector cache
        place_vectors_map = {}
        
        # Assign variables to global app state
        app.state.model = model
        app.state.engine = engine
        app.state.category_names = category_names
        app.state.category_vectors = category_vectors
        app.state.place_vectors_map = place_vectors_map
        
        duration = (time.perf_counter() - start_time) * 1000
        print(f"[STARTUP] v4 Decomposed Category Similarity engine initialized in {duration:.2f}ms.")
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
    description="v4 Decomposed Category Similarity: category-level prototype matching + name-level fine ranking.",
    version="4.0.0",
    lifespan=lifespan
)

@app.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Location Search (v4 Decomposed Category Similarity)",
    description="Matches query intent against category prototypes, filters candidates via PostGIS, then ranks by name similarity."
)
async def semantic_search(request: Request, body: SearchRequest):
    """
    v4 Decomposed Category Similarity Pipeline:
    1. Encode query vector.
    2. Match query against pre-cached category prototype vectors.
    3. Filter categories above threshold → use as PostGIS category filter.
    4. Query PostGIS for spatial + category-filtered candidates.
    5. Batch-encode candidate names → compute name similarity.
    6. Final score = α × category_sim + β × name_sim.
    7. Return top_k results sorted by final score.
    """
    model: SentenceTransformer = request.app.state.model
    engine = request.app.state.engine
    category_names: List[str] = request.app.state.category_names
    category_vectors: np.ndarray = request.app.state.category_vectors
    place_vectors_map: dict = getattr(request.app.state, "place_vectors_map", {})
    
    if not model or not engine:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "PostGIS connection or AI Model is not initialized."}
        )
        
    start_time = time.perf_counter()
    
    try:
        # ==================================================================
        # Step 1: Encode query vector
        # ==================================================================
        with torch.no_grad():
            query_vector = model.encode(
                [body.query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')  # Shape: (1, 768)
        
        # ==================================================================
        # Step 2: Category prototype matching
        # ==================================================================
        # Compute cosine similarity between query and all category prototypes
        # category_vectors shape: (num_categories, 768), query_vector shape: (1, 768)
        cat_scores = np.dot(category_vectors, query_vector.T).squeeze()  # Shape: (num_categories,)
        
        # Select categories above threshold
        matched_categories = []
        category_sim_map = {}  # category_name -> similarity_score
        
        for i, score in enumerate(cat_scores):
            if float(score) >= body.category_threshold:
                matched_categories.append(category_names[i])
                category_sim_map[category_names[i]] = float(score)
        
        # Sort matched categories by score for logging
        matched_categories.sort(key=lambda c: category_sim_map[c], reverse=True)
        
        # Removed early return so we do a fallback broad search
        
        # ==================================================================
        # Step 3: PostGIS spatial + category-filtered query
        # ==================================================================
        degrees = (body.radius_meters / 111000.0) * 1.1
        
        sql_params = {
            "lon": body.user_longitude,
            "lat": body.user_latitude,
            "degrees": degrees
        }
        
        if matched_categories:
            cat_placeholders = ", ".join([f":cat_{i}" for i in range(len(matched_categories))])
            for i, cat in enumerate(matched_categories):
                sql_params[f"cat_{i}"] = cat
            cat_condition = f"AND category IN ({cat_placeholders})"
        else:
            cat_condition = ""
        
        sql_query = text(f"""
            SELECT id, name, category, latitude, longitude, address,
                   ST_Distance(location::geography, ST_SetSRID(ST_Point(:lon, :lat), 4326)::geography) AS distance_meters
            FROM places
            WHERE location && ST_Expand(ST_SetSRID(ST_Point(:lon, :lat), 4326), :degrees)
              AND ST_DWithin(location, ST_SetSRID(ST_Point(:lon, :lat), 4326), :degrees)
              {cat_condition}
            ORDER BY distance_meters ASC
            LIMIT 100
        """)
        
        candidates = []
        with engine.connect() as conn:
            result = conn.execute(sql_query, sql_params)
            for r in result:
                candidates.append({
                    "id": r[0],
                    "name": r[1],
                    "category": r[2],
                    "latitude": r[3],
                    "longitude": r[4],
                    "address": r[5],
                    "distance_meters": float(r[6])
                })
        
        if not candidates:
            latency = (time.perf_counter() - start_time) * 1000
            return SearchResponse(
                query=body.query,
                matched_categories=matched_categories,
                results=[],
                latency_ms=round(latency, 2)
            )
        
        # Dynamic cached encoding for name matching
        candidate_vectors = []
        fallback_names = []
        fallback_indices = []
        fallback_pids = []
        
        for idx, c in enumerate(candidates):
            pid = c["id"]
            if pid in place_vectors_map:
                candidate_vectors.append(place_vectors_map[pid])
            else:
                candidate_vectors.append(None)
                fallback_names.append(f"{c['name']} {c['category']}")
                fallback_indices.append(idx)
                fallback_pids.append(pid)
                
        if fallback_names:
            with torch.no_grad():
                encoded_fallback = model.encode(fallback_names, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
            for i, f_idx in enumerate(fallback_indices):
                vec = encoded_fallback[i]
                candidate_vectors[f_idx] = vec
                place_vectors_map[fallback_pids[i]] = vec # Cache for future
                
        name_vectors = np.array(candidate_vectors, dtype=np.float32)
        
        # Compute name similarity scores
        name_scores = np.dot(name_vectors, query_vector.T).squeeze()
        name_scores = np.atleast_1d(name_scores)
        
        # ==================================================================
        # Step 5: Compute final combined score and rank
        # ==================================================================
        scored_candidates = []
        for idx, candidate in enumerate(candidates):
            cat_sim = category_sim_map.get(candidate["category"], 0.0)
            name_sim = float(name_scores[idx])
            
            # Weighted combination
            final_score = ALPHA_CATEGORY * cat_sim + BETA_NAME * max(name_sim, 0.0)
            
            if final_score >= body.similarity_threshold:
                scored_candidates.append({
                    "candidate": candidate,
                    "final_score": final_score,
                    "cat_score": cat_sim,
                    "name_score": name_sim
                })
        
        # Sort by final score descending
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # ==================================================================
        # Step 6: Assemble response
        # ==================================================================
        search_results = []
        for item in scored_candidates[:body.top_k]:
            c = item["candidate"]
            cat_desc = CATEGORY_DESCRIPTIONS.get(c["category"], "")
            search_results.append(
                PlaceSearchResult(
                    place_id=c["id"],
                    name=c["name"],
                    category=c["category"],
                    latitude=c["latitude"],
                    longitude=c["longitude"],
                    description=cat_desc,
                    address=c.get("address", ""),
                    distance_meters=round(c["distance_meters"], 1),
                    similarity_score=round(item["final_score"], 4),
                    category_score=round(item["cat_score"], 4),
                    name_score=round(item["name_score"], 4)
                )
            )
            
        latency = (time.perf_counter() - start_time) * 1000
        
        return SearchResponse(
            query=body.query,
            matched_categories=matched_categories,
            results=search_results,
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Search execution failure: {str(e)}"}
        )


# ==============================================================================
# [BENCHMARK RUNNER - v4 Decomposed Category Similarity]
# Tests category matching accuracy, name-level ranking, and latency profile.
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("      SpotSync AI v4 - Decomposed Category Similarity Benchmark")
    print("="*80)
    
    # 1. Setup local resources
    print(f"Loading model: {MODEL_NAME}...")
    t0 = time.perf_counter()
    bench_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Model loaded in: {((time.perf_counter() - t0)*1000):.2f} ms")
    
    # 2. Build category prototype vectors
    print(f"Encoding {len(CATEGORY_DESCRIPTIONS)} category prototypes...")
    cat_names = list(CATEGORY_DESCRIPTIONS.keys())
    cat_texts = list(CATEGORY_DESCRIPTIONS.values())
    
    with torch.no_grad():
        cat_vectors = bench_model.encode(cat_texts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    print(f"Category prototype matrix: {cat_vectors.shape}")
    
    # 3. Encode mock place names
    mock_name_texts = [f"{p['name']} {p['category']}" for p in MOCK_PLACES]
    mock_categories = [p["category"] for p in MOCK_PLACES]
    
    with torch.no_grad():
        mock_name_vectors = bench_model.encode(mock_name_texts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    
    # 4. Test queries
    test_scenarios = [
        {"query": "드럼이랑 마이크 성능 좋은 방음 잘되는 음악 합주실", "expected": "싱크사운드 신촌점"},
        {"query": "노트북 들고 공부하기 편한 조용하고 편안한 카페", "expected": "카페 조용한 공간"},
        {"query": "컴퓨터 그래픽카드 최고 사양 게이밍 모니터 넓은 피씨방", "expected": "아이린 PC방 연세대점"},
        {"query": "혼자 가서 보컬 연습하고 마이크 녹음하기 조용한 스튜디오", "expected": "스타 보컬 스튜디오"}
    ]
    
    # 5. Run v4 decomposed benchmark
    print("\nRunning v4 Decomposed Category Similarity Verification...")
    print("-" * 80)
    
    success_count = 0
    all_latencies = []
    verification_results = []
    
    for scene in test_scenarios:
        q = scene["query"]
        t_start = time.perf_counter()
        
        with torch.no_grad():
            q_vec = bench_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        
        # Step A: Category matching
        cat_scores = np.dot(cat_vectors, q_vec.T).squeeze()
        matched_cats = []
        cat_sim_map = {}
        for i, score in enumerate(cat_scores):
            if float(score) >= 0.45:
                matched_cats.append(cat_names[i])
                cat_sim_map[cat_names[i]] = float(score)
        
        # Step B: Filter mock places by matched categories
        filtered_indices = [i for i, c in enumerate(mock_categories) if c in cat_sim_map]
        
        if not filtered_indices:
            # Fallback: use all places
            filtered_indices = list(range(len(MOCK_PLACES)))
        
        # Step C: Compute name similarity for filtered places
        filtered_name_vecs = mock_name_vectors[filtered_indices]
        name_scores = np.dot(filtered_name_vecs, q_vec.T).squeeze()
        name_scores = np.atleast_1d(name_scores)
        
        # Step D: Combined score
        best_score = -1.0
        best_idx = -1
        for rank, fi in enumerate(filtered_indices):
            cat_sim = cat_sim_map.get(mock_categories[fi], 0.0)
            n_sim = float(name_scores[rank])
            final = ALPHA_CATEGORY * cat_sim + BETA_NAME * max(n_sim, 0.0)
            if final > best_score:
                best_score = final
                best_idx = fi
        
        t_end = time.perf_counter()
        lat = (t_end - t_start) * 1000
        all_latencies.append(lat)
        
        matched_name = MOCK_PLACES[best_idx]["name"] if best_idx >= 0 else "None"
        matched_category = MOCK_PLACES[best_idx]["category"] if best_idx >= 0 else "None"
        
        is_success = matched_name == scene["expected"]
        if is_success:
            success_count += 1
            status_tag = "✅ MATCH"
        else:
            status_tag = "❌ MISMATCH"
            
        # Show matched categories
        top_cats = sorted(cat_sim_map.items(), key=lambda x: x[1], reverse=True)[:5]
        cat_display = ", ".join([f"{c}({s:.3f})" for c, s in top_cats])
        
        print(f"Query:    \"{q}\"")
        print(f"Cat Match: [{cat_display}]")
        print(f"Result:   {matched_name} [{matched_category}] (Score: {best_score:.4f} | Latency: {lat:.2f}ms) -> {status_tag}")
        print("-" * 80)
        
        verification_results.append({
            "query": q,
            "expected": scene["expected"],
            "matched_name": matched_name,
            "matched_category": matched_category,
            "score": float(best_score),
            "matched_categories": [c for c, _ in top_cats],
            "latency": lat,
            "is_success": is_success
        })
        
    # Latency profiling
    extra_iterations = 30
    print(f"Profiling latency stability across {extra_iterations} iterations...")
    for i in range(extra_iterations):
        q = test_scenarios[i % len(test_scenarios)]["query"]
        t_s = time.perf_counter()
        with torch.no_grad():
            q_vec = bench_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
            cat_scores = np.dot(cat_vectors, q_vec.T).squeeze()
            name_scores = np.dot(mock_name_vectors, q_vec.T).squeeze()
        all_latencies.append((time.perf_counter() - t_s) * 1000)
        
    mean_lat = np.mean(all_latencies)
    p90 = np.percentile(all_latencies, 90)
    p99 = np.percentile(all_latencies, 99)
    
    print("\n" + "="*50)
    print("          v4 BENCHMARK RESULT METRICS")
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
    
    # Generate Markdown Report
    import datetime
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report.md")
    
    status_emoji = "🎉 SUCCESS" if p90 < 200 else "⚠️ WARNING"
    status_desc = "Tail latency is well under the 200ms threshold." if p90 < 200 else "Tail latency exceeds target budget."
    
    report_content = f"""# SpotSync AI v4 - Decomposed Category Similarity Benchmark Report

Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏗️ Architecture: Decomposed Category Similarity (v4)
- **Approach**: Query → Category Prototype Matching → PostGIS Category Filter → Name Similarity → Weighted Score
- **Scoring Formula**: `final = {ALPHA_CATEGORY} × category_sim + {BETA_NAME} × name_sim`
- **Category Threshold**: `0.45`

## 🖥️ System & Model Configurations
- **Model Name**: `{MODEL_NAME}`
- **Device**: `{DEVICE}`
- **Category Prototypes**: `{len(CATEGORY_DESCRIPTIONS)}` categories pre-cached
- **PyTorch Thread Limit**: `{torch.get_num_threads()} threads`
- **Total Mock Places**: `{len(MOCK_PLACES)}`

## 📊 Performance Metrics Summary
- **Overall Status**: **{status_emoji}** ({status_desc})
- **Matching Accuracy**: **{success_count}/{len(test_scenarios)}** ({(success_count/len(test_scenarios)*100):.1f}%)
- **Average Search Latency**: `{mean_lat:.2f} ms`
- **Tail Latency (p90)**: `{p90:.2f} ms`
- **Peak Latency (p99)**: `{p99:.2f} ms`

## 🧪 Detailed Verification Scenarios
| # | Query | Expected | Matched | Category | Score | Latency (ms) | Status |
|---|-------|----------|---------|----------|-------|-------------|--------|
"""
    for idx, res in enumerate(verification_results, 1):
        status_tag = "✅ MATCH" if res['is_success'] else "❌ MISMATCH"
        cats = ", ".join(res['matched_categories'][:3])
        report_content += f"| {idx} | {res['query']} | {res['expected']} | {res['matched_name']} | {cats} | {res['score']:.4f} | {res['latency']:.2f} | {status_tag} |\n"
        
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[BENCHMARK] Report saved to:\n  -> {report_path}\n")
    except Exception as e:
        print(f"[BENCHMARK ERROR] Failed to save report: {str(e)}")
