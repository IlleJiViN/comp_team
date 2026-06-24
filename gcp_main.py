import os
import time
import math
import asyncio
import httpx
import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from categories import CATEGORY_DESCRIPTIONS

load_dotenv()

# ==============================================================================
# [GCP Production Backend - SpotSync Multi-User Location Search]
# ==============================================================================

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"
DIMENSION = 768
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/spotsync")
KAKAO_API_KEY = os.environ.get("KAKAO_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

ALPHA_CATEGORY = 0.5   
BETA_NAME = 0.3        
GAMMA_RATING = 0.2     

class UserLocation(BaseModel):
    name: str
    latitude: float
    longitude: float

class SearchRequest(BaseModel):
    query: str
    users: List[UserLocation]
    radius_meters: float = 3000.0  # Increased default radius for multiple people
    category_threshold: float = 0.45
    similarity_threshold: float = 0.35
    top_k: int = 15

class UserDistanceInfo(BaseModel):
    name: str
    real_distance_m: Optional[float] = None
    real_time_sec: Optional[int] = None

class PlaceSearchResult(BaseModel):
    place_id: int
    name: str
    category: str
    latitude: float
    longitude: float
    description: str
    address: str
    straight_distance_m: float
    user_distances: List[UserDistanceInfo]
    avg_time_sec: Optional[float] = None
    max_time_sec: Optional[float] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    similarity_score: float
    final_score: float

class SearchResponse(BaseModel):
    query: str
    center_latitude: float
    center_longitude: float
    results: List[PlaceSearchResult]
    latency_ms: float

async def fetch_kakao_distance(client: httpx.AsyncClient, user_name, origin_lon, origin_lat, dest_lon, dest_lat):
    if not KAKAO_API_KEY:
        return user_name, None, None
    url = "https://apis-navi.kakaomobility.com/v1/directions"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {
        "origin": f"{origin_lon},{origin_lat}",
        "destination": f"{dest_lon},{dest_lat}"
    }
    try:
        resp = await client.get(url, headers=headers, params=params, timeout=3.0)
        data = resp.json()
        if "routes" in data and len(data["routes"]) > 0:
            summary = data["routes"][0]["summary"]
            return user_name, summary["distance"], summary["duration"]
    except Exception as e:
        print(f"Kakao API error for {user_name}: {e}")
    return user_name, None, None

async def fetch_google_rating(client: httpx.AsyncClient, name, address):
    if not GOOGLE_API_KEY:
        return None, None
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{name} {address}",
        "key": GOOGLE_API_KEY
    }
    try:
        resp = await client.get(url, params=params, timeout=3.0)
        data = resp.json()
        if "results" in data and len(data["results"]) > 0:
            best = data["results"][0]
            return best.get("rating", 0.0), best.get("user_ratings_total", 0)
    except Exception as e:
        print(f"Google API error: {e}")
    return None, None

async def enrich_place(client, place, users):
    dist_tasks = [
        fetch_kakao_distance(client, u.name, u.longitude, u.latitude, place["longitude"], place["latitude"]) 
        for u in users
    ]
    rate_task = fetch_google_rating(client, place["name"], place["address"])
    
    results = await asyncio.gather(*dist_tasks, rate_task)
    
    dist_results = results[:-1]
    rating_result = results[-1]
    
    user_distances = []
    valid_times = []
    for user_name, dist, time_sec in dist_results:
        user_distances.append({
            "name": user_name,
            "real_distance_m": dist,
            "real_time_sec": time_sec
        })
        if time_sec is not None:
            valid_times.append(time_sec)
            
    place["user_distances"] = user_distances
    place["avg_time_sec"] = sum(valid_times)/len(valid_times) if valid_times else None
    place["max_time_sec"] = max(valid_times) if valid_times else None
    
    rating, reviews = rating_result
    place["rating"] = rating if rating else 0.0
    place["review_count"] = reviews if reviews else 0
    return place

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[GCP_MAIN] Starting GCP Production Backend...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    category_names = list(CATEGORY_DESCRIPTIONS.keys())
    category_texts = list(CATEGORY_DESCRIPTIONS.values())
    with torch.no_grad():
        category_vectors = model.encode(category_texts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    engine = create_engine(DATABASE_URL)
    app.state.model = model
    app.state.engine = engine
    app.state.category_names = category_names
    app.state.category_vectors = category_vectors
    print("[GCP_MAIN] Ready.")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search", response_model=SearchResponse)
async def gcp_search(request: Request, body: SearchRequest):
    if not body.users:
        return SearchResponse(query=body.query, center_latitude=0, center_longitude=0, results=[], latency_ms=0)
        
    # Calculate Centroid
    center_lat = sum(u.latitude for u in body.users) / len(body.users)
    center_lon = sum(u.longitude for u in body.users) / len(body.users)
    
    model = request.app.state.model
    engine = request.app.state.engine
    cat_names = request.app.state.category_names
    cat_vectors = request.app.state.category_vectors
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        q_vec = model.encode([body.query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        
    cat_scores = np.dot(cat_vectors, q_vec.T).squeeze()
    matched_cats = []
    cat_sim_map = {}
    for i, s in enumerate(cat_scores):
        if float(s) >= body.category_threshold:
            matched_cats.append(cat_names[i])
            cat_sim_map[cat_names[i]] = float(s)
            
    if not matched_cats:
        return SearchResponse(query=body.query, center_latitude=center_lat, center_longitude=center_lon, results=[], latency_ms=0)
        
    deg = (body.radius_meters / 111000.0) * 1.1
    cat_placeholders = ", ".join([f":c{i}" for i in range(len(matched_cats))])
    params = {"lon": center_lon, "lat": center_lat, "deg": deg}
    for i, c in enumerate(matched_cats): params[f"c{i}"] = c
    
    sql = text(f"""
        SELECT id, name, category, latitude, longitude, address,
               ST_Distance(location::geography, ST_SetSRID(ST_Point(:lon, :lat), 4326)::geography) AS dist,
               embedding_vector_v4
        FROM places
        WHERE ST_DWithin(location, ST_SetSRID(ST_Point(:lon, :lat), 4326), :deg)
          AND category IN ({cat_placeholders})
          AND embedding_vector_v4 IS NOT NULL
        ORDER BY dist ASC LIMIT 200
    """)
    
    cands = []
    with engine.connect() as conn:
        res = conn.execute(sql, params)
        for r in res:
            cands.append({
                "id": r[0], "name": r[1], "category": r[2], "latitude": r[3], "longitude": r[4], 
                "address": r[5], "dist": float(r[6]), "emb": np.array(r[7], dtype='float32')
            })
            
    if not cands:
        return SearchResponse(query=body.query, center_latitude=center_lat, center_longitude=center_lon, results=[], latency_ms=0)
        
    name_vecs = np.stack([c["emb"] for c in cands])
    name_scores = np.atleast_1d(np.dot(name_vecs, q_vec.T).squeeze())
    
    scored = []
    for i, c in enumerate(cands):
        c_sim = cat_sim_map.get(c["category"], 0.0)
        n_sim = float(name_scores[i])
        score = ALPHA_CATEGORY * c_sim + BETA_NAME * max(n_sim, 0.0)
        if score >= body.similarity_threshold:
            c["semantic_score"] = score
            scored.append(c)
            
    scored.sort(key=lambda x: x["semantic_score"], reverse=True)
    top_k_items = scored[:body.top_k]
    
    async with httpx.AsyncClient() as client:
        tasks = [enrich_place(client, item, body.users) for item in top_k_items]
        enriched_items = await asyncio.gather(*tasks)
        
    results = []
    for item in enriched_items:
        norm_rating = item["rating"] / 5.0
        final_score = item["semantic_score"] + (norm_rating * GAMMA_RATING)
        
        # We can penalize places where max_time_sec is too high, but let's keep it simple for now
        if item["max_time_sec"]:
            # Small penalty if someone has to travel way longer than average
            variance_penalty = (item["max_time_sec"] - item["avg_time_sec"]) / 3600.0 # 1 hour = 1.0 penalty
            final_score -= (variance_penalty * 0.1)
            
        results.append(PlaceSearchResult(
            place_id=item["id"], name=item["name"], category=item["category"],
            latitude=item["latitude"], longitude=item["longitude"],
            description=CATEGORY_DESCRIPTIONS.get(item["category"], ""), address=item["address"],
            straight_distance_m=round(item["dist"], 1),
            user_distances=[UserDistanceInfo(**u) for u in item["user_distances"]],
            avg_time_sec=item["avg_time_sec"], max_time_sec=item["max_time_sec"],
            rating=item["rating"], review_count=item["review_count"],
            similarity_score=round(item["semantic_score"], 4), final_score=round(final_score, 4)
        ))
        
    results.sort(key=lambda x: x.final_score, reverse=True)
    latency = (time.perf_counter() - start_time) * 1000
    
    return SearchResponse(
        query=body.query, 
        center_latitude=center_lat, 
        center_longitude=center_lon, 
        results=results, 
        latency_ms=round(latency, 2)
    )
