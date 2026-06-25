# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

# Global memory
GLOBAL_IDS = None
GLOBAL_VECTORS = None
model = None

from typing import List

class UserLocation(BaseModel):
    name: str
    latitude: float
    longitude: float

class SearchRequest(BaseModel):
    query: str
    users: List[UserLocation]
    radius_meters: float = 3000.0
    top_k: int = 15

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, GLOBAL_VECTORS, GLOBAL_IDS
    print("[STARTUP] Initializing SpotSync AI v5 (Hybrid Vector Search)...")
    
    model = SentenceTransformer("jhgan/ko-sroberta-multitask", device="cpu")
    
    print("[STARTUP] Loading NPZ dump to RAM...")
    start = time.time()
    data = np.load("embeddings_v4.npz")
    
    raw_ids = data['ids']
    raw_vectors = data['vectors']
    
    # Sort by ID for fast binary search (searchsorted)
    sort_idx = np.argsort(raw_ids)
    GLOBAL_IDS = raw_ids[sort_idx]
    GLOBAL_VECTORS = raw_vectors[sort_idx]
    
    print(f"[STARTUP] Loaded {len(GLOBAL_IDS)} vectors in {time.time() - start:.2f}s")
    yield
    print("[SHUTDOWN] Clearing memory...")
    del GLOBAL_IDS
    del GLOBAL_VECTORS
    del model

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
def search_places(req: SearchRequest):
    start_time = time.time()
    
    try:
        conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/spotsync")
        cur = conn.cursor()
        
        # Calculate centroid of all users
        if not req.users:
            raise HTTPException(status_code=400, detail="No users provided")
            
        center_lat = sum(u.latitude for u in req.users) / len(req.users)
        center_lon = sum(u.longitude for u in req.users) / len(req.users)
        
        # 1. Spatial Filtering (PostGIS)
        cur.execute(f"""
            SELECT id, name, category, latitude, longitude, address, 
                   ST_DistanceSphere(location, ST_SetSRID(ST_MakePoint({center_lon}, {center_lat}), 4326)) as dist
            FROM places
            WHERE ST_DWithin(location, ST_SetSRID(ST_MakePoint({center_lon}, {center_lat}), 4326), {req.radius_meters / 111320.0})
        """)
        candidates = cur.fetchall()
        cur.close()
        conn.close()
        
        if not candidates:
            return {"query": req.query, "results": [], "latency_ms": (time.time() - start_time) * 1000}
            
        candidate_ids = np.array([row[0] for row in candidates], dtype=np.int32)
        
        # 2. Map candidate IDs to GLOBAL_VECTORS index
        # searchsorted finds the index where the id would be inserted to maintain order.
        idxs = np.searchsorted(GLOBAL_IDS, candidate_ids)
        
        # Filter out IDs that don't actually exist in GLOBAL_IDS (just in case)
        valid_mask = (idxs < len(GLOBAL_IDS)) & (GLOBAL_IDS[idxs] == candidate_ids)
        valid_idxs = idxs[valid_mask]
        valid_candidates = [candidates[i] for i in range(len(candidates)) if valid_mask[i]]
        
        if len(valid_candidates) == 0:
            return {"query": req.query, "results": [], "latency_ms": (time.time() - start_time) * 1000}
            
        candidate_vectors = GLOBAL_VECTORS[valid_idxs]
        
        # 3. Compute Query Embedding
        query_embedding = model.encode([req.query], normalize_embeddings=True)[0]
        
        # 4. Dense Vector Similarity (Dot Product)
        similarities = np.dot(candidate_vectors, query_embedding)
        
        # 5. Hybrid Keyword Boost (Solves the "dilution" problem user mentioned)
        # Give a massive boost if query keywords appear in the name or category
        query_words = set(req.query.split())
        final_scores = []
        
        for i, row in enumerate(valid_candidates):
            name = str(row[1])
            category = str(row[2])
            sim = float(similarities[i])
            
            boost = 0.0
            for word in query_words:
                if len(word) >= 2: # Ignore 1-character particles
                    if word in name:
                        boost += 0.4
                    elif word in category:
                        boost += 0.2
                        
            final_scores.append(sim + boost)
            
        final_scores = np.array(final_scores)
        
        # 6. Sort and get Top 20
        top_k = min(20, len(valid_candidates))
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            row = valid_candidates[idx]
            results.append({
                "place_id": row[0],
                "name": row[1],
                "category": row[2],
                "latitude": row[3],
                "longitude": row[4],
                "address": row[5],
                "distance_meters": round(row[6], 1),
                "similarity_score": round(final_scores[idx], 4)
            })
            
        return {
            "query": req.query,
            "results": results,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
