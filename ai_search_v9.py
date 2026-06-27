from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification
import time
import psycopg2
import os
import json
import requests
import secrets
from dotenv import load_dotenv

# Gemini API 제거됨 - 유료 API 호출 없음

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
FRONTEND_URL = "http://localhost:5173"

app = FastAPI(title="SpotSync AI Search V8 (NER + RAG Streaming)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embed_model = None
try:
    print("[INFO] Loading BGE-M3 model for embeddings...")
    embed_model = SentenceTransformer('BAAI/bge-m3', device='cpu')
except Exception as e:
    print(f"[WARN] Could not load embedding model: {e}")


def get_embedding(text: str):
    if embed_model is None:
        return np.zeros(384, dtype=np.float32)
    return embed_model.encode(text, normalize_embeddings=True)


ner_model_path = "./models/spotsync-ner"
ner_tokenizer = None
ner_model = None
label_list = []
try:
    print("[INFO] Loading SpotSync NER model...")
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    with open(os.path.join(ner_model_path, "label_config.json"), "r", encoding="utf-8") as f:
        label_config = json.load(f)
    label_list = label_config.get("label_list", [])
except Exception as e:
    print(f"[WARN] Could not load NER model: {e}")

# LLM 없음 - 로컬 요약만 사용

try:
    from elasticsearch import Elasticsearch
except Exception as e:
    Elasticsearch = None
    print(f"[WARN] Elasticsearch package unavailable: {e}")

if Elasticsearch is not None:
    print("[INFO] Connecting to Elasticsearch...")
    es = Elasticsearch("http://localhost:9200", request_timeout=10)
else:
    es = None

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/auth/naver/login")
def naver_login():
    state = secrets.token_urlsafe(16)
    redirect_uri = f"{FRONTEND_URL}"
    url = f"https://nid.naver.com/oauth2.0/authorize?response_type=code&client_id={NAVER_CLIENT_ID}&redirect_uri={redirect_uri}&state={state}"
    return {"url": url}

@app.post("/auth/naver/token")
async def naver_token(req: Request):
    data = await req.json()
    code = data.get("code")
    state = data.get("state")
    
    token_url = "https://nid.naver.com/oauth2.0/token"
    res = requests.get(token_url, params={
        "grant_type": "authorization_code",
        "client_id": NAVER_CLIENT_ID,
        "client_secret": NAVER_CLIENT_SECRET,
        "code": code,
        "state": state
    }).json()
    
    access_token = res.get("access_token")
    if not access_token:
        return {"error": "Failed to get access token"}
        
    profile_res = requests.get("https://openapi.naver.com/v1/nid/me", headers={
        "Authorization": f"Bearer {access_token}"
    }).json()
    
    if profile_res.get("resultcode") != "00":
        return {"error": "Failed to get profile"}
        
    p = profile_res["response"]
    naver_id = p.get("id")
    name = p.get("name", "네이버유저")
    email = p.get("email", "")
    profile_image = p.get("profile_image", "")
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (provider, provider_id, name, email, profile_image)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (provider, provider_id) 
        DO UPDATE SET name=EXCLUDED.name, profile_image=EXCLUDED.profile_image
        RETURNING id, name, profile_image
    """, ("naver", naver_id, name, email, profile_image))
    user_row = cur.fetchone()
    conn.commit()
    conn.close()
    
    return {
        "user_id": user_row[0],
        "name": user_row[1],
        "profile_image": user_row[2]
    }

@app.get("/auth/kakao/login")
def kakao_login():
    redirect_uri = f"{FRONTEND_URL}"
    state = "kakao"
    url = f"https://kauth.kakao.com/oauth/authorize?client_id={KAKAO_API_KEY}&redirect_uri={redirect_uri}&response_type=code&state={state}"
    return {"url": url}

@app.post("/auth/kakao/token")
async def kakao_token(req: Request):
    data = await req.json()
    code = data.get("code")
    
    token_url = "https://kauth.kakao.com/oauth/token"
    res = requests.post(token_url, data={
        "grant_type": "authorization_code",
        "client_id": KAKAO_API_KEY,
        "redirect_uri": f"{FRONTEND_URL}",
        "code": code
    }).json()
    
    access_token = res.get("access_token")
    if not access_token:
        return {"error": "Failed to get Kakao access token", "details": res}
        
    profile_res = requests.get("https://kapi.kakao.com/v2/user/me", headers={
        "Authorization": f"Bearer {access_token}"
    }).json()
    print(f"[DEBUG] Kakao Profile Response: {profile_res}", flush=True)
    
    kakao_id = str(profile_res.get("id"))
    props = profile_res.get("properties") or {}
    kakao_account = profile_res.get("kakao_account") or {}
    profile = kakao_account.get("profile") or {}
    
    name = props.get("nickname") or profile.get("nickname") or "카카오유저"
    email = kakao_account.get("email") or ""
    profile_image = props.get("profile_image") or profile.get("thumbnail_image_url") or profile.get("profile_image_url") or ""
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (provider, provider_id, name, email, profile_image)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (provider, provider_id) 
        DO UPDATE SET name=EXCLUDED.name, profile_image=EXCLUDED.profile_image
        RETURNING id, name, profile_image
    """, ("kakao", kakao_id, name, email, profile_image))
    user_row = cur.fetchone()
    conn.commit()
    conn.close()
    
    return {
        "user_id": user_row[0],
        "name": user_row[1],
        "profile_image": user_row[2]
    }

class RoomCreateRequest(BaseModel):
    host_id: int
    latitude: float
    longitude: float

class RoomJoinRequest(BaseModel):
    room_id: str
    user_id: int
    latitude: float
    longitude: float

class LocationUpdateRequest(BaseModel):
    room_id: str
    user_id: int
    latitude: float
    longitude: float

@app.post("/room/create")
def create_room(req: RoomCreateRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    import string
    import random
    while True:
        room_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        cur.execute("SELECT id FROM rooms WHERE id = %s", (room_id,))
        if not cur.fetchone():
            break
            
    try:
        cur.execute("INSERT INTO rooms (id, host_id) VALUES (%s, %s)", (room_id, req.host_id))
        cur.execute("""
            INSERT INTO room_members (room_id, user_id, latitude, longitude)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (room_id, user_id)
            DO UPDATE SET latitude=EXCLUDED.latitude, longitude=EXCLUDED.longitude
        """, (room_id, req.host_id, req.latitude, req.longitude))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return {"error": f"Failed to create room: {e}"}
        
    conn.close()
    return {"room_id": room_id}

@app.post("/room/join")
def join_room(req: RoomJoinRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM rooms WHERE id = %s", (req.room_id,))
    if not cur.fetchone():
        conn.close()
        return {"error": "존재하지 않는 방입니다."}
        
    try:
        cur.execute("""
            INSERT INTO room_members (room_id, user_id, latitude, longitude)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (room_id, user_id)
            DO UPDATE SET latitude=EXCLUDED.latitude, longitude=EXCLUDED.longitude
        """, (req.room_id, req.user_id, req.latitude, req.longitude))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return {"error": f"Failed to join room: {e}"}
        
    conn.close()
    return {"success": True}

@app.get("/room/{room_id}/members")
def get_room_members(room_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT rm.user_id, u.name, u.profile_image, rm.latitude, rm.longitude
        FROM room_members rm
        JOIN users u ON rm.user_id = u.id
        WHERE rm.room_id = %s
        ORDER BY rm.joined_at ASC
    """, (room_id,))
    rows = cur.fetchall()
    conn.close()
    
    members = []
    for r in rows:
        members.append({
            "user_id": r[0],
            "name": r[1],
            "profile_image": r[2],
            "latitude": r[3],
            "longitude": r[4]
        })
    return {"members": members}

@app.post("/room/update_location")
def update_location(req: LocationUpdateRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE room_members
            SET latitude = %s, longitude = %s
            WHERE room_id = %s AND user_id = %s
        """, (req.latitude, req.longitude, req.room_id, req.user_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return {"error": f"Failed to update location: {e}"}
    conn.close()
    return {"success": True}

class Location(BaseModel):
    name: Optional[str] = "익명"
    lat: float
    lng: float

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3
    user_locations: List[Location] = []

def calculate_midpoint(locations: List[Location]):
    if not locations:
        return None
    avg_lat = sum(loc.lat for loc in locations) / len(locations)
    avg_lng = sum(loc.lng for loc in locations) / len(locations)
    return avg_lat, avg_lng

def extract_entities(text: str):
    tokens = text.split()
    if not tokens or ner_tokenizer is None or ner_model is None or not label_list:
        return {"location": [], "brand": [], "category": [], "attribute": []}
    
    inputs = ner_tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = ner_model(**inputs)
        
    predictions = torch.argmax(outputs.logits, dim=2)[0]
    word_ids = inputs.word_ids()
    
    entities = {"location": [], "brand": [], "category": [], "attribute": []}
    current_entity = None
    current_tokens = []
    
    prev_word_idx = None
    results = []
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == prev_word_idx:
            continue
        tag = label_list[predictions[idx].item()]
        results.append((tokens[word_idx], tag))
        prev_word_idx = word_idx
        
    for token, tag in results:
        if tag.startswith("B-"):
            if current_entity:
                key = {"LOC": "location", "BRAND": "brand", "CAT": "category", "ATTR": "attribute"}[current_entity]
                entities[key].append(" ".join(current_tokens))
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity == tag[2:]:
            current_tokens.append(token)
        else:
            if current_entity:
                key = {"LOC": "location", "BRAND": "brand", "CAT": "category", "ATTR": "attribute"}[current_entity]
                entities[key].append(" ".join(current_tokens))
            current_entity = None
            current_tokens = []
            
    if current_entity:
        key = {"LOC": "location", "BRAND": "brand", "CAT": "category", "ATTR": "attribute"}[current_entity]
        entities[key].append(" ".join(current_tokens))
        
    return entities

def extract_filters(entities):
    # Strict filters removed: 'region' and 'category' mappings are too inconsistent 
    # (e.g. Kakao gives '음식점', NER gives '국밥집').
    # We rely purely on KNN + BM25 boosts for recall.
    return []

@app.post("/search_rag")
async def search_rag(req: SearchQuery):
    t_start = time.time()
    
    # NER Inference
    entities = extract_entities(req.query)
    print(f"[NER] Query: '{req.query}' -> Extracted: {entities}")
    
    query_emb = get_embedding(req.query)
    # Filter by Category
    es_filters = extract_filters(entities)

    # 1. PostGIS Geo-Filtering (Radius search) before ES retrieval
    midpoint = calculate_midpoint(req.user_locations)
    nearby_place_ids = []
    if midpoint:
        mid_lat, mid_lng = midpoint
        conn = get_db_connection()
        cur = conn.cursor()
        # Dynamic radius expansion: 7km -> 15km -> 50km
        for radius in [7000, 15000, 50000]:
            cur.execute("""
                SELECT id FROM places 
                WHERE ST_DWithin(location::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
            """, (mid_lng, mid_lat, radius))
            nearby_place_ids = [row[0] for row in cur.fetchall()]
            if len(nearby_place_ids) >= 10:
                print(f"[GEO-FILTER] Found {len(nearby_place_ids)} places within {radius}m of midpoint")
                break
        conn.close()
        
        if nearby_place_ids:
            es_filters.append({"terms": {"place_id": nearby_place_ids}})
        else:
            print("[GEO-FILTER] No places found within 50km of midpoint. Skipping location filter.")
    
    # 1. Elasticsearch Hybrid Query (KNN + BM25)
    should_clauses = [
        {"match": {"name": {"query": req.query, "boost": 2.0}}},
        {"match": {"category": {"query": req.query, "boost": 1.0}}}
    ]
    
    # Location Boost
    for loc in entities["location"]:
        should_clauses.append({"match": {"text": {"query": loc, "boost": 3.0}}})
    
    # If Brand is detected, heavily boost brand match in name
    for brand in entities["brand"]:
        should_clauses.append({"match": {"name": {"query": brand, "boost": 5.0}}})
    
    # If Attribute is detected, boost text/chunk matches
    for attr in entities["attribute"]:
        should_clauses.append({"match": {"text": {"query": attr, "boost": 3.0}}})

    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_emb.tolist(),
            "k": 50,
            "num_candidates": 500,
            "boost": 0.8  # Semantic Weight
        },
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 0
            }
        },
        "size": 150,
        "collapse": {
            "field": "place_id"
        }
    }
    print(f"DEBUG ES QUERY BODY: {json.dumps(body, ensure_ascii=False)}", flush=True)
    
    if es_filters:
        body["knn"]["filter"] = es_filters
        body["query"]["bool"]["filter"] = es_filters
        print(f"DEBUG ES_FILTERS ADDED: {es_filters}", flush=True)
    
    try:
        res = es.options(request_timeout=15).search(index="spotsync_chunks", body=body)
    except Exception as e:
        return {"error": f"Elasticsearch query failed: {e}"}
        
    hits = res['hits']['hits']
    print(f"DEBUG: Initial hits = {len(hits)}", flush=True)
    
    # --- LIVE FALLBACK MECHANISM ---
    KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
    
    is_hallucination = False
    if len(hits) > 0:
        # We consider it hallucination if the top 5 results don't contain any of the extracted brand or category.
        # If no brand/category were extracted, fallback to the old core keyword logic.
        core_keywords = entities["brand"] + entities["category"]
        if not core_keywords:
            core_keywords = [w for w in req.query.split() if w not in ["마포", "마포구", "홍대", "합정", "망원", "신촌", "연남", "상수", "근처", "어디야", "찾아줘"]]
            
        if core_keywords:
            top_names = [h['_source']['name'] for h in hits[:5]]
            top_cats = [h['_source']['category'] for h in hits[:5]]
            match_found = False
            for kw in core_keywords:
                for name, cat in zip(top_names, top_cats):
                    if kw in name or kw in cat:
                        match_found = True
                        break
            if not match_found:
                is_hallucination = True

    fallback_success = False
    if (len(hits) == 0 or is_hallucination) and KAKAO_API_KEY:
        # Construct a precise keyword query for Kakao API using NER entities
        kakao_query_parts = []
        if entities["location"]:
            kakao_query_parts.extend(entities["location"])
        if entities["brand"]:
            kakao_query_parts.extend(entities["brand"])
        if entities["category"]:
            kakao_query_parts.extend(entities["category"])
            
        kakao_query = " ".join(kakao_query_parts)
        if not kakao_query:
            kakao_query = req.query # fallback to original if NER failed completely
            
        print(f"[FALLBACK] Triggering Kakao Local API fallback for '{kakao_query}' (Original: {req.query}, Hits: {len(hits)})")
        
        local_url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {
            "Authorization": f"KakaoAK {KAKAO_API_KEY}",
            "Origin": "http://localhost:5173",
            "KA": "sdk/1.0 os/javascript lang/en-US device/Win32 origin/http%3A%2F%2Flocalhost%3A5173"
        }
        kakao_params = {"query": kakao_query, "size": 3}
        midpoint = calculate_midpoint(req.user_locations)
        if midpoint:
            # y is lat, x is lng
            kakao_params.update({"y": midpoint[0], "x": midpoint[1], "radius": 20000})

        try:
            local_res = requests.get(local_url, headers=headers, params=kakao_params, timeout=5).json()
            docs = local_res.get("documents", [])
            print(f"[DEBUG] Kakao retry returned {len(docs)} documents.", flush=True)
            
            # If Kakao query fails (0 hits), retry with just the category
            if not docs and entities.get("category"):
                retry_query = " ".join(entities["category"])
                print(f"[FALLBACK] Retry with category only: '{retry_query}'", flush=True)
                kakao_params["query"] = retry_query
                local_res = requests.get(local_url, headers=headers, params=kakao_params, timeout=5).json()
                docs = local_res.get("documents", [])
                print(f"[DEBUG] Kakao retry returned {len(docs)} documents.", flush=True)
            
            if docs:
                print(f"[DEBUG] Found {len(docs)} documents from Kakao API. Inserting to DB...", flush=True)
                conn = get_db_connection()
                cur = conn.cursor()
                new_place_ids = []
                
                for doc in docs:
                    place_id = doc.get("id", "KAKAO_LOCAL")
                    name = doc.get("place_name", "")
                    address = doc.get("road_address_name", "") or doc.get("address_name", "")
                    category = doc.get("category_group_name", "") or "소매업"
                    lat = float(doc.get("y", 0.0))
                    lng = float(doc.get("x", 0.0))
                    
                    cur.execute("""
                        INSERT INTO places (place_id, name, category, address, latitude, longitude, location, embedding_text, is_enriched, description, is_premium)
                        VALUES (%s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), '', FALSE, '', FALSE)
                        ON CONFLICT (place_id) DO UPDATE SET name = EXCLUDED.name
                        RETURNING id
                    """, (place_id, name, category, address, lat, lng, lng, lat))
                    result = cur.fetchone()
                    if result:
                        pid = result[0]
                        conn.commit()
                        new_place_ids.append((pid, name, address))
                
                print(f"[DEBUG] DB insertion completed. Proceeding to fetch blogs for {len(new_place_ids)} places...", flush=True)
                
                for pid, name, addr in new_place_ids:
                    print(f"[DEBUG] Fetching blogs for place: {name}", flush=True)
                    contents = []
                    metadata_list = []
                    
                    # 1. Kakao Blog Search (Max 3)
                    try:
                        kakao_blog_url = "https://dapi.kakao.com/v2/search/blog"
                        k_res = requests.get(kakao_blog_url, headers=headers, params={"query": f"{addr} {name}", "size": 1}, timeout=5).json()
                        for b_doc in k_res.get("documents", []):
                            text = b_doc.get("contents", "").replace("<b>", "").replace("</b>", "").strip()
                            if text: 
                                contents.append(text)
                                metadata_list.append({
                                    "source": "kakao",
                                    "title": b_doc.get("title", "").replace("<b>", "").replace("</b>", "").strip(),
                                    "url": b_doc.get("url", ""),
                                    "postdate": b_doc.get("datetime", "")[:10],
                                    "thumbnail": b_doc.get("thumbnail", "")
                                })
                    except Exception as e:
                        print(f"Kakao blog search error: {e}")

                    # 2. Naver Blog Search (Max 15)
                    if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
                        try:
                            naver_blog_url = "https://openapi.naver.com/v1/search/blog.json"
                            n_headers = {
                                "X-Naver-Client-Id": NAVER_CLIENT_ID,
                                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
                            }
                            n_res = requests.get(naver_blog_url, headers=n_headers, params={"query": f"{addr} {name}", "display": 1}, timeout=5).json()
                            for item in n_res.get("items", []):
                                text = item.get("description", "").replace("<b>", "").replace("</b>", "").strip()
                                if text:
                                    contents.append(text)
                                    metadata_list.append({
                                        "source": "naver",
                                        "title": item.get("title", "").replace("<b>", "").replace("</b>", "").strip(),
                                        "url": item.get("link", ""),
                                        "postdate": item.get("postdate", ""),
                                        "thumbnail": ""
                                    })
                        except Exception as e:
                            print(f"Naver blog search error: {e}")
                    
                    combined_text = " ".join(contents)
                    if combined_text:
                        cur.execute("""
                            UPDATE places 
                            SET description = %s, blog_metadata = %s::jsonb, is_enriched = TRUE 
                            WHERE id = %s
                        """, (combined_text, json.dumps(metadata_list, ensure_ascii=False), pid))
                        conn.commit()
                        
                        print(f"[DEBUG] Embedding and indexing {len(contents)} chunks for {name}", flush=True)
                        
                        # CPU 과부하(병목) 방지를 위해 실시간 처리 시에는 첫 번째 청크(최대 300자) 1개만 임베딩합니다.
                        # 전체 텍스트는 DB에 저장되어 RAG 요약에는 활용됩니다.
                        chunks = [combined_text[:300]] if combined_text else []
                        region_val = entities["location"][0] if entities["location"] else "자동수집"
                        for i, chunk in enumerate(chunks):
                            if len(chunk) < 10: continue
                            print(f"[DEBUG] Encoding chunk {i+1}/{len(chunks)}...", flush=True)
                            emb = get_embedding(chunk)
                            doc_body = {
                                "place_id": pid,
                                "name": name,
                                "region": region_val,
                                "category": category,
                                "chunk_index": i,
                                "text": chunk,
                                "embedding": emb.tolist()
                            }
                            print(f"[DEBUG] Indexing chunk {i+1}/{len(chunks)} to ES...", flush=True)
                            es.index(index="spotsync_chunks", document=doc_body)
                            print(f"[DEBUG] Chunk {i+1} done.", flush=True)
                
                es.indices.refresh(index="spotsync_chunks")
                conn.close()
                
                # Fallback: Update nearby_place_ids and ES filter body to include newly created place IDs
                if new_place_ids:
                    new_pids = [item[0] for item in new_place_ids]
                    for pid in new_pids:
                        if pid not in nearby_place_ids:
                            nearby_place_ids.append(pid)
                    if es_filters:
                        for f in es_filters:
                            if "terms" in f and "place_id" in f["terms"]:
                                f["terms"]["place_id"] = nearby_place_ids
                        body["knn"]["filter"] = es_filters
                        body["query"]["bool"]["filter"] = es_filters
                
                print("[FALLBACK] Indexing complete! Re-running ES query...")
                res = es.options(request_timeout=15).search(index="spotsync_chunks", body=body)
                hits = res['hits']['hits']
                print(f"DEBUG: Fallback hits = {len(hits)}", flush=True)
                if len(hits) > 0:
                    fallback_success = True
                
        except Exception as e:
            print(f"[FALLBACK] Error during fallback: {e}")
            pass
            
    # If it was a hallucination (gibberish query) and the fallback also found nothing, 
    # we should completely drop the garbage ES results to prevent them from scaling to 80 points.
    if is_hallucination and not fallback_success:
        print("DEBUG: Gibberish/Hallucination detected and fallback failed. Clearing hits.")
        hits = []

    from collections import defaultdict
    place_chunk_scores = defaultdict(list)
    for hit in hits:
        pid = hit['_source']['place_id']
        score = hit['_score']
        place_chunk_scores[pid].append(score)
        
    final_place_scores = {}
    for pid, score_list in place_chunk_scores.items():
        score_list.sort(reverse=True)
        take_n = max(1, min(2, len(score_list)))
        top_scores = score_list[:take_n]
        final_place_scores[pid] = sum(top_scores) / len(top_scores)
            
    conn = get_db_connection()
    cur = conn.cursor()

    # Normalize semantic scores to 0.0 ~ 0.5
    if final_place_scores:
        max_es_score = max(final_place_scores.values())
        if max_es_score == 0: max_es_score = 1.0
        for pid in final_place_scores:
            final_place_scores[pid] = (final_place_scores[pid] / max_es_score) * 0.5

    # --- SPOTSYNC RE-RANKING LOGIC ---
    midpoint = calculate_midpoint(req.user_locations)
    distances = {}
    if midpoint and final_place_scores:
        mid_lat, mid_lng = midpoint
        place_ids_tuple = tuple(final_place_scores.keys())
        if len(place_ids_tuple) == 1:
            place_ids_tuple_str = f"({place_ids_tuple[0]})"
        else:
            place_ids_tuple_str = str(place_ids_tuple)
            
        cur.execute(f"""
            SELECT id, ST_Distance(location::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) as dist
            FROM places
            WHERE id IN {place_ids_tuple_str}
        """, (mid_lng, mid_lat))
        
        for row in cur.fetchall():
            pid = row[0]
            dist = float(row[1])
            distances[pid] = dist
            
            # Re-calculate score: Combine normalized semantic score + distance penalty
            original_score = final_place_scores[pid]
            # Apply Distance Bonus (within 10km -> 0.5 max bonus)
            distance_bonus = 0
            if dist <= 10000:
                distance_bonus = max(0, 0.5 * (1 - (dist / 10000.0)))
            
            final_place_scores[pid] = original_score + distance_bonus
            print(f"[RE-RANK] PID: {pid}, Norm ES: {original_score:.3f}, Dist: {dist:.1f}m, Bonus: +{distance_bonus:.3f}, Final: {final_place_scores[pid]:.3f}", flush=True)

    # Cutoff Filter: Drop results with final score < 0.45 (45%)
    cutoff_threshold = 0.45
    filtered_places = {pid: score for pid, score in final_place_scores.items() if score >= cutoff_threshold}
    
    sorted_places = sorted(filtered_places.items(), key=lambda x: x[1], reverse=True)[:req.top_k]
    print(f"DEBUG: sorted_places after cutoff = {sorted_places}", flush=True)
    
    import math
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def get_kakao_travel_time(lat1, lon1, lat2, lon2):
        if not KAKAO_API_KEY: return None
        try:
            url = "https://apis-navi.kakaomobility.com/v1/directions"
            headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
            params = {"origin": f"{lon1},{lat1}", "destination": f"{lon2},{lat2}"}
            res = requests.get(url, headers=headers, params=params, timeout=2).json()
            if "routes" in res and len(res["routes"]) > 0:
                duration_sec = res["routes"][0]["summary"]["duration"]
                return int(duration_sec / 60)
        except Exception:
            pass
        return None

    results = []
    context_texts = []
    
    for pid, score in sorted_places:
        dist_m = distances.get(pid, -1)
        cur.execute("SELECT id, name, category, address, COALESCE(description, ''), latitude, longitude, COALESCE(blog_metadata, '[]'::jsonb) FROM places WHERE id = %s", (pid,))
        row = cur.fetchone()
        if row:
            db_id, name, category, address, desc, lat, lng, blog_metadata = row
            
            travel_times = []
            for u in req.user_locations:
                mins = get_kakao_travel_time(u.lat, u.lng, float(lat), float(lng))
                if mins is None:
                    u_dist = haversine(u.lat, u.lng, float(lat), float(lng))
                    routing_dist = u_dist * 1.4
                    mins = int(routing_dist / 416)
                    if mins < 1: mins = 1
                travel_times.append({"name": u.name, "minutes": mins})
                
            results.append({
                "id": db_id,
                "name": name,
                "category": category,
                "address": address,
                "score": score,
                "distance_to_midpoint_m": round(dist_m, 1) if dist_m != -1 else None,
                "latitude": float(lat) if lat else 0.0,
                "longitude": float(lng) if lng else 0.0,
                "blog_metadata": blog_metadata,
                "travel_times": travel_times
            })
            context_texts.append(f"DB ID: {db_id} | [{name}] 카테고리: {category}\n설명/리뷰: {str(desc)[:400]}")
            
    conn.close()
    async def event_generator():
        elapsed = time.time() - t_start
        yield f"data: {json.dumps({'type': 'results', 'results': results, 'elapsed_sec': round(elapsed, 2)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
