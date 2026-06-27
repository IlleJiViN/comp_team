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
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="SpotSync AI Search V8 (NER + RAG Streaming)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Loading BAAI/bge-m3 model...")
model = SentenceTransformer('BAAI/bge-m3', device='cpu')

print("[INFO] Loading SpotSync NER model...")
ner_model_path = "./models/spotsync-ner"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
with open(os.path.join(ner_model_path, "label_config.json"), "r", encoding="utf-8") as f:
    label_config = json.load(f)
label_list = label_config["label_list"]

print("[INFO] Loading Gemini LLM via LangChain...")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=GOOGLE_API_KEY, streaming=True)

from elasticsearch import Elasticsearch

print("[INFO] Connecting to Elasticsearch...")
es = Elasticsearch("http://localhost:9200", request_timeout=10)

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

class Location(BaseModel):
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
    if not tokens:
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
    
    query_emb = model.encode(req.query, convert_to_numpy=True, normalize_embeddings=True)
    
    # Filter by Category
    es_filters = extract_filters(entities)
    
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
            "boost": 0.6  # Semantic Weight
        },
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 0
            }
        },
        "size": 50
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
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        try:
            local_res = requests.get(local_url, headers=headers, params={"query": kakao_query, "size": 3}).json()
            docs = local_res.get("documents", [])
            
            if docs:
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
                
                blog_url = "https://dapi.kakao.com/v2/search/blog"
                
                for pid, name, addr in new_place_ids:
                    blog_res = requests.get(blog_url, headers=headers, params={"query": f"{addr} {name}", "size": 3}).json()
                    b_docs = blog_res.get("documents", [])
                    contents = []
                    metadata_list = []
                    for b_doc in b_docs:
                        text = b_doc.get("contents", "").replace("<b>", "").replace("</b>", "").strip()
                        if text: 
                            contents.append(text)
                            metadata_list.append({
                                "source": "kakao",
                                "title": b_doc.get("title", "").replace("<b>", "").replace("</b>", "").strip(),
                                "url": b_doc.get("url", ""),
                                "postdate": b_doc.get("datetime", "")[:10],
                                "bloggername": b_doc.get("blogname", ""),
                                "thumbnail": b_doc.get("thumbnail", "")
                            })
                    
                    combined_text = " ".join(contents)
                    if combined_text:
                        cur.execute("""
                            UPDATE places 
                            SET description = %s, blog_metadata = %s::jsonb, is_enriched = TRUE 
                            WHERE id = %s
                        """, (combined_text, json.dumps(metadata_list, ensure_ascii=False), pid))
                        conn.commit()
                        
                        chunks = [combined_text[i:i+300] for i in range(0, len(combined_text), 300)]
                        region_val = entities["location"][0] if entities["location"] else "자동수집"
                        for i, chunk in enumerate(chunks):
                            if len(chunk) < 10: continue
                            emb = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
                            doc_body = {
                                "place_id": pid,
                                "name": name,
                                "region": region_val,
                                "category": category,
                                "chunk_index": i,
                                "text": chunk,
                                "embedding": emb.tolist()
                            }
                            es.index(index="spotsync_chunks", document=doc_body)
                
                es.indices.refresh(index="spotsync_chunks")
                conn.close()
                
                print("[FALLBACK] Indexing complete! Re-running ES query...")
                res = es.options(request_timeout=15).search(index="spotsync_chunks", body=body)
                hits = res['hits']['hits']
                print(f"DEBUG: Fallback hits = {len(hits)}", flush=True)
                
        except Exception as e:
            print(f"[FALLBACK] Error during fallback: {e}")
            pass
    
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
            
    # Normalize semantic scores to 0.0 ~ 0.5
    if final_place_scores:
        max_es_score = max(final_place_scores.values())
        if max_es_score == 0: max_es_score = 1.0
        for pid in final_place_scores:
            final_place_scores[pid] = (final_place_scores[pid] / max_es_score) * 0.5

    # Distance Re-Ranking
    midpoint = calculate_midpoint(req.user_locations)
    distances = {}
    if midpoint and final_place_scores:
        mid_lat, mid_lng = midpoint
        place_ids_tuple = tuple(final_place_scores.keys())
        if len(place_ids_tuple) == 1:
            place_ids_tuple_str = f"({place_ids_tuple[0]})"
        else:
            place_ids_tuple_str = str(place_ids_tuple)
            
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT id, ST_Distance(location::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) as dist
            FROM places
            WHERE id IN {place_ids_tuple_str}
        """, (mid_lng, mid_lat))
        
        for row in cur.fetchall():
            pid = row[0]
            dist = float(row[1])
            distances[pid] = dist
            
            original_score = final_place_scores[pid]
            distance_bonus = 0
            if dist <= 10000:
                distance_bonus = max(0, 0.5 * (1 - (dist / 10000.0)))
            
            final_place_scores[pid] = original_score + distance_bonus
        
        cur.close()
        conn.close()
            
    sorted_places = sorted(final_place_scores.items(), key=lambda x: x[1], reverse=True)[:req.top_k]
    print(f"DEBUG: final_place_scores = {final_place_scores}, sorted_places = {sorted_places}", flush=True)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    results = []
    context_texts = []
    
    for pid, score in sorted_places:
        cur.execute("SELECT id, name, category, address, COALESCE(description, ''), latitude, longitude, COALESCE(blog_metadata, '[]'::jsonb) FROM places WHERE id = %s", (pid,))
        row = cur.fetchone()
        if row:
            db_id, name, category, address, desc, lat, lng, blog_metadata = row
            results.append({
                "id": db_id,
                "name": name,
                "category": category,
                "address": address,
                "score": score,
                "latitude": float(lat) if lat else 0.0,
                "longitude": float(lng) if lng else 0.0,
                "blog_metadata": blog_metadata
            })
            context_texts.append(f"[{name}] 카테고리: {category}\n설명/리뷰: {str(desc)[:500]}")
            
    conn.close()
    context_str = "\n\n".join(context_texts)
    
    template = """당신은 유저의 취향을 저격하는 전문적이고 친절한 맛집 가이드 AI 'SpotSync'입니다.
아래 검색된 장소 정보(블로그 리뷰 포함)를 바탕으로 유저의 질문에 답변해주세요.
장소들의 특징(분위기, 추천 메뉴 등)을 리뷰에서 추출하여 매력적으로 소개해주세요.
인사말과 함께 이모지를 적극적으로 사용하여 예쁘게 꾸며주세요!
검색 결과가 질문과 맞지 않다면 솔직하게 말해주세요.

[검색된 장소 정보]
{context}

유저 질문: {query}
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    chain = prompt | llm
    
    async def event_generator():
        elapsed = time.time() - t_start
        yield f"data: {json.dumps({'type': 'results', 'results': results, 'elapsed_sec': round(elapsed, 2)})}\n\n"
        
        async for chunk in chain.astream({"context": context_str, "query": req.query}):
            if chunk.content:
                text_chunk = chunk.content
                if isinstance(text_chunk, list):
                    text_chunk = "".join([part.get("text", "") for part in text_chunk if isinstance(part, dict)])
                elif not isinstance(text_chunk, str):
                    text_chunk = str(text_chunk)
                    
                yield f"data: {json.dumps({'type': 'chunk', 'text': text_chunk})}\n\n"
                
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
