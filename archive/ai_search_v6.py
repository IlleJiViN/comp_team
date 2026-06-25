from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import time
import psycopg2
import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="SpotSync AI Search V6 (RAG Streaming)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Loading BAAI/bge-m3 model...")
model = SentenceTransformer('BAAI/bge-m3', device='cpu')

print("[INFO] Loading Gemini LLM via LangChain...")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=GOOGLE_API_KEY, streaming=True)

print("[INFO] Loading Mapo-gu Embeddings (V7 Chunked)...")
try:
    data = np.load("data/embeddings_v7_chunked.npz", allow_pickle=True)
    place_ids = data['ids']
    embeddings = data['embeddings']
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1e-10, norms)
    print(f"[SUCCESS] Loaded {len(place_ids)} embeddings")
except Exception as e:
    print(f"[WARNING] Could not load embeddings: {e}")
    place_ids = np.array([])
    embeddings = np.array([])

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search_rag")
async def search_rag(req: SearchQuery):
    if len(embeddings) == 0:
        return {"error": "Embeddings not loaded."}

    t_start = time.time()
    
    query_emb = model.encode(req.query, convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(embeddings, query_emb)
    from collections import defaultdict
    place_chunk_scores = defaultdict(list)
    for i, pid in enumerate(place_ids):
        place_chunk_scores[int(pid)].append(float(scores[i]))
        
    final_place_scores = {}
    for pid, score_list in place_chunk_scores.items():
        score_list.sort(reverse=True)
        # 상위 20% 의 평균 (최소 1개)
        take_n = max(1, int(len(score_list) * 0.2))
        top_scores = score_list[:take_n]
        final_place_scores[pid] = sum(top_scores) / len(top_scores)
            
    # Sort unique places by their aggregated score
    sorted_places = sorted(final_place_scores.items(), key=lambda x: x[1], reverse=True)[:req.top_k]
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    results = []
    context_texts = []
    
    for pid, score in sorted_places:
        cur.execute("SELECT id, name, category, address, COALESCE(description, ''), latitude, longitude FROM places WHERE id = %s", (pid,))
        row = cur.fetchone()
        if row:
            db_id, name, category, address, desc, lat, lng = row
            results.append({
                "id": db_id,
                "name": name,
                "category": category,
                "address": address,
                "score": score,
                "latitude": float(lat) if lat else 0.0,
                "longitude": float(lng) if lng else 0.0
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
        # Send initial results
        yield f"data: {json.dumps({'type': 'results', 'results': results, 'elapsed_sec': round(elapsed, 2)})}\n\n"
        
        # Stream LLM chunks
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
