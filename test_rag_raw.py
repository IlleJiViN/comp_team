import os
import psycopg2
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def main():
    print("Loading BGE-M3 model...")
    embedding_model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    
    print("Loading DB...")
    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/spotsync")
    cur = conn.cursor()
    
    print("Loading embeddings_v6.npz...")
    data = np.load("data/embeddings_v6.npz")
    place_ids = data['ids']
    embeddings = data['embeddings']
    
    query = "비 오는 날 어울리는 홍대/합정 식당 추천해 줘"
    print(f"\n[유저 질문]: {query}\n")
    
    # 1. Search DB
    query_emb = embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(embeddings, query_emb)
    top_indices = np.argsort(scores)[::-1][:3] # Top 3
    
    context_texts = []
    for idx in top_indices:
        pid = int(place_ids[idx])
        cur.execute("SELECT name, category, description FROM places WHERE id = %s", (pid,))
        row = cur.fetchone()
        if row:
            name, cat, desc = row
            context_texts.append(f"[{name}] 카테고리: {cat}\n설명 및 리뷰: {str(desc)[:400]}...")
            
    context_str = "\n\n".join(context_texts)
    
    # 2. Call Gemini
    prompt = f"""당신은 친절한 맛집 추천 가이드입니다. 
아래 검색된 장소 정보를 바탕으로 유저의 질문에 대답해주세요.
블로그 리뷰 내용이 있다면 참고해서 매력적으로 추천해주세요.

[검색된 장소 정보]
{context_str}

유저 질문: {query}
"""
    
    print("답변 생성 중...\n")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7}
    }
    
    res = requests.post(url, json=payload)
    if res.status_code == 200:
        data = res.json()
        text = data['candidates'][0]['content']['parts'][0]['text']
        print("=== 최종 AI 답변 ===")
        print(text)
    else:
        print(f"Error: {res.status_code}")
        print(res.text)

if __name__ == "__main__":
    main()
