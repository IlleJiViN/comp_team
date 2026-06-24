import os
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def main():
    print("Loading LangChain and LLM...")
    try:
        # Use the gemini-3.5-flash model we saw in the API list
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=GOOGLE_API_KEY)
        
        print("Loading BGE-M3 model...")
        embedding_model = SentenceTransformer('BAAI/bge-m3', device='cpu')
        
        print("Loading DB...")
        conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/spotsync")
        cur = conn.cursor()
        
        print("Loading embeddings_v6.npz...")
        data = np.load("data/embeddings_v6.npz", allow_pickle=True)
        place_ids = data['ids']
        embeddings = data['embeddings']
        
        import sys
        query = sys.argv[1] if len(sys.argv) > 1 else "비 오는 날 어울리는 홍대/합정 식당 추천해 줘"
        print(f"Query: {query}")
        
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
        print("\n--- 검색된 문맥(Context) ---")
        print(context_str)
        print("---------------------------\n")
        
        # 2. Prompt Template
        template = """당신은 친절한 맛집 추천 가이드입니다. 
아래 검색된 추천 장소 3곳의 정보를 바탕으로 유저의 질문에 자연스럽게 대답해주세요.
블로그 리뷰 내용이 있다면 참고해서 매력적으로 추천해주세요.
검색 결과에 적합한 장소가 없다면 솔직하게 말해주세요.

[검색된 장소 정보]
{context}

유저 질문: {query}
"""
        prompt = PromptTemplate(template=template, input_variables=["context", "query"])
        
        # 3. Chain
        chain = prompt | llm
        
        print("답변 생성 중...\n")
        response = chain.invoke({"context": context_str, "query": query})
        
        print("=== 최종 AI 답변 ===")
        print(response.content)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
