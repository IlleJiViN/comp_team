import os
import json
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from google.cloud import discoveryengine_v1beta as discoveryengine

# ==============================================================================
# Google Vertex AI Search (자체 RAG) 서버
# 포트: 8001
# ==============================================================================

app = FastAPI(title="SpotSync Vertex AI RAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID = "88052320203"
LOCATION = "global"
ENGINE_ID = "spotsync-search-engine"

# Vertex AI Search Client 초기화
search_client = discoveryengine.SearchServiceClient()
serving_config = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/engines/{ENGINE_ID}/servingConfigs/default_search"

class Location(BaseModel):
    name: Optional[str] = "익명"
    lat: float
    lng: float

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3
    user_locations: List[Location] = []

@app.post("/search_rag")
async def search_rag(req: SearchQuery):
    """
    Google Vertex AI Search의 자체 RAG (Summary) 기능을 사용하여 검색 결과를 반환합니다.
    """
    t_start = time.time()
    
    async def event_generator():
        try:
            # 1. Vertex AI Search RAG 요청 생성
            request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=req.query,
                page_size=req.top_k,
                content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                    extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                        max_extractive_answer_count=1
                    ),
                    summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                        summary_result_count=req.top_k,
                        include_citations=True,
                        ignore_adversarial_query=True,
                        ignore_non_summary_seeking_query=True
                    )
                )
            )
            
            # 2. 구글 API 호출
            response = search_client.search(request)
            
            # 3. 검색된 장소 결과 파싱
            results = []
            for r in response.results:
                doc = r.document
                # struct_data의 필드를 dict로 변환
                sd = {}
                for key, val in doc.struct_data.fields.items():
                    kind = val.WhichOneof('kind')
                    if kind == 'string_value':   sd[key] = val.string_value
                    elif kind == 'number_value': sd[key] = val.number_value
                    elif kind == 'bool_value':   sd[key] = val.bool_value
                    else: sd[key] = str(val)

                results.append({
                    "id": int(sd.get("id", 0)) or doc.id,
                    "name": sd.get("title", "이름없음"),
                    "category": sd.get("category", ""),
                    "address": sd.get("address", ""),
                    "latitude": float(sd.get("latitude", 0.0)),
                    "longitude": float(sd.get("longitude", 0.0)),
                    "score": 1.0,
                    "distance_to_midpoint_m": None,
                    "blog_metadata": [],
                    "travel_times": []
                })
                
            elapsed = time.time() - t_start
            yield f"data: {json.dumps({'type': 'results', 'results': results, 'elapsed_sec': round(elapsed, 2)})}\n\n"
            
            # 4. RAG 생성 결과 (Summary) 스트리밍 반환
            if response.summary and response.summary.summary_text:
                summary_text = response.summary.summary_text
                # Streaming effect 
                chunk_size = 20
                for i in range(0, len(summary_text), chunk_size):
                    chunk = summary_text[i:i+chunk_size]
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                    time.sleep(0.05)
            else:
                yield f"data: {json.dumps({'type': 'chunk', 'text': '구글 Vertex AI가 생성한 요약이 없습니다.'})}\n\n"
                
        except Exception as e:
            error_msg = str(e)
            print(f"Vertex AI Search Error: {error_msg}", flush=True)
            if "enterprise edition" in error_msg.lower():
                yield f"data: {json.dumps({'type': 'chunk', 'text': '\\n\\n[시스템 안내: 구글 클라우드 콘솔에서 Enterprise Edition 업그레이드가 필요합니다.]'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'chunk', 'text': f'\\n\\n[검색 오류: {error_msg[:100]}...]'})}\n\n"
                
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Vertex AI RAG 전용 서버 포트는 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
