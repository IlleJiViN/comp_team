import time
import json
import requests

queries = [
    "홍대 올리브영",
    "망원동 분위기 좋은 카페",
    "조용한 치킨 합정동",
    "마포구 미용실 추천해줘",
    "다이소",
    "혼밥 백반/한정식 상수동 근처",
    "연남동 일식 회/초밥",
    "맥도날드 서교동",
    "성수 힙한 카페",
    "강남 헬스장"
]

def test_query(q):
    try:
        response = requests.post(
            "http://localhost:8001/search_rag", 
            json={"query": q, "top_k": 3},
            stream=True,
            timeout=15
        )
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    data = json.loads(decoded[6:])
                    if data.get("type") == "results":
                        latency = data.get("elapsed_sec")
                        results = data.get("results", [])
                        names = [r["name"] for r in results]
                        return latency, names
    except Exception as e:
        return -1, str(e)
    return -1, "No results"

results = []
print("Starting Performance Test...")
for q in queries:
    lat, names = test_query(q)
    results.append({"query": q, "latency": lat, "results": names})
    time.sleep(0.2)

print("\n" + "="*80)
print("🎯 SpotSync v8 (NER + RAG) Performance Report")
print("="*80)
total_lat = 0
success_cnt = 0
for r in results:
    lat_str = f"{r['latency']:.2f}s" if isinstance(r['latency'], float) else str(r['latency'])
    if isinstance(r['latency'], float) and r['latency'] > 0:
        total_lat += r['latency']
        success_cnt += 1
    
    hits = r['results'] if isinstance(r['results'], list) else [r['results']]
    print(f"[{lat_str:>5}] {r['query']:<25} ➔ {', '.join(hits)}")

if success_cnt > 0:
    print("-"*80)
    print(f"Average Search Latency: {total_lat/success_cnt:.2f}s")
print("="*80)
