import sys
import requests
import json
import time

# Configure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

url = "http://localhost:8001/search_rag"

# Let's say 3 friends are meeting:
# Person 1: 홍대입구역 근처 (37.5568, 126.9242)
# Person 2: 신촌역 근처 (37.5552, 126.9368)
# Person 3: 합정역 근처 (37.5499, 126.9142)
# Midpoint should be somewhere in the middle (e.g. 홍대/서교동)

payload = {
    "query": "노트북 작업하기 좋은 조용한 카페",
    "top_k": 5,
    "user_locations": [
        {"name": "철수", "lat": 37.5568, "lng": 126.9242},
        {"name": "영희", "lat": 37.5552, "lng": 126.9368},
        {"name": "민수", "lat": 37.5499, "lng": 126.9142}
    ]
}

print("Testing v9 Search API with multiple user locations...")
try:
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    data = json.loads(data_str)
                    if data.get("type") == "results":
                        results = data.get("results", [])
                        print(f"\n--- AI TOP {len(results)} RECOMMENDATIONS ---")
                        for idx, res in enumerate(results):
                            dist_str = f"{res.get('distance_to_midpoint_m')}m" if res.get('distance_to_midpoint_m') is not None else "N/A"
                            print(f"{idx+1}. {res['name']} (Score: {res['score']:.3f})")
                            print(f"   카테고리: {res['category']}")
                            print(f"   거리(중간 지점으로부터): {dist_str}")
                            print(f"   주소: {res['address']}")
                            print("-" * 40)
                    elif data.get("type") == "chunk":
                        # Print LLM reasoning stream
                        print(data.get("text", ""), end="", flush=True)
                    elif data.get("type") == "done":
                        print("\n\n[Done]")
except Exception as e:
    print(f"Error: {e}")
