import requests
import json
import sys

r = requests.post('http://127.0.0.1:8001/search_rag', json={'query': '부산역 근처 조용한 국밥집'})
print(f"Status Code: {r.status_code}")
print(f"Response: {r.text[:500]}")
