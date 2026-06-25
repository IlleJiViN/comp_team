import json
import sys
from elasticsearch import Elasticsearch
sys.stdout.reconfigure(encoding='utf-8')

es = Elasticsearch('http://localhost:9200')

with open("C:/Users/dev/.gemini/antigravity-cli/brain/7f4d8ac9-3374-4252-94d3-a94776ac1853/.system_generated/tasks/task-5814.log", "r", encoding="utf-8") as f:
    log_data = f.read()
    start_idx = log_data.find("DEBUG ES QUERY BODY: ") + len("DEBUG ES QUERY BODY: ")
    end_idx = log_data.find("\n", start_idx)
    body_str = log_data[start_idx:end_idx]
    
try:
    body = json.loads(body_str)
    res = es.search(index='spotsync_chunks', body=body)
    hits = res['hits']['hits']
    print(f"Hits: {len(hits)}")
    for i, hit in enumerate(hits[:5]):
        print(f"Hit {i}: Score={hit['_score']}, place_id={hit['_source'].get('place_id')}, name={hit['_source'].get('name')}")
except Exception as e:
    print(f"Error: {e}")
