# -*- coding: utf-8 -*-
import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

payload = {
    'query': '조용하고 분위기 좋은 맥도날드',
    'user_latitude': 37.5562,
    'user_longitude': 126.9371,
    'radius_meters': 3000.0,
    'category_threshold': 0.15,
    'similarity_threshold': 0.0
}

res = requests.post('http://127.0.0.1:8000/search', json=payload)
print(json.dumps(res.json(), ensure_ascii=False, indent=2))
