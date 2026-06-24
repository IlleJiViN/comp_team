import time
import requests
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("Waiting for server...")
for _ in range(30):
    try:
        requests.get("http://localhost:8001", timeout=1)
        break
    except:
        time.sleep(1)

print("Server is up! Sending request...")
url = "http://localhost:8001/search_rag"
headers = {"Content-Type": "application/json"}
data = {
    "query": "부산역 근처 조용한 국밥집",
    "top_k": 3
}

response = requests.post(url, headers=headers, json=data, stream=True)
print(f"Status Code: {response.status_code}")
for line in response.iter_lines():
    if line:
        decoded_line = line.decode('utf-8')
        print(decoded_line)
        try:
            js = json.loads(decoded_line.replace("data: ", ""))
            if js["type"] == "results":
                print(f"Found {len(js['results'])} results!")
        except:
            pass
