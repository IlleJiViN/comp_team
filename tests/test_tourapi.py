import urllib.request
import urllib.parse
import json

API_KEY = "42507167655a46953cb66781879c893e9097ab815b81c6227315dab4818c6ad9"
URL = "https://apis.data.go.kr/B551011/KorService2/areaBasedSyncList2"

# MobileOS: ETC, MobileApp: AppTest, _type: json
url = f"{URL}?serviceKey={API_KEY}&numOfRows=10&pageNo=1&MobileOS=ETC&MobileApp=AppTest&_type=json&showflag=1"

try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        body = response.read().decode('utf-8')
        try:
            print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
        except:
            print(body)
except Exception as e:
    print("Error:", e)
