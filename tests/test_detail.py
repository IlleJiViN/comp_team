import urllib.request
import urllib.parse
import json

API_KEY = "42507167655a46953cb66781879c893e9097ab815b81c6227315dab4818c6ad9"
URL = "https://apis.data.go.kr/B551011/KorService2/detailCommon2"

def get_detail(content_id):
    url = f"{URL}?serviceKey={API_KEY}&MobileOS=ETC&MobileApp=AppTest&_type=json&contentId={content_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            body = response.read().decode('utf-8')
            try:
                data = json.loads(body)
                if 'response' not in data:
                    print(f"[{content_id}] Raw data:", data)
                    return
                item = data['response']['body']['items']['item'][0]
                print(f"[{item.get('title', '제목없음')}]")
                print(f"- 개요(Overview):\n{item.get('overview', '개요 없음')}")
                print("-" * 50)
            except Exception as e:
                print("Body:", body)
    except Exception as e:
        print(f"Error for {content_id}:", e)

get_detail("1433504")
get_detail("127480")
