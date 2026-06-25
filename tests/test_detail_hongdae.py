import urllib.request
import urllib.parse
import json

API_KEY = "42507167655a46953cb66781879c893e9097ab815b81c6227315dab4818c6ad9"
BASE_URL = "https://apis.data.go.kr/B551011/KorService2"

def get_hongdae_shop():
    # areaCode=1 (서울), sigunguCode=13 (마포구)
    search_url = f"{BASE_URL}/areaBasedList2?serviceKey={API_KEY}&MobileOS=ETC&MobileApp=AppTest&_type=json&contentTypeId=39&areaCode=1&sigunguCode=13&numOfRows=5&pageNo=1"
    
    try:
        req = urllib.request.Request(search_url)
        with urllib.request.urlopen(req) as response:
            body = response.read().decode('utf-8')
            data = json.loads(body)
            items = data.get('response', {}).get('body', {}).get('items', "")
            
            if not items or items == "":
                print("마포구 음식점 검색 결과가 없습니다.")
                return
                
            item_list = items.get('item', [])
            if isinstance(item_list, dict):
                item_list = [item_list]

            for item in item_list:
                content_id = item['contentid']
                title = item['title']
                addr = item.get('addr1', '')
                
                # Fetch details
                detail_url = f"{BASE_URL}/detailCommon2?serviceKey={API_KEY}&MobileOS=ETC&MobileApp=AppTest&_type=json&contentId={content_id}"
                detail_req = urllib.request.Request(detail_url)
                with urllib.request.urlopen(detail_req) as d_response:
                    d_body = d_response.read().decode('utf-8')
                    d_data = json.loads(d_body)
                    d_items = d_data.get('response', {}).get('body', {}).get('items', "")
                    if not d_items or d_items == "":
                        continue
                    d_item_list = d_items.get('item', [])
                    if isinstance(d_item_list, list):
                        d_item = d_item_list[0]
                    else:
                        d_item = d_item_list
                    
                    print(f"[{title}]")
                    print(f"- 주소: {addr}")
                    print(f"- 개요:\n{d_item.get('overview', '개요 없음')}")
                    print("-" * 50)
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

get_hongdae_shop()
