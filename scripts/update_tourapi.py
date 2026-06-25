import urllib.request
import urllib.parse
import json
import time
from sqlalchemy import create_engine, text

API_KEY = "42507167655a46953cb66781879c893e9097ab815b81c6227315dab4818c6ad9"
BASE_URL = "https://apis.data.go.kr/B551011/KorService2"
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def update_descriptions():
    engine = create_engine(DATABASE_URL)
    
    # 1. DB에 description 컬럼 추가
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS description TEXT;"))
    
    # 2. 관광공사 마포구(areaCode=1, sigunguCode=13) 음식점(39) 및 관광지 데이터 1000개 수집
    print("관광공사 마포구 상권 데이터 수집 중...")
    search_url = f"{BASE_URL}/areaBasedList2?serviceKey={API_KEY}&MobileOS=ETC&MobileApp=AppTest&_type=json&areaCode=1&sigunguCode=13&numOfRows=1000&pageNo=1"
    
    tour_places = {}
    try:
        req = urllib.request.Request(search_url)
        with urllib.request.urlopen(req) as response:
            body = response.read().decode('utf-8')
            data = json.loads(body)
            items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
            
            for item in items:
                # 상호명 공백 제거해서 매칭 확률 높임
                name_key = str(item['title']).replace(" ", "").strip()
                tour_places[name_key] = item['contentid']
    except Exception as e:
        print("API Error:", e)
        return

    print(f"총 {len(tour_places)}개의 관광공사 데이터를 가져왔습니다.")
    
    # 3. 로컬 소상공인 DB(places)에서 마포구 상가 불러오기
    print("로컬 소상공인 DB에서 마포구 식당 조회 중...")
    matched_ids = []
    with engine.connect() as conn:
        # 마포구 주소가 포함된 상가
        result = conn.execute(text("SELECT id, name FROM places WHERE address LIKE '%마포구%'")).fetchall()
        
        for row in result:
            db_id = row[0]
            db_name = str(row[1]).replace(" ", "").strip()
            
            if db_name in tour_places:
                matched_ids.append((db_id, tour_places[db_name], row[1]))
                
    print(f"로컬 소상공인 DB 상호명과 매칭된 식당: {len(matched_ids)}곳!")
    
    # 4. 매칭된 곳의 상세 설명(overview) 가져와서 DB에 업데이트
    print("매칭된 식당들의 상세 설명(Overview) 다운로드 및 DB 업데이트 시작...")
    success_count = 0
    for db_id, content_id, original_name in matched_ids:
        detail_url = f"{BASE_URL}/detailCommon2?serviceKey={API_KEY}&MobileOS=ETC&MobileApp=AppTest&_type=json&contentId={content_id}"
        try:
            d_req = urllib.request.Request(detail_url)
            with urllib.request.urlopen(d_req) as d_response:
                d_body = d_response.read().decode('utf-8')
                d_data = json.loads(d_body)
                d_items = d_data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                if d_items:
                    overview = d_items[0].get('overview', '')
                    if overview and len(overview) > 10:
                        with engine.begin() as conn:
                            conn.execute(
                                text("UPDATE places SET description = :desc WHERE id = :id"),
                                {"desc": overview, "id": db_id}
                            )
                        success_count += 1
                        if success_count <= 3:
                            print(f"[업데이트 완료] {original_name} -> {overview[:50]}...")
            time.sleep(0.1) # API 호출 제한 방지
        except Exception as e:
            continue
            
    print(f"작업 완료! 총 {success_count}곳의 소상공인 데이터에 관광공사 상세 설명이 결합되었습니다.")

if __name__ == "__main__":
    update_descriptions()
