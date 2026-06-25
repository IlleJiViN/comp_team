import urllib.request
import json
import time
from sqlalchemy import create_engine, text
import math

API_KEY = "42507167655a46953cb66781879c893e9097ab815b81c6227315dab4818c6ad9"
BASE_URL = "https://apis.data.go.kr/B551011/KorService2"
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def fetch_all_kto():
    engine = create_engine(DATABASE_URL)
    
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS description TEXT;"))
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS is_premium BOOLEAN DEFAULT FALSE;"))
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_vector_v6 float8[];"))
    
    print("[1] Fetching TourAPI Food Places (contentTypeId=39)...")
    content_types = [39, 12] # Food, Attractions
    
    tour_places = {} # name -> contentid
    
    for c_type in content_types:
        try:
            # Get total count
            u = f"{BASE_URL}/areaBasedList2?serviceKey={API_KEY}&MobileOS=ETC&MobileApp=AppTest&_type=json&contentTypeId={c_type}&numOfRows=1&pageNo=1"
            req = urllib.request.Request(u)
            with urllib.request.urlopen(req) as res:
                data = json.loads(res.read().decode('utf-8'))
                total_count = data['response']['body']['totalCount']
                
            print(f"Content Type {c_type} has {total_count} items. Fetching...")
            
            num_of_rows = 5000
            pages = math.ceil(total_count / num_of_rows)
            for page in range(1, pages + 1):
                page_url = f"{BASE_URL}/areaBasedList2?serviceKey={API_KEY}&MobileOS=ETC&MobileApp=AppTest&_type=json&contentTypeId={c_type}&numOfRows={num_of_rows}&pageNo={page}"
                req = urllib.request.Request(page_url)
                with urllib.request.urlopen(req) as res:
                    page_data = json.loads(res.read().decode('utf-8'))
                    items = page_data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                    for item in items:
                        name_key = str(item['title']).replace(" ", "").strip()
                        tour_places[name_key] = item['contentid']
        except Exception as e:
            print(f"Error fetching list for type {c_type}:", e)

    print(f"Total unique TourAPI places fetched: {len(tour_places)}")
    
    print("[2] Matching with local PostgreSQL 'places' table...")
    matched_ids = []
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, name FROM places")).fetchall()
        for row in result:
            db_id = row[0]
            db_name = str(row[1]).replace(" ", "").strip()
            if db_name in tour_places:
                matched_ids.append((db_id, tour_places[db_name], row[1]))
                
    print(f"Total local DB matches: {len(matched_ids)}")
    
    print("[3] Fetching overview for matched places and updating DB...")
    success_count = 0
    for db_id, content_id, original_name in matched_ids:
        detail_url = f"{BASE_URL}/detailCommon2?serviceKey={API_KEY}&MobileOS=ETC&MobileApp=AppTest&_type=json&contentId={content_id}&overviewYN=Y"
        try:
            req = urllib.request.Request(detail_url)
            with urllib.request.urlopen(req) as res:
                body = res.read().decode('utf-8')
                data = json.loads(body)
                items = data.get('response', {}).get('body', {}).get('items', "")
                if not items or items == "":
                    continue
                item_list = items.get('item', [])
                if isinstance(item_list, dict):
                    item_list = [item_list]
                
                if item_list:
                    overview = item_list[0].get('overview', '')
                    if overview and len(overview) > 10:
                        with engine.begin() as conn:
                            conn.execute(
                                text("UPDATE places SET description = :desc, is_premium = TRUE WHERE id = :id"),
                                {"desc": overview, "id": db_id}
                            )
                        success_count += 1
                        if success_count % 100 == 0:
                            print(f"Updated {success_count}/{len(matched_ids)} premium places...")
                time.sleep(0.1)  # Add rate limit to avoid 429 Too Many Requests
        except Exception as e:
            time.sleep(0.1)
            pass
            
    print(f"[DONE] Successfully updated {success_count} premium places in DB!")

if __name__ == "__main__":
    fetch_all_kto()
