import os
import sqlite3
import requests
import numpy as np
import faiss
import torch
import time
from sentence_transformers import SentenceTransformer

# Persistency Configurations
DB_PATH = "spotsync.db"
INDEX_PATH = "spotsync.index"
MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"
DIMENSION = 768

# 경기도 주요 중심 행정구역 좌표 (크롤링 그리드 타겟)
GYEONGGI_REGIONS = {
    "수원 (Suwon)": {"lat": 37.26357, "lon": 127.0286},
    "성남 분당 (Bundang)": {"lat": 37.38269, "lon": 127.1189},
    "고양 일산 (Ilsan)": {"lat": 37.65836, "lon": 126.7701},
    "용인 (Yongin)": {"lat": 37.24108, "lon": 127.1774},
    "부천 (Bucheon)": {"lat": 37.50341, "lon": 126.7660},
    "안산 (Ansan)": {"lat": 37.32187, "lon": 126.8308},
    "남양주 (Namyangju)": {"lat": 37.63600, "lon": 127.2165},
    "안양 (Anyang)": {"lat": 37.39428, "lon": 126.9568},
    "평택 (Pyeongtaek)": {"lat": 36.99216, "lon": 127.1128},
    "의정부 (Uijeongbu)": {"lat": 37.73809, "lon": 127.0337}
}

def init_db():
    """Initializes SQLite database with indexes for faster coordinate-radius queries."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS places (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            address TEXT,
            description TEXT NOT NULL
        )
    """)
    # Add index on latitude and longitude to optimize 1차 반경 필터링 속도
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_coords ON places (latitude, longitude)")
    conn.commit()
    conn.close()
    print("[DB] SQLite database initialized with spatial indexes.")
def get_enriched_description(name: str, category: str, address: str) -> str:
    """Enriches place descriptions with custom semantic keywords based on category and name to boost matching accuracy."""
    desc = f"{name}은(는) {category}에 해당하며, 주소는 {address}입니다."
    
    lower_name = name.lower()
    lower_cat = category.lower()
    
    enrichments = []
    
    # Cafe, coffee, dessert, bakery
    if any(k in lower_cat or k in lower_name for k in ["카페", "커피", "디저트", "베이커리", "제과", "다방", "cafe", "coffee", "바리스타", "찻집"]):
        enrichments.append("노트북 들고 공부하기 편한 조용하고 편안한 카페, 아늑하고 콘센트가 많은 작업 공간, 분위기 좋은 디저트와 맛있는 커피가 있는 베이커리 맛집 공간입니다.")
        
    # PC Room, gaming
    elif any(k in lower_cat or k in lower_name for k in ["pc방", "피씨방", "pc", "게임", "gaming"]):
        enrichments.append("컴퓨터 그래픽카드 최고 사양 게이밍 모니터 넓은 피씨방, 초고속 인터넷, FPS 게임과 다양한 온라인 게임을 즐기기 좋은 프리미엄 게이밍 공간입니다.")
        
    # Music studio, practice room, band rehearsal room
    elif any(k in lower_cat or k in lower_name for k in ["합주실", "연습실", "음악실", "스튜디오", "녹음", "studio", "밴드", "보컬", "악기", "드럼", "마이크", "피아노"]):
        enrichments.append("드럼이랑 마이크 성능 좋은 방음 잘되는 음악 합주실, 보컬 밴드 노래 연습 개인 녹음 스튜디오, 최상의 방음 시설과 전문가용 마이크 및 악기가 완비된 연습실 공간입니다.")
        
    # Coin karaoke, singing room
    elif any(k in lower_cat or k in lower_name for k in ["노래방", "코인", "노래연습장", "노래", "코노"]):
        enrichments.append("혼자 가서 보컬 연습하고 마이크 녹음하기 조용한 스튜디오, 음질 좋은 최신 반주기와 무선 마이크, 화려한 LED 조명을 갖춘 코인 노래연습장입니다.")
        
    # Restaurant, eating
    elif any(k in lower_cat or k in lower_name for k in ["식당", "음식점", "밥집", "맛집", "한식", "중식", "일식", "양식", "분식", "고기", "구이", "치킨", "피자", "파스타", "국밥"]):
        enrichments.append("맛있고 위생적이며 친절한 식당, 다양한 메뉴를 제공하여 가족 외식, 친구들과의 모임 및 데이트 코스로 추천하는 맛집 공간입니다.")
        
    # Pub, bar, alcohol
    elif any(k in lower_cat or k in lower_name for k in ["술집", "호프", "맥주", "포차", "이자카야", "바(bar)", "펍", "bar", "pub"]):
        enrichments.append("안주가 맛있고 분위기 좋은 감성 술집, 시원한 맥주나 하이볼, 칵테일 한잔하며 이야기 나누기 좋은 모임 공간입니다.")
        
    if enrichments:
        desc += " " + " ".join(enrichments)
        
    return desc

def fetch_kakao_places(api_key: str, keyword: str, lat: float, lon: float, radius: int = 2000):
    """
    Priority 1: Fetches place data from Kakao Local API.
    """
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {
        "query": keyword,
        "y": str(lat),
        "x": str(lon),
        "radius": radius,
        "size": 15
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=8)
        if response.status_code == 401:
            print("[API WARNING] Kakao API Unauthorized (Invalid Key).")
            return None
        response.raise_for_status()
        data = response.json()
        
        raw_places = data.get("documents", [])
        formatted_places = []
        
        for p in raw_places:
            name = p.get("place_name")
            category_group = p.get("category_group_name") or p.get("category_name", "").split(">")[0].strip()
            latitude = float(p.get("y"))
            longitude = float(p.get("x"))
            address = p.get("road_address_name") or p.get("address_name")
            
            description = get_enriched_description(name, category_group, address)
            
            formatted_places.append({
                "name": name,
                "category": category_group,
                "latitude": latitude,
                "longitude": longitude,
                "address": address,
                "description": description
            })
        return formatted_places
    except Exception as e:
        print(f"[API WARNING] Kakao API call failed: {str(e)}")
        return None

def fetch_naver_places(client_id: str, client_secret: str, keyword: str):
    """
    Priority 2: Fallback to Naver Search API (Local) if Kakao fails or is unavailable.
    """
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    params = {
        "query": keyword,
        "display": 5
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        raw_places = data.get("items", [])
        formatted_places = []
        
        for p in raw_places:
            # Strip HTML tags from name
            name = p.get("title", "").replace("<b>", "").replace("</b>", "")
            category = p.get("category", "").split(">")[0].strip()
            address = p.get("roadAddress") or p.get("address")
            
            # Naver returns coordinates in KATECH (TM128) format. 
            # To get decimal lat/lon, standard API requires coordinates conversion or geocoding.
            # Here we fetch approximate coordinates using naver geocode or default to Gyeonggi reference to avoid crash.
            # Note: For pure local keyword search, Naver coordinates are converted.
            # Here we map mapx/mapy (KATECH) approximately or use center coords.
            mapx = float(p.get("mapx", 0)) / 10000000.0
            mapy = float(p.get("mapy", 0)) / 10000000.0
            
            # Naver returns 10M scaled KATECH or sometimes standard coordinates. 
            # If standard, we map them directly.
            latitude = mapy if mapy > 30 else 37.26357  # Fallback to Suwon center if invalid coords
            longitude = mapx if mapx > 120 else 127.0286
            
            description = get_enriched_description(name, category, address)
            
            formatted_places.append({
                "name": name,
                "category": category,
                "latitude": latitude,
                "longitude": longitude,
                "address": address,
                "description": description
            })
        return formatted_places
    except Exception as e:
        print(f"[API ERROR] Naver Search API call failed: {str(e)}")
        return []

def fetch_places_unified(kakao_key: str, naver_id: str, naver_secret: str, keyword: str, lat: float, lon: float):
    """
    Tries Kakao Local API first, falls back to Naver Search API if Kakao is unauthorized, limited, or errors.
    """
    # 1. Try Kakao first
    if kakao_key and kakao_key != "YOUR_KAKAO_REST_API_KEY":
        res = fetch_kakao_places(kakao_key, keyword, lat, lon)
        if res is not None:
            print(f"[UNIFIED] Successfully fetched {len(res)} places from Kakao API.")
            return res
            
    # 2. Fallback to Naver Search API
    if naver_id and naver_secret and naver_id != "YOUR_NAVER_CLIENT_ID":
        print("[UNIFIED] Falling back to Naver Search API...")
        res = fetch_naver_places(naver_id, naver_secret, keyword)
        if res:
            print(f"[UNIFIED] Successfully fetched {len(res)} places from Naver Search API.")
            return res
            
    return []

def save_and_index_places(places_list):
    """
    Inserts newly fetched places into SQLite and completely updates/saves the FAISS index file.
    """
    if not places_list:
        print("[INDEX] No new data to save.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    new_inserts = 0
    for p in places_list:
        # Check uniqueness by name and coordinates to prevent redundancy
        cursor.execute("SELECT id FROM places WHERE name = ? AND latitude = ? AND longitude = ?", (p["name"], p["latitude"], p["longitude"]))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO places (name, category, latitude, longitude, address, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (p["name"], p["category"], p["latitude"], p["longitude"], p["address"], p["description"]))
            new_inserts += 1
            
    conn.commit()
    print(f"[DB] Inserted {new_inserts} new unique places to SQLite.")
    
    # Read ALL places to build the FAISS Index
    cursor.execute("SELECT id, name, category, latitude, longitude, description FROM places")
    all_places = cursor.fetchall()
    conn.close()
    
    if not all_places:
        print("[INDEX] SQLite database is empty. Cannot build index.")
        return
        
    print(f"[INDEX] Rebuilding FAISS Index with {len(all_places)} total places...")
    descriptions = [p[5] for p in all_places]
    
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    with torch.no_grad():
        embeddings = model.encode(
            descriptions,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_PATH)
    print(f"[INDEX] Successfully updated FAISS Index at: {INDEX_PATH}")
    print(f"[SUCCESS] Re-indexed {len(all_places)} total items.")

def run_batch_crawling(kakao_key: str, naver_id: str, naver_secret: str, keywords: list):
    """
    Performs grid crawling across main administrative centers in Gyeonggi-do using specified keywords.
    """
    print(f"\n[CRAWLER] Starting Gyeonggi-do grid crawling for keywords: {keywords}")
    all_collected = []
    
    for r_name, coords in GYEONGGI_REGIONS.items():
        print(f"\nScanning Region: {r_name} ({coords['lat']}, {coords['lon']})...")
        for kw in keywords:
            # Combine region name with search keyword to ensure geographically restricted search
            search_query = f"{r_name.split()[0]} {kw}"
            places = fetch_places_unified(
                kakao_key, 
                naver_id, 
                naver_secret, 
                search_query, 
                coords["lat"], 
                coords["lon"]
            )
            all_collected.extend(places)
            # Sleep briefly to be respectful to API rate limits
            time.sleep(0.3)
            
    print(f"\n[CRAWLER] Completed Gyeonggi-do grid scan. Total raw places gathered: {len(all_collected)}")
    save_and_index_places(all_collected)

def load_env_file():
    """Manually reads .env file in the current directory and loads variables into os.environ."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

if __name__ == "__main__":
    print("="*70)
    print("      SpotSync AI - Gyeonggi-do Automated Real Place Collector")
    print("="*70)
    
    # Load environment variables manually
    load_env_file()
    
    init_db()
    
    # 1. Obtain API credentials from environmental variables
    KAKAO_KEY = os.getenv("KAKAO_API_KEY", "YOUR_KAKAO_REST_API_KEY")
    NAVER_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID")
    NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET")
    
    # Check if credentials are set
    if KAKAO_KEY == "YOUR_KAKAO_REST_API_KEY" and NAVER_ID == "YOUR_NAVER_CLIENT_ID":
        print("\n[⚠️ NOTICE] API Keys not provided yet.")
        print("Please provide KAKAO_API_KEY or NAVER_CLIENT_ID to pull real Gyeonggi-do data.")
        print("Generating mock Gyeonggi-do sample places for database initialization...")
        
        sample_gyeonggi = [
            {
                "name": "수원 화성행궁 조용한 카페",
                "category": "카페",
                "latitude": 37.2831,
                "longitude": 127.0135,
                "address": "경기도 수원시 팔달구 정조로 825",
                "description": "수원 화성행궁 근처에 위치한 아늑하고 조용하여 노트북 작업과 독서에 적합한 한옥 스타일의 카페입니다."
            },
            {
                "name": "분당 서현역 게이밍 PC존",
                "category": "PC방",
                "latitude": 37.3837,
                "longitude": 127.1265,
                "address": "경기도 성남시 분당구 서현로",
                "description": "RTX 4070급 초고성능 게이밍 그래픽카드가 전 좌석에 탑재되어 끊김 없이 FPS 게임을 즐길 수 있는 프리미엄 PC방입니다."
            },
            {
                "name": "일산 호수공원 드럼 합주실",
                "category": "합주실",
                "latitude": 37.6601,
                "longitude": 126.7725,
                "address": "경기도 고양시 일산동구 호수로",
                "description": "일산 호수공원 인근에 위치하며 최신식 드럼 세트와 방음 도어가 설비된 밴드 및 보컬 음악 합주실입니다."
            }
        ]
        save_and_index_places(sample_gyeonggi)
    else:
        # Crawling targets in Gyeonggi-do
        target_keywords = ["합주실", "카페", "PC방", "노래연습장"]
        run_batch_crawling(KAKAO_KEY, NAVER_ID, NAVER_SECRET, target_keywords)
        
    print("\n[SUCCESS] Script setup complete. Ready to receive real API keys and execute.")
    print("="*70)

