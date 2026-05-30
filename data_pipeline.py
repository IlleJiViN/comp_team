import os
import glob
import zipfile
import pandas as pd
import time
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, insert, text
from sqlalchemy.orm import declarative_base
from geoalchemy2 import Geometry

# Reconfigure stdout/stderr encoding for UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Database Connection URI
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

# SQLAlchemy Model Definition
Base = declarative_base()

class Place(Base):
    __tablename__ = 'places'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    place_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    category = Column(String(100))
    address = Column(Text)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    embedding_text = Column(Text, nullable=False)
    embedding_text_v2 = Column(Text, nullable=True)
    embedding_text_v3 = Column(Text, nullable=True)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)

def extract_and_load_csv():
    """Locates and extracts the zip file if CSV is not already present, then loads it with pandas."""
    zip_path = "소상공인시장진흥공단_상가(상권)정보_20260331.zip"
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Could not find Gyeonggi-do ZIP data file: {zip_path}")
        print(f"[PIPELINE] CSV not found. Extracting ZIP file: {zip_path}...")
        t0 = time.perf_counter()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"[PIPELINE] Extracted ZIP in {time.perf_counter() - t0:.2f} seconds.")
        csv_files = glob.glob("*.csv")
        
    # Target only the Gyeonggi-do CSV file specifically
    gyeonggi_files = [f for f in csv_files if "경기" in f]
    if gyeonggi_files:
        csv_file_path = gyeonggi_files[0]
    else:
        csv_file_path = csv_files[0]
    print(f"[PIPELINE] Found target CSV file to parse: {csv_file_path}")
    
    # Load using UTF-8 with robust replacement of corrupted bytes to ensure perfect Korean loading
    print(f"[PIPELINE] Loading CSV with UTF-8 encoding (resilient bad byte replacement)...")
    t0 = time.perf_counter()
    df = pd.read_csv(
        csv_file_path, 
        sep=',', 
        encoding='utf-8', 
        encoding_errors='replace', 
        dtype=str
    )
    print(f"[PIPELINE] Loaded successfully in {time.perf_counter() - t0:.2f}s. Row count: {len(df)}")
    return df

def clean_and_transform(df: pd.DataFrame):
    """Performs data cleaning, maps columns, filters Gyeonggi-do, and constructs embedding_text."""
    print("[PIPELINE] Beginning data cleaning and transformation...")
    
    # Import category descriptions from shared module
    from categories import CATEGORY_DESCRIPTIONS

    # 1. Check and filter out Closed ('영업상태') status
    status_col = [col for col in df.columns if "영업상태" in col]
    if status_col:
        col_name = status_col[0]
        initial_len = len(df)
        df = df[~df[col_name].astype(str).str.contains("폐업", na=False)]
        print(f"[PIPELINE] Removed {initial_len - len(df)} closed businesses. Remaining: {len(df)}")
        
    # 2. Extract column mappings programmatically to prevent header mismatch
    id_col = [col for col in df.columns if "상가업소번호" in col][0]
    name_col = [col for col in df.columns if "상호명" in col][0]
    
    # Prioritize subcategory (소분류명) over major category (대분류명)
    sub_cats = [col for col in df.columns if "상권업종소분류명" in col]
    category_col = sub_cats[0] if sub_cats else [col for col in df.columns if "상권업종대분류명" in col][0]
    
    address_col = [col for col in df.columns if "도로명주소" in col or "지번주소" in col][0]
    lon_col = [col for col in df.columns if "경도" in col][0]
    lat_col = [col for col in df.columns if "위도" in col][0]
    
    print(f"[PIPELINE] Identified column headers:\n"
          f"  - ID: {id_col}\n"
          f"  - Name: {name_col}\n"
          f"  - Category: {category_col}\n"
          f"  - Address: {address_col}\n"
          f"  - Lon/Lat: {lon_col}/{lat_col}")
          
    # 3. Filter only Gyeonggi-do places (as requested to avoid massive nationwide processing)
    sido_col = [col for col in df.columns if "시도명" in col]
    if sido_col:
        df = df[df[sido_col[0]].astype(str).str.contains("경기", na=False)]
    else:
        df = df[df[address_col].astype(str).str.contains("경기도|경기 ", na=False)]
    print(f"[PIPELINE] Filtered Gyeonggi-do places. Count: {len(df)}")
    
    # Prevent empty records on coordinates
    df = df.dropna(subset=[lat_col, lon_col])
    
    # 3.5 Filter out non-relevant categories (meeting spots themes only) to maintain DB sanity and speed
    all_subcategories = list(CATEGORY_DESCRIPTIONS.keys())
        
    initial_len = len(df)
    df = df[df[category_col].astype(str).isin(all_subcategories)]
    print(f"[PIPELINE] Filtered relevant categories only. Removed {initial_len - len(df)} non-relevant places. Remaining: {len(df)}")
    
    # 4. Generate Embedding Text
    print("[PIPELINE] Building embedding texts...")
    
    # Identify auxiliary columns for richer semantic construction if available
    branch_cols = [col for col in df.columns if "지점명" in col]
    major_cat_cols = [col for col in df.columns if "상권업종대분류명" in col]
    mid_cat_cols = [col for col in df.columns if "상권업종중분류명" in col]
    bldg_cols = [col for col in df.columns if "건물명" in col]
    
    branch_series = df[branch_cols[0]].fillna("") if branch_cols else pd.Series("", index=df.index)
    major_cat_series = df[major_cat_cols[0]].fillna("") if major_cat_cols else pd.Series("", index=df.index)
    mid_cat_series = df[mid_cat_cols[0]].fillna("") if mid_cat_cols else pd.Series("", index=df.index)
    bldg_series = df[bldg_cols[0]].fillna("") if bldg_cols else pd.Series("", index=df.index)
    
    def enrich_row_v1(row):
        addr = str(row[address_col])
        name = str(row[name_col])
        cat = str(row[category_col])
        branch = str(row["_branch"])
        major = str(row["_major"])
        mid = str(row["_mid"])
        bldg = str(row["_bldg"])
        
        full_name = f"{name} {branch}".strip()
        
        desc = f"이 장소는 {addr}"
        if bldg:
            desc += f" ({bldg})"
        desc += f"에 위치한 '{full_name}'이며, 업종은 {cat}입니다."
        
        cat_hierarchy = []
        if major:
            cat_hierarchy.append(major)
        if mid:
            cat_hierarchy.append(mid)
        if cat and cat not in cat_hierarchy:
            cat_hierarchy.append(cat)
            
        desc += " 관련 분류로는 " + ", ".join(cat_hierarchy) + " 등이 있습니다."
        
        lower_name = name.lower()
        lower_cat = cat.lower()
        lower_mid = mid.lower()
        
        enrichments = []
        
        # Cafe, coffee, dessert, bakery
        if any(k in lower_cat or k in lower_mid or k in lower_name for k in ["카페", "커피", "디저트", "베이커리", "제과", "다방", "cafe", "coffee", "바리스타", "찻집"]):
            enrichments.append("노트북 들고 공부하기 편한 조용하고 편안한 카페, 아늑하고 콘센트가 많은 작업 공간, 분위기 좋은 디저트와 맛있는 커피가 있는 베이커리 맛집 공간입니다.")
        # PC Room, gaming
        elif any(k in lower_cat or k in lower_mid or k in lower_name for k in ["pc방", "피씨방", "pc", "게임", "gaming"]):
            enrichments.append("컴퓨터 그래픽카드 최고 사양 게이밍 모니터 넓은 피씨방, 초고속 인터넷, FPS 게임과 다양한 온라인 게임을 즐기기 좋은 프리미엄 게이밍 공간입니다.")
        # Rehearsal room
        elif any(k in lower_cat or k in lower_mid or k in lower_name for k in ["합주실", "연습실", "음악실", "스튜디오", "녹음", "studio", "밴드", "드럼", "마이크", "방음", "음악", "보컬", "악기", "기타", "피아노"]):
            enrichments.append("드럼이랑 마이크 성능 좋은 방음 잘되는 음악 합주실, 보컬 밴드 노래 연습 개인 녹음 스튜디오, 최상의 방음 시설과 전문가용 마이크 및 악기가 완비된 연습실 공간입니다.")
        # Coin karaoke
        elif any(k in lower_cat or k in lower_mid or k in lower_name for k in ["노래방", "코인", "노래연습장", "노래", "코노"]):
            enrichments.append("혼자 가서 보컬 연습하고 마이크 녹음하기 조용한 스튜디오, 음질 좋은 최신 반주기와 무선 마이크, 화려한 LED 조명을 갖춘 코인 노래연습장입니다.")
        # Restaurant
        elif any(k in lower_cat or k in lower_mid or k in lower_name for k in ["식당", "음식점", "밥집", "맛집", "한식", "중식", "일식", "양식", "분식", "고기", "구이", "치킨", "피자", "파스타", "국밥"]):
            enrichments.append("맛있고 위생적이며 친절한 식당, 다양한 메뉴를 제공하여 가족 외식, 친구들과의 모임 및 데이트 코스로 추천하는 맛집 공간입니다.")
        # Pub
        elif any(k in lower_cat or k in lower_mid or k in lower_name for k in ["술집", "호프", "맥주", "포차", "이자카야", "바(bar)", "펍", "bar", "pub"]):
            enrichments.append("안주가 맛있고 분위기 좋은 감성 술집, 시원한 맥주나 하이볼, 칵테일 한잔하며 이야기 나누기 좋은 모임 공간입니다.")
            
        if enrichments:
            desc += " " + " ".join(enrichments)
            
        return desc

    def enrich_row_v2(row):
        name = str(row[name_col])
        cat = str(row[category_col])
        branch = str(row["_branch"])
        major = str(row["_major"])
        mid = str(row["_mid"])
        
        full_name = f"{name} {branch}".strip()
        
        cat_hierarchy = []
        if major:
            cat_hierarchy.append(major)
        if mid:
            cat_hierarchy.append(mid)
            
        hierarchy_str = " > ".join(cat_hierarchy)
        
        if hierarchy_str:
            desc = f"{full_name}은 {cat} ({hierarchy_str})"
        else:
            desc = f"{full_name}은 {cat}"
        return desc
    
    def enrich_row_v3(row):
        name = str(row[name_col])
        cat = str(row[category_col])
        branch = str(row["_branch"])
        major = str(row["_major"])
        mid = str(row["_mid"])
        
        full_name = f"{name} {branch}".strip()
        
        desc = CATEGORY_DESCRIPTIONS.get(cat)
        if desc:
            return f"{full_name}은 {cat} ({major} > {mid}) 업종에 속하며, {desc}"
        else:
            desc_fallback = f"전문성과 오랜 경험을 바탕으로 친절하고 체계적인 서비스를 제공하여 방문하시는 모든 고객에게 깊은 만족을 선사하는 신뢰도 높고 깔끔한 {major} {mid} {cat} 관련 비즈니스 전문 공간입니다."
            return f"{full_name}은 {cat} ({major} > {mid}) 업종에 속하며, {desc_fallback}"

    df["_branch"] = branch_series
    df["_major"] = major_cat_series
    df["_mid"] = mid_cat_series
    df["_bldg"] = bldg_series
    
    df["embedding_text"] = df.apply(enrich_row_v1, axis=1)
    df["embedding_text_v2"] = df.apply(enrich_row_v2, axis=1)
    df["embedding_text_v3"] = df.apply(enrich_row_v3, axis=1)
    
    df = df.drop(columns=["_branch", "_major", "_mid", "_bldg"])
    
    # Convert lat/lon to float
    df[lat_col] = df[lat_col].astype(float)
    df[lon_col] = df[lon_col].astype(float)
    
    return df, id_col, name_col, category_col, address_col, lon_col, lat_col

def load_to_postgis(df: pd.DataFrame, id_col, name_col, category_col, address_col, lon_col, lat_col):
    """Creates the PostGIS database table and batch inserts the filtered records."""
    print(f"[DB] Connecting to PostGIS database at: {DATABASE_URL}...")
    engine = create_engine(DATABASE_URL)
    
    # 1. Enable PostGIS Extension if not already active
    with engine.begin() as conn:
        print("[DB] Activating PostGIS extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        
    # 2. Create tables
    print("[DB] Creating 'places' database table schema...")
    Base.metadata.create_all(engine)
    
    # 3. Prepare dataset dictionary structures
    print("[DB] Preparing data rows for database insertion...")
    t_prep = time.perf_counter()
    records = df[[id_col, name_col, category_col, address_col, lat_col, lon_col, "embedding_text", "embedding_text_v2", "embedding_text_v3"]].to_dict(orient='records')
    places_to_insert = [
        {
            "place_id": str(row[id_col]),
            "name": str(row[name_col]),
            "category": str(row[category_col]),
            "address": str(row[address_col]),
            "latitude": float(row[lat_col]),
            "longitude": float(row[lon_col]),
            "embedding_text": str(row["embedding_text"]),
            "embedding_text_v2": str(row["embedding_text_v2"]),
            "embedding_text_v3": str(row["embedding_text_v3"]),
            "location": f"SRID=4326;POINT({row[lon_col]} {row[lat_col]})"
        }
        for row in records
    ]
    print(f"[DB] Preprocessed {len(places_to_insert)} records in {time.perf_counter() - t_prep:.2f} seconds.")
        
    # 4. Batch Insertion
    batch_size = 2000
    total_records = len(places_to_insert)
    print(f"[DB] Initiating batch insertion of {total_records} records (size: {batch_size})...")
    
    t0 = time.perf_counter()
    with engine.begin() as conn:
        # Clear existing places in target table
        conn.execute(text("TRUNCATE TABLE places RESTART IDENTITY CASCADE;"))
        
        for i in range(0, total_records, batch_size):
            batch = places_to_insert[i:i+batch_size]
            conn.execute(insert(Place), batch)
            print(f"[DB] Inserted batch {i//batch_size + 1}/{(total_records-1)//batch_size + 1} ({min(i+batch_size, total_records)}/{total_records})")
            
    print(f"[SUCCESS] PostGIS load completed successfully in {time.perf_counter() - t0:.2f}s!")

if __name__ == "__main__":
    print("="*80)
    print("      SpotSync AI - PostGIS Data Pipeline & ETL Processor")
    print("="*80)
    
    try:
        raw_df = extract_and_load_csv()
        cleaned_df, id_c, name_c, cat_c, addr_c, lon_c, lat_c = clean_and_transform(raw_df)
        load_to_postgis(cleaned_df, id_c, name_c, cat_c, addr_c, lon_c, lat_c)
        print("\n🎉 ETL Data Pipeline execution finished successfully!")
    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR] Pipeline failed: {str(e)}")
        print("Please ensure your local PostGIS Docker container is running ('docker-compose up -d').")
    print("="*80)
