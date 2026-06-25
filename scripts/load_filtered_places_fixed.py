import os
import glob
import pandas as pd
import time
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean, insert, text
from sqlalchemy.orm import declarative_base
from geoalchemy2 import Geometry

# Reconfigure stdout/stderr encoding for UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Database Connection URI
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

Base = declarative_base()

class Place(Base):
    __tablename__ = 'places'
    
    id = Column(Integer, primary_key=True)
    place_id = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    category = Column(String(100))
    address = Column(Text)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    embedding_text = Column(Text, nullable=True, default="")
    embedding_text_v2 = Column(Text, nullable=True, default="")
    embedding_text_v3 = Column(Text, nullable=True, default="")
    description = Column(Text, nullable=True, default="")
    is_enriched = Column(Boolean, nullable=True, default=False)
    is_premium = Column(Boolean, nullable=True, default=False)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)

def main():
    print("="*80)
    print("      SpotSync AI - Robust PostGIS Data Loader & ID Mapper")
    print("="*80)

    # 1. Load target unique place details from all_chunked_for_bge.csv
    chunked_csv_path = "data/all_chunked_for_bge.csv"
    if not os.path.exists(chunked_csv_path):
        print(f"[ERROR] Could not find chunked CSV: {chunked_csv_path}")
        return

    print(f"[1/4] Loading unique place details from {chunked_csv_path}...")
    t0 = time.perf_counter()
    df_chunked = pd.read_csv(chunked_csv_path, usecols=['id', 'name', 'category'])
    unique_places = df_chunked.drop_duplicates(subset=['id']).copy()
    total_targets = len(unique_places)
    print(f"Loaded {total_targets} unique target places in {time.perf_counter() - t0:.2f} seconds.")

    # Build a lookup index for ultra-fast matching
    # Map name -> list of (id, category)
    lookup = {}
    for _, row in unique_places.iterrows():
        pid = int(row['id'])
        name = str(row['name']).strip()
        cat = str(row['category']).strip() if pd.notna(row['category']) else ""
        if name not in lookup:
            lookup[name] = []
        lookup[name].append((pid, cat))

    print(f"Lookup index built with {len(lookup)} unique names.")

    # 2. Locate and scan all regional CSV files to match targets
    csv_files = glob.glob("data/소상공인시장진흥공단_상가*.csv")
    if not csv_files:
        print("[ERROR] No regional CSV files found in 'data/' directory.")
        return

    print(f"[2/4] Scanning {len(csv_files)} regional CSV files to match coordinates and addresses...")
    
    matched_data = {} # pid -> record dict
    t_scan = time.perf_counter()

    for csv_file in csv_files:
        print(f"  - Scanning {csv_file}...")
        encoding = 'cp949' if '강원' in csv_file else 'utf-8'
        
        try:
            # Read columns we need to save memory
            df_part = pd.read_csv(csv_file, encoding=encoding, encoding_errors='replace', dtype=str)
            
            # Find column headers programmatically
            id_col = [col for col in df_part.columns if "상가업소번호" in col][0]
            name_col = [col for col in df_part.columns if "상호명" in col][0]
            
            sub_cats = [col for col in df_part.columns if "상권업종소분류명" in col]
            category_col = sub_cats[0] if sub_cats else [col for col in df_part.columns if "상권업종대분류명" in col][0]
            
            addr_cols = [col for col in df_part.columns if "도로명주소" in col]
            address_col = addr_cols[0] if addr_cols else [col for col in df_part.columns if "지번주소" in col][0]
            
            lon_col = [col for col in df_part.columns if "경도" in col][0]
            lat_col = [col for col in df_part.columns if "위도" in col][0]
            
            file_matches = 0
            for _, row in df_part.iterrows():
                name = str(row[name_col]).strip()
                if name in lookup:
                    cat = str(row[category_col]).strip() if pd.notna(row[category_col]) else ""
                    candidates = lookup[name]
                    
                    # Try matching by name and category, or name-only if 1 candidate
                    for pid, pcat in candidates:
                        if pid not in matched_data:
                            if pcat == cat or len(candidates) == 1:
                                matched_data[pid] = {
                                    "id": pid,
                                    "place_id": str(row[id_col]), # The new alphanumeric MA... key
                                    "name": str(row[name_col]),
                                    "category": str(row[category_col]),
                                    "address": str(row[address_col]),
                                    "latitude": float(row[lat_col]),
                                    "longitude": float(row[lon_col]),
                                    "embedding_text": "",
                                    "embedding_text_v2": "",
                                    "embedding_text_v3": "",
                                    "description": "",
                                    "is_enriched": False,
                                    "is_premium": False,
                                    "location": f"SRID=4326;POINT({row[lon_col]} {row[lat_col]})"
                                }
                                file_matches += 1
                                break
            print(f"    * Matched {file_matches} unique places from this file.")
            
        except Exception as e:
            print(f"    [ERROR] Failed to process {csv_file}: {e}")

    print(f"Fuzzy-mapping scan complete. Matched {len(matched_data)}/{total_targets} places in {time.perf_counter() - t_scan:.2f} seconds.")

    # 3. Handle unmatched places using robust default fallback
    unmatched_pids = set(unique_places['id'].tolist()) - set(matched_data.keys())
    if unmatched_pids:
        print(f"  - [WARNING] {len(unmatched_pids)} places were unmatched. Using default fallback coordinates (Seoul center)...")
        for pid in unmatched_pids:
            row_info = unique_places[unique_places['id'] == pid].iloc[0]
            matched_data[pid] = {
                "id": pid,
                "place_id": f"FALLBACK_{pid}",
                "name": str(row_info['name']),
                "category": str(row_info['category']) if pd.notna(row_info['category']) else "기타",
                "address": "서울특별시 마포구 (상세주소 미확인)",
                "latitude": 37.5562, # Central Mapo-gu coordinate fallback
                "longitude": 126.9223,
                "embedding_text": "",
                "embedding_text_v2": "",
                "embedding_text_v3": "",
                "description": "",
                "is_enriched": False,
                "is_premium": False,
                "location": "SRID=4326;POINT(126.9223 37.5562)"
            }

    places_to_insert = list(matched_data.values())
    print(f"Total places ready for database loading: {len(places_to_insert)}")

    # 4. Connect to PostGIS and load data
    print(f"[3/4] Connecting to PostGIS database...")
    engine = create_engine(DATABASE_URL)
    
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        
    Base.metadata.create_all(engine)
    
    print(f"[4/4] Truncating 'places' table and inserting {len(places_to_insert)} mapped records...")
    t_load = time.perf_counter()
    
    batch_size = 2000
    total_records = len(places_to_insert)
    
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE places RESTART IDENTITY CASCADE;"))
        
        for i in range(0, total_records, batch_size):
            batch = places_to_insert[i:i+batch_size]
            conn.execute(insert(Place), batch)
            if (i + batch_size) % 10000 == 0 or i + batch_size >= total_records:
                print(f"  - Inserted {min(i+batch_size, total_records)}/{total_records} records...")
                
    print(f"\n🎉 [SUCCESS] PostgreSQL places table populated successfully in {time.perf_counter() - t_load:.2f}s!")
    print(f"Total processing time: {time.perf_counter() - t0:.2f} seconds.")
    print("="*80)

if __name__ == "__main__":
    main()
