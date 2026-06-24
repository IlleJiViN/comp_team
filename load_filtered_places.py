import os
import glob
import pandas as pd
import time
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, insert, text
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

def main():
    print("="*80)
    print("      SpotSync AI - Filtered PostGIS Data Loader")
    print("="*80)

    # 1. Load target unique IDs from all_chunked_for_bge.csv
    chunked_csv_path = "data/all_chunked_for_bge.csv"
    if not os.path.exists(chunked_csv_path):
        print(f"[ERROR] Could not find chunked CSV: {chunked_csv_path}")
        return

    print(f"[1/4] Loading unique place IDs from {chunked_csv_path}...")
    t0 = time.perf_counter()
    df_chunked = pd.read_csv(chunked_csv_path, usecols=['id'], dtype=int)
    target_ids = set(df_chunked['id'].tolist())
    print(f"Loaded {len(target_ids)} unique target place IDs in {time.perf_counter() - t0:.2f} seconds.")

    # 2. Locate and process all regional CSV files
    csv_files = glob.glob("data/소상공인시장진흥공단_상가*.csv")
    if not csv_files:
        print("[ERROR] No regional CSV files found in 'data/' directory.")
        return

    print(f"[2/4] Scanning and filtering {len(csv_files)} regional CSV files...")
    places_to_insert = []
    processed_ids = set()

    t_scan = time.perf_counter()
    for csv_file in csv_files:
        print(f"  - Processing {csv_file}...")
        # Gangwon is encoded in CP949, others are in UTF-8
        encoding = 'cp949' if '강원' in csv_file else 'utf-8'
        
        try:
            # Read in chunks or completely
            df_part = pd.read_csv(csv_file, encoding=encoding, encoding_errors='replace', dtype=str)
            
            # Find column headers
            id_col = [col for col in df_part.columns if "상가업소번호" in col][0]
            name_col = [col for col in df_part.columns if "상호명" in col][0]
            
            sub_cats = [col for col in df_part.columns if "상권업종소분류명" in col]
            category_col = sub_cats[0] if sub_cats else [col for col in df_part.columns if "상권업종대분류명" in col][0]
            
            addr_cols = [col for col in df_part.columns if "도로명주소" in col]
            address_col = addr_cols[0] if addr_cols else [col for col in df_part.columns if "지번주소" in col][0]
            
            lon_col = [col for col in df_part.columns if "경도" in col][0]
            lat_col = [col for col in df_part.columns if "위도" in col][0]
            
            # Filter rows where ID (as integer) is in target_ids
            df_part[id_col] = df_part[id_col].astype(float).fillna(0).astype(int) # robust cast
            df_matched = df_part[df_part[id_col].isin(target_ids)]
            
            print(f"    * Found {len(df_matched)} matching places out of {len(df_part)} rows.")
            
            for _, row in df_matched.iterrows():
                pid = int(row[id_col])
                if pid in processed_ids:
                    continue
                processed_ids.add(pid)
                
                places_to_insert.append({
                    "id": pid,
                    "place_id": str(pid),
                    "name": str(row[name_col]),
                    "category": str(row[category_col]),
                    "address": str(row[address_col]),
                    "latitude": float(row[lat_col]),
                    "longitude": float(row[lon_col]),
                    "embedding_text": "",
                    "embedding_text_v2": "",
                    "embedding_text_v3": "",
                    "location": f"SRID=4326;POINT({row[lon_col]} {row[lat_col]})"
                })
                
        except Exception as e:
            print(f"    [ERROR] Failed to process {csv_file}: {e}")

    print(f"Filtered total {len(places_to_insert)} unique places in {time.perf_counter() - t_scan:.2f} seconds.")

    # 3. Create schema and load into PostgreSQL
    print(f"[3/4] Connecting to PostGIS and creating table...")
    engine = create_engine(DATABASE_URL)
    
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        
    Base.metadata.create_all(engine)
    
    print(f"[4/4] Truncating 'places' table and inserting {len(places_to_insert)} records...")
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
                
    print(f"\n🎉 [SUCCESS] PostGIS load completed successfully in {time.perf_counter() - t_load:.2f}s!")
    print(f"Total processing time: {time.perf_counter() - t0:.2f} seconds.")
    print("="*80)

if __name__ == "__main__":
    main()
