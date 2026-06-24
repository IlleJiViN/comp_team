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
        
    # Target all nationwide CSV files
    target_files = csv_files
    
    if not target_files:
        raise FileNotFoundError("Could not find any CSV files even after extraction.")
        
    print(f"[PIPELINE] Found target CSV files to parse: {target_files}")
    
    # Load using UTF-8 with robust replacement of corrupted bytes to ensure perfect Korean loading
    print(f"[PIPELINE] Loading CSVs with UTF-8 encoding (resilient bad byte replacement)...")
    t0 = time.perf_counter()
    dfs = []
    for csv_file_path in target_files:
        print(f"  - Loading {csv_file_path}...")
        df_part = pd.read_csv(
            csv_file_path, 
            sep=',', 
            encoding='utf-8', 
            encoding_errors='replace', 
            dtype=str
        )
        dfs.append(df_part)
    
    df = pd.concat(dfs, ignore_index=True)
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
          
    # No region filtering - loading nationwide data.
    print(f"[PIPELINE] Initial Row Count: {len(df)}")
    
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
    
    # V4 uses raw name and category on-the-fly, so no need to build embedding_text columns here.
    # We create empty placeholder columns just to avoid changing the DB load schema.
    df["embedding_text"] = ""
    df["embedding_text_v2"] = ""
    df["embedding_text_v3"] = ""
    
    df = df.drop(columns=["_branch", "_major", "_mid", "_bldg"], errors='ignore')
    
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
