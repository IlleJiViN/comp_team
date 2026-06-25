# -*- coding: utf-8 -*-
import requests
import psycopg2
import sys
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

def fetch_osm_data():
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:25];
    area["name"="서울특별시"]->.searchArea;
    (
      node["name"~"맥도날드|CGV|스타벅스|버거킹|투썸플레이스"](area.searchArea);
    );
    out body;
    """
    print("Fetching data from OpenStreetMap Overpass API...")
    headers = {
        "User-Agent": "AntigravityDataBot/1.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers)
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return []
        
    try:
        data = response.json()
        elements = data.get('elements', [])
        print(f"Found {len(elements)} brand locations.")
        return elements
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        return []

def insert_into_db(elements):
    print("Loading embedding model...")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask", device="cpu")
    
    print("Connecting to DB...")
    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/spotsync")
    cur = conn.cursor()
    
    inserted = 0
    for el in elements:
        osm_id = el.get('id')
        name = el.get('tags', {}).get('name', 'Unknown')
        lat = el.get('lat')
        lon = el.get('lon')
        
        # Determine category based on name
        category = "카페"
        if "맥도날드" in name or "버거킹" in name:
            category = "패스트푸드"
        elif "CGV" in name:
            category = "영화관"
            
        address = "서울특별시 " + el.get('tags', {}).get('addr:street', '')
        
        # Check if already exists by checking distance < 10m
        cur.execute("""
            SELECT id FROM places 
            WHERE name = %s 
            AND ST_DWithin(location, ST_SetSRID(ST_MakePoint(%s, %s), 4326), 0.0001)
        """, (name, lon, lat))
        
        if cur.fetchone():
            continue
            
        # Generate embedding
        text_for_embedding = f"{name} {category}"
        embedding = model.encode([text_for_embedding], normalize_embeddings=True)[0]
        
        # Insert
        try:
            cur.execute("""
                INSERT INTO places (place_id, name, category, latitude, longitude, address, location, embedding_vector_v4, embedding_text)
                VALUES (%s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s)
            """, (str(osm_id), name, category, lat, lon, address, lon, lat, embedding.tolist(), text_for_embedding))
            inserted += 1
        except Exception as e:
            print(f"Failed to insert {name}: {e}")
            conn.rollback()
            continue
        
        if inserted % 100 == 0:
            print(f"Inserted {inserted} locations...")
            conn.commit()
            
    conn.commit()
    print(f"Successfully inserted {inserted} new brand locations into the DB!")
    cur.close()
    conn.close()

if __name__ == "__main__":
    elements = fetch_osm_data()
    if elements:
        insert_into_db(elements)
