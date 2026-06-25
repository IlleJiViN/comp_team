# -*- coding: utf-8 -*-
import requests
import psycopg2
import sys
import json

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
        print(f"Response text: {response.text}")
        return []

if __name__ == "__main__":
    elements = fetch_osm_data()
    if elements:
        print(f"Sample: {json.dumps(elements[:2], ensure_ascii=False, indent=2)}")
