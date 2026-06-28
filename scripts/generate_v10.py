import re

with open("ai_search_v9.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove ES imports
content = content.replace("from elasticsearch import Elasticsearch", "")
content = content.replace('es = Elasticsearch("http://localhost:9200", request_timeout=10)', '')

# 2. Replace the Search Logic
search_start_marker = "    # 1. PostGIS Geo-Filtering (Radius search) before ES retrieval"
search_end_marker = "    # --- LIVE FALLBACK MECHANISM ---"

new_search_logic = """    # 1. PostGIS Geo-Filtering & Vector Search (V10)
    midpoint = calculate_midpoint(req.user_locations)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    query_emb_str = f"[{','.join(map(str, query_emb[:512]))}]" # Truncated to 512!
    
    pg_hits = []
    try:
        if explicit_location_requested or not midpoint:
            print("[V10] Global Vector Search without hard radius...")
            cur.execute(\"\"\"
                SELECT p.id, p.name, p.category, MIN(c.embedding <=> %s::halfvec) as dist
                FROM places_chunks c
                JOIN places p ON p.id = c.place_id
                GROUP BY p.id, p.name, p.category
                ORDER BY dist ASC
                LIMIT 150
            \"\"\", (query_emb_str,))
            pg_hits = cur.fetchall()
        else:
            mid_lat, mid_lng = midpoint
            for radius in [7000, 15000, 50000]:
                print(f"[V10] Spatial-Semantic Search within {radius}m...")
                cur.execute(\"\"\"
                    SELECT p.id, p.name, p.category, MIN(c.embedding <=> %s::halfvec) as dist
                    FROM places_chunks c
                    JOIN places p ON p.id = c.place_id
                    WHERE ST_DWithin(p.location::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
                    GROUP BY p.id, p.name, p.category
                    ORDER BY dist ASC
                    LIMIT 150
                \"\"\", (query_emb_str, mid_lng, mid_lat, radius))
                pg_hits = cur.fetchall()
                if len(pg_hits) >= 10:
                    break
    except Exception as e:
        print(f"[ERROR] V10 DB query failed: {e}")
    finally:
        conn.close()

    # Convert to ES hits format to maintain compatibility with the rest of the code
    hits = []
    for row in pg_hits:
        hits.append({
            "_source": {
                "place_id": row[0],
                "name": row[1],
                "category": row[2]
            },
            "_score": max(0.0, 1.0 - float(row[3])) # cosine distance to score
        })
        
    print(f"DEBUG: Initial hits = {len(hits)}", flush=True)
    
"""

# Extract everything before start marker, and everything from end marker
before = content.split(search_start_marker)[0]
after = search_end_marker + content.split(search_end_marker)[1]

final_content = before + new_search_logic + after

with open("ai_search_v10.py", "w", encoding="utf-8") as f:
    f.write(final_content)

print("Generated ai_search_v10.py!")
