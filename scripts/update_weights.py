import re

with open("ai_search_v10.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update SearchQuery
old_sq = """class SearchQuery(BaseModel):
    query: str
    top_k: int = 3
    user_locations: List[Location] = []"""

new_sq = """class SearchQuery(BaseModel):
    query: str
    top_k: int = 3
    user_locations: List[Location] = []
    semantic_weight: float = 0.8
    spatial_weight: float = 0.2"""

content = content.replace(old_sq, new_sq)

# 2. Replace the V10 search query block
old_v10 = """        if explicit_location_requested or not midpoint:
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
    except Exception as e:"""

new_v10 = """        if explicit_location_requested or not midpoint:
            print("[V10] Global Vector Search without hard radius...")
            cur.execute(\"\"\"
                SELECT p.id, p.name, p.category, MIN(c.embedding <=> %s::halfvec) as dist, MIN(c.embedding <=> %s::halfvec) as dist_raw
                FROM places_chunks c
                JOIN places p ON p.id = c.place_id
                GROUP BY p.id, p.name, p.category
                ORDER BY dist ASC
                LIMIT 150
            \"\"\", (query_emb_str, query_emb_str))
            pg_hits = cur.fetchall()
        else:
            mid_lat, mid_lng = midpoint
            for radius in [7000, 15000, 50000]:
                print(f"[V10] Spatial-Semantic Search within {radius}m... (semantic: {req.semantic_weight}, spatial: {req.spatial_weight})")
                cur.execute(\"\"\"
                    SELECT p.id, p.name, p.category, 
                           MIN(%s * (c.embedding <=> %s::halfvec) + 
                               %s * (ST_Distance(p.location::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) / %s)) as combined_score,
                           MIN(c.embedding <=> %s::halfvec) as dist_raw
                    FROM places_chunks c
                    JOIN places p ON p.id = c.place_id
                    WHERE ST_DWithin(p.location::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
                    GROUP BY p.id, p.name, p.category
                    ORDER BY combined_score ASC
                    LIMIT 150
                \"\"\", (req.semantic_weight, query_emb_str, req.spatial_weight, mid_lng, mid_lat, radius, query_emb_str, mid_lng, mid_lat, radius))
                pg_hits = cur.fetchall()
                if len(pg_hits) >= 10:
                    break
    except Exception as e:"""

content = content.replace(old_v10, new_v10)

with open("ai_search_v10.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated weights in V10!")
