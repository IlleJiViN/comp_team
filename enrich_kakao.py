import os
import psycopg2
import requests
import time
from dotenv import load_dotenv

load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def search_blog(query):
    url = "https://dapi.kakao.com/v2/search/blog"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": query, "size": 3}
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        
        contents = []
        for doc in data.get("documents", []):
            text = doc.get("contents", "").replace("<b>", "").replace("</b>", "").strip()
            if text:
                contents.append(text)
                
        return " ".join(contents)
    except Exception as e:
        print(f"Error fetching {query}: {e}")
        return ""

def main():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    
    # We will grab 5 places that have is_enriched=TRUE but description IS NULL
    # Because my previous query accidentally set description to NULL!
    cur.execute("""
        SELECT id, name, category, address 
        FROM places 
        WHERE is_enriched = TRUE AND description IS NULL
        LIMIT 5
    """)
    rows = cur.fetchall()
    
    for row in rows:
        pid, name, category, address = row
        
        address_parts = address.split()
        dong = ""
        for part in address_parts:
            if part.endswith('동') or part.endswith('로') or part.endswith('길'):
                dong = part
                break
                
        search_query = f"{dong} {name} {category}"
        blog_text = search_blog(search_query)
        
        if blog_text:
            # Fix: use COALESCE
            cur.execute("""
                UPDATE places 
                SET description = COALESCE(description, '') || '[블로그 리뷰 요약] ' || %s
                WHERE id = %s
            """, (blog_text, pid))
            print(f"Updated {name}: {blog_text[:100]}...")
        
        time.sleep(0.2)

if __name__ == "__main__":
    main()
