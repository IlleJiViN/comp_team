import json
import os
import time
import psycopg2
from psycopg2.extras import DictCursor
from tqdm import tqdm

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
OUTPUT_JSONL_FILE = "data/spotsync_summaries.jsonl"

def generate_summary_mock(name, category, reviews):
    """
    Mock function representing the API call to Vertex AI / Google GenAI.
    """
    if not reviews or len(reviews.strip()) < 10:
        return "정보 부족으로 인한 요약 불가"
        
    return (
       f"[{name}]은(는) {category} 장소로, 포근하고 아늑한 분위기가 돋보이는 곳입니다. "
       f"방문자들은 특히 이곳의 뛰어난 맛과 재료의 품질에 대해 입을 모아 칭찬합니다. "
       f"가격대는 다소 높은 편이지만 그에 걸맞은 높은 수준의 친절한 서비스와 전문적인 응대가 제공됩니다. "
       f"조용한 대화나 모임, 혹은 특별한 날 데이트를 즐기고자 하는 추천대상에게 안성맞춤인 공간입니다."
    )

def main():
    print("Connecting to local PostgreSQL database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=DictCursor)
    
    cur.execute("""
        SELECT id, place_id, name, category, address, description 
        FROM places 
        WHERE address LIKE '서울%' 
          AND is_enriched = TRUE 
          AND description IS NOT NULL 
          AND description != ''
    """)
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} enriched places to process...")
    
    os.makedirs(os.path.dirname(OUTPUT_JSONL_FILE), exist_ok=True)
    
    processed_ids = set()
    if os.path.exists(OUTPUT_JSONL_FILE):
        with open(OUTPUT_JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_ids.add(record.get("id"))
                    except:
                        pass
    print(f"Already processed {len(processed_ids)} places. Resuming...")

    with open(OUTPUT_JSONL_FILE, "a", encoding="utf-8") as f:
        for row in tqdm(rows, desc="Summarizing places (Mock)"):
            place_id = row['place_id']
            if place_id in processed_ids:
                continue

            name = row['name']
            category = row['category']
            address = row['address']
            reviews = row['description']
            
            summary = generate_summary_mock(name, category, reviews)
            
            record = {
                "id": place_id,
                "title": name,
                "category": category,
                "address": address,
                "normalized_summary": summary,
                "raw_reviews": reviews
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Processing complete! JSONL output saved to: {OUTPUT_JSONL_FILE}")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
