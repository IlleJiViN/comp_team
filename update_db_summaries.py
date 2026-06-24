import json
import psycopg2

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
INPUT_JSONL = "data/spotsync_summaries.jsonl"

def update_summaries():
    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    updates = []
    
    print(f"Reading summaries from {INPUT_JSONL}...")
    try:
        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    place_id = record['id']
                    summary = record['normalized_summary']
                    # We will update description with the normalized summary.
                    # Or we can append it. Let's just overwrite for now as it's the final normalized summary.
                    updates.append((summary, place_id))
    except Exception as e:
        print(f"Failed to read jsonl: {e}")
        return
        
    print(f"Loaded {len(updates)} summaries. Updating DB...")
    
    from psycopg2.extras import execute_values
    
    query = """
    UPDATE places AS p
    SET description = v.summary
    FROM (VALUES %s) AS v(summary, place_id)
    WHERE p.place_id = v.place_id
    """
    
    try:
        execute_values(cur, query, updates, template="(%s, %s)", page_size=1000)
        conn.commit()
        print("Successfully updated DB with new summaries!")
    except Exception as e:
        conn.rollback()
        print(f"Failed to update DB: {e}")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    update_summaries()
