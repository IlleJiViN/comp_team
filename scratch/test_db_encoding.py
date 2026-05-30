import sys
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        sample = conn.execute(text("""
            SELECT id, name, embedding_text_v3
            FROM places
            WHERE embedding_text_v3 IS NOT NULL
            LIMIT 10
        """)).all()
        
        with open("scratch/db_test_utf8.txt", "w", encoding="utf-8") as f:
            for row in sample:
                f.write(f"ID: {row[0]}\n")
                f.write(f"Name: {row[1]}\n")
                f.write(f"V3 Text: {row[2]}\n")
                f.write("-" * 50 + "\n")
        print("Success! File scratch/db_test_utf8.txt written.")
except Exception as e:
    print("Error:", e)
