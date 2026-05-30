from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM places")).scalar()
        korean_names = conn.execute(text("SELECT count(*) FROM places WHERE name NOT LIKE '%%'")).scalar()
        v3_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_text_v3 IS NOT NULL")).scalar()
        
        print(f"Total rows in DB: {total}")
        print(f"Rows with clean Korean names: {korean_names}")
        print(f"Rows with V3 text populated: {v3_not_null}")
        
        # Sample rows
        sample = conn.execute(text("""
            SELECT id, place_id, name, category, address, embedding_text_v3
            FROM places
            WHERE embedding_text_v3 IS NOT NULL
            LIMIT 3
        """)).all()
        for i, row in enumerate(sample, 1):
            print(f"\nSample #{i}:")
            print(f"  - ID: {row[0]}")
            print(f"  - Name: {row[2]}")
            print(f"  - Category: {row[3]}")
            print(f"  - Address: {row[4]}")
            print(f"  - V3 Text: {row[5][:150]}...")
except Exception as e:
    print("Error:", e)
