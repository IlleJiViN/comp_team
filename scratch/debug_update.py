from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        # Fetch 5 rows
        rows = conn.execute(text("SELECT id FROM places WHERE embedding_vector IS NOT NULL LIMIT 5")).all()
        ids = [r[0] for r in rows]
        print("Selected IDs:", ids)
        
    # Try updating in a transaction block
    test_vector = [0.1] * 768
    
    print("\nExecuting update inside engine.begin()...")
    with engine.begin() as conn:
        for idx in ids:
            res = conn.execute(
                text("UPDATE places SET embedding_vector_v2 = :emb WHERE id = :id"),
                {"emb": test_vector, "id": idx}
            )
            print(f"Updated ID {idx}, rowcount: {res.rowcount}")
            
    # Check if they persisted
    print("\nVerifying in a new connection...")
    with engine.connect() as conn:
        for idx in ids:
            val = conn.execute(text("SELECT embedding_vector_v2 FROM places WHERE id = :id"), {"id": idx}).scalar()
            print(f"ID {idx} in DB has vector? {val is not None}")
            
except Exception as e:
    print("Error:", e)
