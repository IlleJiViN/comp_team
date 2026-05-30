from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        # Check if we can select a row
        row = conn.execute(text("SELECT id, name FROM places WHERE embedding_vector IS NOT NULL LIMIT 1")).first()
        print("Selected Row ID:", row[0], "Name:", row[1])
        
        # Try updating
        test_emb = [0.1] * 768
        conn.execute(
            text("UPDATE places SET embedding_vector_v2 = :emb WHERE id = :id"),
            {"emb": test_emb, "id": row[0]}
        )
        print("Update statement executed.")
        
    with engine.connect() as conn:
        val = conn.execute(text("SELECT embedding_vector_v2 FROM places WHERE id = :id"), {"id": row[0]}).scalar()
        print("Updated value in DB (is not None?):", val is not None)
        if val is not None:
            print("Length of vector:", len(val))
            
except Exception as e:
    print("Error:", e)
