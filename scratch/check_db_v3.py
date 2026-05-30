from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM places")).scalar()
        v1_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector IS NOT NULL")).scalar()
        v2_text_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_text_v2 IS NOT NULL")).scalar()
        v2_vector_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector_v2 IS NOT NULL")).scalar()
        v3_text_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_text_v3 IS NOT NULL")).scalar()
        v3_vector_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector_v3 IS NOT NULL")).scalar()
        
        print(f"Total rows: {total}")
        print(f"embedding_vector (V1) IS NOT NULL: {v1_not_null}")
        print(f"embedding_text_v2 IS NOT NULL: {v2_text_not_null}")
        print(f"embedding_vector_v2 (V2) IS NOT NULL: {v2_vector_not_null}")
        print(f"embedding_text_v3 IS NOT NULL: {v3_text_not_null}")
        print(f"embedding_vector_v3 (V3) IS NOT NULL: {v3_vector_not_null}")
        
        # Print a sample row
        sample = conn.execute(text("""
            SELECT id, name, embedding_text_v3, 
                   (embedding_vector_v3 IS NOT NULL) as has_v3_vec
            FROM places
            WHERE embedding_text_v3 IS NOT NULL
            LIMIT 1
        """)).first()
        if sample:
            print("\nSample V3 Row:")
            print(f" - ID: {sample[0]}")
            print(f" - Name: {sample[1]}")
            print(f" - V3 Text: {sample[2][:200]}...")
            print(f" - Has V3 Vector: {sample[3]}")
except Exception as e:
    print("Error:", e)
