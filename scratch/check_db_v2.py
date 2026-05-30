from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM places")).scalar()
        v1_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector IS NOT NULL")).scalar()
        v2_text_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_text_v2 IS NOT NULL")).scalar()
        v2_vector_not_null = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector_v2 IS NOT NULL")).scalar()
        
        print(f"Total rows: {total}")
        print(f"embedding_vector (V1) IS NOT NULL: {v1_not_null}")
        print(f"embedding_text_v2 IS NOT NULL: {v2_text_not_null}")
        print(f"embedding_vector_v2 (V2) IS NOT NULL: {v2_vector_not_null}")
        
        # Print a sample row
        sample = conn.execute(text("""
            SELECT id, name, embedding_text, embedding_text_v2, 
                   (embedding_vector IS NOT NULL) as has_v1_vec,
                   (embedding_vector_v2 IS NOT NULL) as has_v2_vec
            FROM places
            WHERE embedding_vector IS NOT NULL
            LIMIT 1
        """)).first()
        if sample:
            print("\nSample Row:")
            print(f" - ID: {sample[0]}")
            print(f" - Name: {sample[1]}")
            print(f" - V1 Text: {sample[2][:100]}...")
            print(f" - V2 Text: {sample[3]}")
            print(f" - Has V1 Vector: {sample[4]}")
            print(f" - Has V2 Vector: {sample[5]}")
except Exception as e:
    print("Error:", e)
