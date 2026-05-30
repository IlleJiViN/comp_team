import sys
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        # Check extensions
        exts = conn.execute(text("SELECT extname FROM pg_extension")).all()
        print("Installed Extensions:", [e[0] for e in exts])
        
        # Check places table columns
        cols = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'places'
        """)).all()
        print("\nColumns in 'places' table:")
        for col in cols:
            print(f" - {col[0]}: {col[1]}")
            
        # Count total rows
        count = conn.execute(text("SELECT count(*) FROM places")).scalar()
        print(f"\nTotal rows: {count}")
        
        # Count rows with embeddings
        embedded_count = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector IS NOT NULL")).scalar()
        print(f"Total embedded rows: {embedded_count}")
        
except Exception as e:
    print("Error:", e)
