from sqlalchemy import create_engine, text
import pandas as pd

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("--- Active Embedded Places (V3 is not null) Category Distribution ---")
        df_active = pd.read_sql(text("""
            SELECT category, count(*) as count 
            FROM places 
            WHERE embedding_vector_v3 IS NOT NULL 
            GROUP BY category 
            ORDER BY count DESC 
            LIMIT 20
        """), conn)
        print(df_active)
        
        print("\n--- Total Places in DB Category Distribution ---")
        df_all = pd.read_sql(text("""
            SELECT category, count(*) as count 
            FROM places 
            GROUP BY category 
            ORDER BY count DESC 
            LIMIT 20
        """), conn)
        print(df_all)
except Exception as e:
    print("Error:", e)
