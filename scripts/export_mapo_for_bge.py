import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def export_mapo():
    engine = create_engine(DATABASE_URL)
    query = """
    SELECT id, name, category, description 
    FROM places 
    WHERE address LIKE '%%마포구%%'
    """
    df = pd.read_sql(query, engine)
    
    # Combine name, category, and description for BGE-m3 embedding
    df['text_val'] = df.apply(
        lambda x: f"{x['name']} ({x['category']})\n{x['description']}" if pd.notna(x['description']) else f"{x['name']} ({x['category']})", 
        axis=1
    )
    
    out_file = 'data/mapo_places_for_bge.csv'
    df[['id', 'text_val']].to_csv(out_file, index=False)
    print(f"Exported {len(df)} places to {out_file}")

if __name__ == "__main__":
    export_mapo()
