import psycopg2
import pandas as pd

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def main():
    print("Connecting to DB...")
    conn = psycopg2.connect(DATABASE_URL)
    
    query = """
        SELECT id, name, category, COALESCE(description, '') as description
        FROM places
        WHERE address LIKE '%마포구%'
        ORDER BY id
    """
    
    print("Fetching enriched Mapo-gu data from places table...")
    df = pd.read_sql(query, conn)
    
    print("Preparing text for BGE-M3 embedding...")
    # BGE-M3 text chunking strategy: Name + Category + Description
    df['text_val'] = df['name'] + " " + df['category'] + " " + df['description']
    
    output_csv = "data/mapo_enriched_for_bge.csv"
    df[['id', 'text_val']].to_csv(output_csv, index=False)
    print(f"Exported {len(df)} rows to {output_csv}.")
    
if __name__ == "__main__":
    main()
