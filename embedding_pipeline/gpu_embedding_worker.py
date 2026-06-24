import os
import time
import torch
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/spotsync")
MODEL_NAME = "jhgan/ko-sroberta-multitask"
BATCH_SIZE = 256
OUTPUT_FILE = "rich_place_embeddings.pt"

def fetch_data(conn):
    print("[1] Fetching all places from database...")
    cur = conn.cursor()
    # Fetch id, name, category, address, and description (reviews)
    cur.execute("SELECT id, name, category, address, description FROM places")
    rows = cur.fetchall()
    
    places = []
    for r in rows:
        pid, name, cat, addr, desc = r
        name = name or ""
        cat = cat or ""
        addr = addr or ""
        desc = desc or ""
        
        # Limit description length to avoid exceeding model's max token length (usually 512 tokens)
        # We take the first 1000 characters of the combined reviews which usually covers the most important points.
        short_desc = desc[:1000]
        
        # Combine into rich text format
        rich_text = f"장소명: {name} | 카테고리: {cat} | 주소: {addr} | 특징 및 리뷰: {short_desc}"
        places.append((pid, rich_text))
        
    print(f"    -> Fetched {len(places)} places.")
    return places

def main():
    print("=== GPU Embedding Worker for SpotSync ===")
    
    # 1. Connect to DB
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please check your .env file or DATABASE_URL.")
        return

    # 2. Fetch data
    places = fetch_data(conn)
    if not places:
        print("No places found. Exiting.")
        return

    # 3. Load Model
    print(f"\n[2] Loading SentenceTransformer model: {MODEL_NAME}")
    # Device setup (uses CUDA if available, fallback to MPS for Mac, or CPU)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"    -> Using device: {device.upper()}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 4. Extract IDs and Texts
    ids = [p[0] for p in places]
    texts = [p[1] for p in places]

    # 5. Generate Embeddings
    print("\n[3] Generating Embeddings (This will be fast on a GPU!)...")
    start_time = time.time()
    
    # encode() handles batching internally, but we specify show_progress_bar
    vectors = model.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    elapsed = time.time() - start_time
    print(f"    -> Encoding completed in {elapsed:.2f} seconds.")

    # 6. Save locally as .pt
    print(f"\n[4] Saving vectors to {OUTPUT_FILE}...")
    torch.save({"ids": ids, "vectors": vectors}, OUTPUT_FILE)
    print("    -> Saved successfully. You can now upload this file to the server.")
    
    conn.close()
    print("\n=== All Done! ===")

if __name__ == "__main__":
    main()
