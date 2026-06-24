import os
import time
import torch
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Settings
MODEL_NAME = "jhgan/ko-sroberta-multitask"
BATCH_SIZE = 256
INPUT_FILE = "places_data.json"
OUTPUT_FILE = "rich_place_embeddings.pt"

def fetch_data():
    print(f"[1] Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please ensure it is in the same directory.")
        return []
        
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)
    
    places = []
    for r in rows:
        pid = r['id']
        name = r['name']
        cat = r['category']
        addr = r['address']
        desc = r['description']
        
        # Limit description length
        short_desc = desc[:1000]
        
        # Combine into rich text format
        rich_text = f"장소명: {name} | 카테고리: {cat} | 주소: {addr} | 특징 및 리뷰: {short_desc}"
        places.append((pid, rich_text))
        
    print(f"    -> Fetched {len(places)} places.")
    return places

def main():
    print("=== GPU Embedding Worker for SpotSync ===")
    
    # 1. Fetch data
    places = fetch_data()
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
    
    print("\n=== All Done! ===")

if __name__ == "__main__":
    main()
