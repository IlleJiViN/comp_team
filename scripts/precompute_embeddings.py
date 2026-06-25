import os
import time
import torch
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/spotsync")
MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"
OUTPUT_FILE = "place_name_embeddings.pt"

def precompute_embeddings():
    print("1. Connecting to DB...")
    engine = create_engine(DATABASE_URL)
    
    print("2. Fetching all places...")
    places = []
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, name, category FROM places ORDER BY id ASC"))
        for row in result:
            places.append({
                "id": row[0],
                "text": f"{row[1]} {row[2]}"
            })
            
    print(f"   -> Fetched {len(places)} places.")
    
    print(f"3. Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    print("4. Encoding all places (this may take a few minutes)...")
    texts = [p["text"] for p in places]
    ids = [p["id"] for p in places]
    
    start_time = time.time()
    with torch.no_grad():
        vectors = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    
    elapsed = time.time() - start_time
    print(f"   -> Encoded in {elapsed:.2f} seconds.")
    
    print("5. Saving to disk...")
    # Save as a dictionary mapping id to vector, or just a tensor
    # We will save the IDs array and the vectors array
    data = {
        "ids": np.array(ids, dtype=np.int32),
        "vectors": vectors.astype(np.float32)
    }
    torch.save(data, OUTPUT_FILE)
    print(f"   -> Saved to {OUTPUT_FILE} successfully!")

if __name__ == "__main__":
    precompute_embeddings()
