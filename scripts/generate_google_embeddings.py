import os
import sys
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import errors

# Configure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Input configuration
CSV_PATH = "data/all_chunked_for_bge.csv"
OUTPUT_NPZ = "google_embeddings.npz"
BATCH_SIZE = 250  # Vertex AI max batch size for text-embedding-004
MAX_WORKERS = 25  # Parallel threads for API calls
PROJECT_ID = "spotsync-500217"
LOCATION = "us-central1"
MODEL_NAME = "text-embedding-004"

def get_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def embed_batch_with_retry(client, texts, batch_idx, max_retries=5):
    # Prepare instances list
    # The official google-genai Client.models.embed_content accepts a list of strings directly
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=texts
            )
            # Extract embeddings
            embeddings = [emb.values for emb in response.embeddings]
            return batch_idx, np.array(embeddings, dtype=np.float32)
        except errors.APIError as e:
            if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                wait_time = (2 ** attempt) + np.random.uniform(0.5, 1.5)
                print(f"[WARN] Batch {batch_idx}: Rate limited (429). Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Batch {batch_idx}: API Error: {e}")
                time.sleep(1)
        except Exception as e:
            print(f"[ERROR] Batch {batch_idx}: Unexpected error: {e}. Retrying...")
            time.sleep(2)
            
    # Return empty on failure
    return batch_idx, None

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV file not found at {CSV_PATH}")
        return

    print(f"[INFO] Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    print(f"[INFO] Loaded {total_rows} rows from CSV.")

    # Initialize client to verify credentials before launching threads
    try:
        client = get_client()
        print("[INFO] Google GenAI Client successfully authenticated.")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return

    # Split texts into batches of BATCH_SIZE
    texts_list = df['text_val'].astype(str).tolist()
    batches = [texts_list[i:i + BATCH_SIZE] for i in range(0, total_rows, BATCH_SIZE)]
    num_batches = len(batches)
    print(f"[INFO] Created {num_batches} batches of size {BATCH_SIZE}.")

    # Check for existing partial file or standard resume logic
    temp_output_dir = "google_emb_temp"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    print("[INFO] Starting parallel embedding generation...")
    start_time = time.time()
    
    # Thread pool for parallel API calls
    results_map = {}
    
    # We will save partial results to prevent data loss
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(embed_batch_with_retry, client, batch, i): i 
            for i, batch in enumerate(batches)
            if not os.path.exists(os.path.join(temp_output_dir, f"part_{i}.npy"))
        }
        
        already_completed = num_batches - len(futures)
        if already_completed > 0:
            print(f"[INFO] Resuming: {already_completed} batches already completed and saved in {temp_output_dir}/.")
            
        completed_count = already_completed
        
        for future in as_completed(futures):
            batch_idx, emb_arr = future.result()
            if emb_arr is not None:
                # Save partial file
                np.save(os.path.join(temp_output_dir, f"part_{batch_idx}.npy"), emb_arr)
                completed_count += 1
                
                if completed_count % 10 == 0 or completed_count == num_batches:
                    elapsed = time.time() - start_time
                    avg_speed = completed_count / elapsed if elapsed > 0 else 0
                    eta_sec = (num_batches - completed_count) / avg_speed if avg_speed > 0 else 0
                    print(f"[PROGRESS] Completed {completed_count}/{num_batches} batches ({completed_count/num_batches*100:.1f}%). Speed: {avg_speed:.2f} batch/s. ETA: {eta_sec/60:.1f}m")
            else:
                print(f"[FATAL] Failed to embed batch {batch_idx} after multiple retries. Exiting.")
                return

    print("[INFO] All batches completed. Assembling final NPZ file...")
    assembled_embeddings = []
    for i in range(num_batches):
        part_file = os.path.join(temp_output_dir, f"part_{i}.npy")
        if os.path.exists(part_file):
            assembled_embeddings.append(np.load(part_file))
        else:
            print(f"[ERROR] Missing part_{i}.npy. Assembling aborted.")
            return
            
    all_embeddings = np.vstack(assembled_embeddings)
    print(f"[INFO] Successfully assembled embeddings. Shape: {all_embeddings.shape}")
    
    np.savez_compressed(OUTPUT_NPZ, embeddings=all_embeddings)
    print(f"[SUCCESS] Saved final compressed embeddings to {OUTPUT_NPZ}")
    
    # Cleanup temp directory
    try:
        for i in range(num_batches):
            os.remove(os.path.join(temp_output_dir, f"part_{i}.npy"))
        os.rmdir(temp_output_dir)
        print("[INFO] Cleaned up temporary files.")
    except Exception as e:
        print(f"[WARN] Cleanup failed: {e}")

if __name__ == "__main__":
    main()
