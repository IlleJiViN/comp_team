import os
import sys
import time
import asyncio
import pandas as pd
import numpy as np
from google import genai
from google.genai import errors

# Configure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

CSV_PATH = "data/all_chunked_for_bge.csv"
OUTPUT_NPZ = "google_embeddings.npz"
BATCH_SIZE = 250  # Vertex AI max batch size
CONCURRENCY_LIMIT = 45  # Limit parallel requests to avoid hitting Vertex AI quota
PROJECT_ID = "spotsync-500217"
LOCATION = "us-central1"
MODEL_NAME = "text-embedding-004"

async def embed_batch_with_retry_async(client, texts, batch_idx, semaphore, max_retries=5):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # Use client.aio for async API calls
                response = await client.aio.models.embed_content(
                    model=MODEL_NAME,
                    contents=texts
                )
                embeddings = [emb.values for emb in response.embeddings]
                return batch_idx, np.array(embeddings, dtype=np.float32)
            except errors.APIError as e:
                if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                    wait_time = (2 ** attempt) + np.random.uniform(0.5, 1.5)
                    print(f"[WARN] Batch {batch_idx}: Rate limited (429). Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[ERROR] Batch {batch_idx}: API Error: {e}")
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx}: Unexpected error: {e}. Retrying...")
                await asyncio.sleep(2)
                
        return batch_idx, None

async def main_async():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV file not found at {CSV_PATH}")
        return

    print(f"[INFO] Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    print(f"[INFO] Loaded {total_rows} rows from CSV.")

    # Initialize client to verify credentials
    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        print("[INFO] Google GenAI Client successfully authenticated.")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return

    # Split texts into batches of BATCH_SIZE
    texts_list = df['text_val'].astype(str).tolist()
    batches = [texts_list[i:i + BATCH_SIZE] for i in range(0, total_rows, BATCH_SIZE)]
    num_batches = len(batches)
    print(f"[INFO] Created {num_batches} batches of size {BATCH_SIZE}.")

    temp_output_dir = "google_emb_temp"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    print(f"[INFO] Starting async embedding generation (Concurrency: {CONCURRENCY_LIMIT})...")
    start_time = time.time()
    
    # Check for already completed batches
    batches_to_run = []
    for i, batch in enumerate(batches):
        part_file = os.path.join(temp_output_dir, f"part_{i}.npy")
        if not os.path.exists(part_file):
            batches_to_run.append((i, batch))
            
    already_completed = num_batches - len(batches_to_run)
    if already_completed > 0:
        print(f"[INFO] Resuming: {already_completed} batches already completed and saved in {temp_output_dir}/.")

    completed_count = already_completed
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # Create background tasks
    async def run_and_save(batch_idx, batch_texts):
        nonlocal completed_count
        idx, emb_arr = await embed_batch_with_retry_async(client, batch_texts, batch_idx, semaphore)
        if emb_arr is not None:
            np.save(os.path.join(temp_output_dir, f"part_{idx}.npy"), emb_arr)
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == num_batches:
                elapsed = time.time() - start_time
                avg_speed = completed_count / elapsed if elapsed > 0 else 0
                eta_sec = (num_batches - completed_count) / avg_speed if avg_speed > 0 else 0
                print(f"[PROGRESS] Completed {completed_count}/{num_batches} batches ({completed_count/num_batches*100:.1f}%). Speed: {avg_speed:.2f} batch/s. ETA: {eta_sec/60:.1f}m")
            return True
        else:
            print(f"[FATAL] Failed to embed batch {batch_idx}. Aborting.")
            return False

    tasks = [run_and_save(idx, batch) for idx, batch in batches_to_run]
    
    results = await asyncio.gather(*tasks)
    
    if not all(results):
        print("[FATAL] One or more batches failed to generate. Assembling aborted.")
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
    asyncio.run(main_async())
