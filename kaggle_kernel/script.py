import pandas as pd
import numpy as np
import os
import gc
import subprocess

# Downgrade PyTorch to support Kaggle P100 GPU (sm_60) which is dropped in torch 2.4+
print("Downgrading PyTorch to 2.2.2 to support P100 GPU (Python 3.12 compatible)...")
subprocess.run(["pip", "install", "torch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)

import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Listing /kaggle/input directory:")
for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        print(os.path.join(root, file))

csv_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        if file.endswith('.csv'):
            csv_path = os.path.join(root, file)
            break

if not csv_path:
    raise FileNotFoundError("CSV not found in /kaggle/input")

print(f"Loading CSV from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} chunks.")

# Enable GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Model on Device: {device}...")
model_path = "/kaggle/input/bge-m3"
# Fallback to downloading if the path doesn't exist
if not os.path.exists(model_path):
    print(f"Local model path {model_path} not found, falling back to HuggingFace BAAI/bge-m3...")
    model_path = "BAAI/bge-m3"

from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
# Use eager attention to avoid CUDA kernel errors on older GPUs (T4/P100)
model = AutoModel.from_pretrained(model_path, local_files_only=True, attn_implementation="eager")
model.to(device)
model.eval()

chunk_size = 50000
batch_size = 64 # Keep it 64 to avoid OOM on Kaggle P100/T4

for i in range(0, len(df), chunk_size):
    file_name = f"embeddings_part_{i//chunk_size + 1}.npz"
    
    df_chunk = df.iloc[i:i+chunk_size]
    texts = df_chunk['text_val'].tolist()
    ids = df_chunk['id'].values
    
    print(f"\n--- Encoding Part {i//chunk_size + 1} (size: {len(texts)}) ---")
    
    part_embeddings = []
    for j in range(0, len(texts), batch_size):
        batch_texts = texts[j:j+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            batch_embeddings = model_output[0][:, 0] # CLS pooling
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
        # Bypass PyTorch's native .numpy() call to avoid Numpy 1.x/2.x ABI crashes
        part_embeddings.append(np.array(batch_embeddings.cpu().tolist(), dtype=np.float32))
        
        if (j // batch_size) % 100 == 0:
            print(f"  Processed {j}/{len(texts)} in part {i//chunk_size + 1}")
            
    embeddings = np.vstack(part_embeddings)
    np.savez_compressed(file_name, embeddings=embeddings, ids=ids)
    print(f"Saved {file_name}")
    
    del embeddings, texts, ids, df_chunk, part_embeddings
    gc.collect()

print("All embeddings generated!")
