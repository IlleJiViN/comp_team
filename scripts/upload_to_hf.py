# -*- coding: utf-8 -*-
"""
SpotSync NER 모델을 Hugging Face Hub에 업로드하는 스크립트
- ille255/spotsync-ner       : 원본 SafeTensors 모델 (BERT 기반)
- ille255/spotsync-ner-onnx  : ONNX 양자화 경량 모델
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from huggingface_hub import HfApi, create_repo
import os

HF_USERNAME = "ille255"
MODEL_NAME_NER = "spotsync-ner"
MODEL_NAME_ONNX = "spotsync-ner-onnx"
MODEL_NAME_BGE = "bge-m3-onnx"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NER_DIR = os.path.join(BASE_DIR, "models", "spotsync-ner")
ONNX_DIR = os.path.join(BASE_DIR, "models", "spotsync-ner-onnx")
BGE_DIR = os.path.join(BASE_DIR, "models", "bge-m3-onnx")

api = HfApi()

README_NER = """\
---
language:
- ko
license: apache-2.0
tags:
- ner
- token-classification
- korean
- place-search
- bert
library_name: transformers
pipeline_tag: token-classification
---

# SpotSync NER - Korean Place Search NER Model

Korean Named Entity Recognition model used in the **SpotSync** intelligent place search pipeline.
Extracts **Location (LOC)**, **Brand (BRAND)**, **Category (CAT)**, and **Attribute (ATTR)** from natural language queries.

## Labels

| Label | Description | Example |
|-------|-------------|---------|
| `B-LOC` / `I-LOC` | Location/region | Hongdae, Gangnam, Sinchon Station |
| `B-BRAND` / `I-BRAND` | Brand/store name | Starbucks, McDonald's |
| `B-CAT` / `I-CAT` | Business category | Cafe, Restaurant, Gym |
| `B-ATTR` / `I-ATTR` | Attribute/characteristic | Quiet, Good atmosphere, 24 hours |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_id = "ille255/spotsync-ner"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

text = "Hongdae nearby cozy cafe"
tokens = text.split()
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=2)[0]
id2label = model.config.id2label
for token, pred in zip(tokens, predictions[1:-1]):
    print(f"{token}: {id2label[pred.item()]}")
```

## Model Architecture

- **Base**: BERT (BertForTokenClassification)
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Layers**: 12
- **Vocab Size**: 32,000

## Related Links

- ONNX Quantized version: [ille255/spotsync-ner-onnx](https://huggingface.co/ille255/spotsync-ner-onnx)
- GitHub: [SpotSync Project](https://github.com/IlleJiViN/comp_team)
"""

README_ONNX = """\
---
language:
- ko
license: apache-2.0
tags:
- ner
- token-classification
- korean
- place-search
- onnx
- quantized
library_name: transformers
pipeline_tag: token-classification
base_model: ille255/spotsync-ner
---

# SpotSync NER ONNX - Quantized Lightweight Model

ONNX + INT8 quantized version of [ille255/spotsync-ner](https://huggingface.co/ille255/spotsync-ner).
Approximately **4x smaller** (440MB -> 111MB) with fast inference on CPU.

## Files

| File | Description |
|------|-------------|
| `model.onnx` | Full ONNX model (~440MB) |
| `model_quantized.onnx` | INT8 quantized model (~111MB, **recommended**) |
| `tokenizer.json` | Tokenizer |
| `label_config.json` | NER label list |

## Usage (with optimum)

```python
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer
import torch

model_id = "ille255/spotsync-ner-onnx"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForTokenClassification.from_pretrained(
    model_id, file_name="model_quantized.onnx"
)

text = "Gangnam station nearby quiet study cafe"
tokens = text.split()
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                   truncation=True, max_length=64)

outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)[0]

label_list = ["O","B-LOC","I-LOC","B-BRAND","I-BRAND","B-CAT","I-CAT","B-ATTR","I-ATTR"]
for token, pred in zip(tokens, predictions[1:1+len(tokens)]):
    print(f"{token}: {label_list[pred.item()]}")
```

## Performance

| | Original | Quantized |
|--|--|--|
| File Size | 440MB | **111MB** |
| Environment | GPU/CPU | **CPU optimized** |

## Related Links

- Original model: [ille255/spotsync-ner](https://huggingface.co/ille255/spotsync-ner)
- GitHub: [SpotSync Project](https://github.com/IlleJiViN/comp_team)
"""

README_BGE = """\
---
language:
- en
license: apache-2.0
tags:
- embedding
- retrieval
- onnx
- quantized
library_name: sentence-transformers
pipeline_tag: feature-extraction
---

# BGE-M3 ONNX Embedding Model

Optimized ONNX version of the BGE-M3 embedding model from **bge-m3** (large multilingual embedding).
Provides fast, CPU-friendly inference for semantic search and retrieval.

## Files

| File | Description |
|------|-------------|
| `model_quantized.onnx` | Quantized ONNX model (~570 MB) |
| `tokenizer.json` | Tokenizer vocabulary |
| `config.json` | Model config |
| `ort_config.json` | ONNX Runtime config |

## Usage (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("ille255/bge-m3-onnx")
emb = model.encode(["example sentence"])
```

## Performance
- **Size**: ~570 MB (quantized)
- **Speed**: ~10-20 ms per sentence on CPU

## License
Apache-2.0
"""


def upload_model(repo_id, local_dir, readme_content, description):
    print(f"\n{'='*60}")
    print(f"[UPLOAD] Starting: {repo_id}")
    print(f"{'='*60}")

    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"[OK] Repository ready: {repo_id}")
    except Exception as e:
        print(f"[WARN] Repo creation: {e}")

    print("[INFO] Uploading README.md ...")
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card"
    )
    print("[OK] README.md uploaded")

    files = os.listdir(local_dir)
    print(f"\n[INFO] Files to upload ({len(files)} total):")
    for f in files:
        size_mb = os.path.getsize(os.path.join(local_dir, f)) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")

    print(f"\n[INFO] Uploading all files (large files may take a while)...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {description}",
        ignore_patterns=["*.tmp"]
    )
    print(f"[OK] All files uploaded!")
    print(f"[URL] https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    # 1. Original NER model (SafeTensors)
    upload_model(
        repo_id=f"{HF_USERNAME}/{MODEL_NAME_NER}",
        local_dir=NER_DIR,
        readme_content=README_NER,
        description="SpotSync NER model (SafeTensors)"
    )

    # 2. ONNX quantized model
    upload_model(
        repo_id=f"{HF_USERNAME}/{MODEL_NAME_ONNX}",
        local_dir=ONNX_DIR,
        readme_content=README_ONNX,
        description="SpotSync NER ONNX quantized model"
    )
    # 3. BGE-M3 ONNX embedding model
    upload_model(
        repo_id=f"{HF_USERNAME}/{MODEL_NAME_BGE}",
        local_dir=BGE_DIR,
        readme_content=README_BGE,
        description="BGE-M3 ONNX quantized embedding model"
    )

    print(f"\n{'='*60}")
    print("[DONE] All models uploaded successfully!")
    print(f"  - https://huggingface.co/{HF_USERNAME}/{MODEL_NAME_NER}")
    print(f"  - https://huggingface.co/{HF_USERNAME}/{MODEL_NAME_ONNX}")
    print(f"  - https://huggingface.co/{HF_USERNAME}/{MODEL_NAME_BGE}")
    print(f"{'='*60}")
