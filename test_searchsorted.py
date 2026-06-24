# -*- coding: utf-8 -*-
import numpy as np
import time

print("Generating dummy data...")
# 1.1 million ids sorted
all_ids = np.sort(np.random.choice(270000000, 1100000, replace=False))
all_vectors = np.random.randn(1100000, 768).astype(np.float32)

print("Generating query ids...")
query_ids = np.random.choice(all_ids, 10000, replace=False)

start = time.time()
indices = np.searchsorted(all_ids, query_ids)
# Verify (in real code we should verify but here we know they exist)
query_vectors = all_vectors[indices]
print(f"Lookup took {time.time() - start:.5f}s")

# Let's test cosine similarity speed
query_vector = np.random.randn(768).astype(np.float32)

start = time.time()
dot_products = np.dot(query_vectors, query_vector)
print(f"Dot products took {time.time() - start:.5f}s")
