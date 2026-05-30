import sys
import glob
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

csv_files = glob.glob("*.csv")
gyeonggi_files = [f for f in csv_files if "경기" in f]
csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]

print(f"Searching for popular chain stores in: {csv_file_path}")

target_chains = ["CGV", "맥도날드", "스타벅스", "버거킹", "롯데리아", "맥도널드"]

try:
    # Read the CSV file in chunks and check for name matching
    chunk_size = 100000
    matched_rows = []
    
    # We only read the required columns to be extremely fast and memory-efficient
    cols = ['상호명', '상권업종소분류명', '도로명주소']
    
    for i, chunk in enumerate(pd.read_csv(csv_file_path, usecols=cols, dtype=str, encoding='utf-8', encoding_errors='replace', chunksize=chunk_size), 1):
        # Check if name contains any of the target chains
        for chain in target_chains:
            matches = chunk[chunk['상호명'].astype(str).str.contains(chain, case=False, na=False)]
            if not matches.empty:
                for _, row in matches.head(3).iterrows():
                    matched_rows.append({
                        "chain": chain,
                        "name": row['상호명'],
                        "category": row['상권업종소분류명'],
                        "address": row['도로명주소']
                    })
        print(f"Processed chunk {i}...")
        
    print(f"\n--- Search Results ({len(matched_rows)} matches found) ---")
    df_results = pd.DataFrame(matched_rows)
    if not df_results.empty:
        # Group by chain and print samples
        for chain, group in df_results.groupby("chain"):
            print(f"\n[Chain: {chain}] (Total sample matches: {len(group)})")
            print(group[["name", "category", "address"]].head(5).to_string(index=False))
    else:
        print("No matches found for any target chains.")
except Exception as e:
    print("Error:", e)
