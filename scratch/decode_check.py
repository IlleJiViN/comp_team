import glob

csv_files = glob.glob("*.csv")
gyeonggi_files = [f for f in csv_files if "경기" in f]
csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]

print(f"Checking encoding for: {csv_file_path}")
with open(csv_file_path, "rb") as f:
    raw = f.read(5000)

with open("scratch/decode_results.txt", "w", encoding="utf-8") as out:
    for enc in ["utf-8", "cp949", "euc-kr", "utf-16", "cp1252"]:
        try:
            decoded = raw.decode(enc)
            out.write(f"=== {enc} Success ===\n")
            out.write(decoded[:800] + "\n\n")
        except Exception as e:
            out.write(f"=== {enc} Failed: {e} ===\n\n")
            
print("Wrote decoding test results to scratch/decode_results.txt.")
