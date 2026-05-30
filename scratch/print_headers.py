import glob

csv_files = glob.glob("*.csv")
gyeonggi_files = [f for f in csv_files if "경기" in f]
csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]

print(f"Reading headers from: {csv_file_path}")
with open(csv_file_path, "r", encoding="utf-8", errors="replace") as f:
    headers_line = f.readline()
    
headers = [h.strip().replace('"', '') for h in headers_line.split(",")]
print("Exact CSV Headers:")
for i, h in enumerate(headers):
    print(f"  {i}: {repr(h)}")
