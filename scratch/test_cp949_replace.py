import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')
csv_files = glob.glob("*.csv")
gyeonggi_files = [f for f in csv_files if "경기" in f]
csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]

print(f"Reading headers from: {csv_file_path}")
with open(csv_file_path, "r", encoding="cp949", errors="replace") as f:
    headers_line = f.readline()
    sample_line = f.readline()
    
headers = [h.strip().replace('"', '') for h in headers_line.split(",")]
sample = [s.strip().replace('"', '') for s in sample_line.split(",")]

print("Decoded Headers (CP949 with replace):")
for i, (h, s) in enumerate(zip(headers, sample)):
    print(f"  {i}: {h} -> Sample: {s}")
