import glob
import chardet

csv_files = glob.glob("*.csv")
gyeonggi_files = [f for f in csv_files if "경기" in f]
csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]

print(f"Detecting encoding for: {csv_file_path}")
with open(csv_file_path, "rb") as f:
    rawdata = f.read(50000)
    result = chardet.detect(rawdata)
    print("Chardet result:", result)
