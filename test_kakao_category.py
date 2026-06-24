import requests, os
from dotenv import load_dotenv
load_dotenv()

res = requests.get(
    'https://dapi.kakao.com/v2/local/search/keyword.json', 
    headers={'Authorization': f'KakaoAK {os.getenv("KAKAO_API_KEY")}'}, 
    params={'query': '부산역 국밥집', 'size': 1}
).json()

cat = res["documents"][0]["category_group_name"]
with open("cat_out.txt", "w", encoding="utf-8") as f:
    f.write(cat)
