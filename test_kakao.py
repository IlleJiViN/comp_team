import requests, os, sys
sys.stdout.reconfigure(encoding='utf-8')
KAKAO_API_KEY = os.getenv('KAKAO_API_KEY')
r = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json', headers={'Authorization': f'KakaoAK {KAKAO_API_KEY}'}, params={'query': 'ÏÎÐí£  ±æ±·Áý', 'size': 3}).json()
print(r)