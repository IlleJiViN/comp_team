import json
import random
import re
import pandas as pd
from collections import defaultdict
import os

print("데이터 로딩 중...")
with open("data/places_data.json", "r", encoding="utf-8") as f:
    places_data = json.load(f)

all_dongs = []
all_cats = []

parsed_places = []
for p in places_data:
    name = str(p.get("name", "")).strip()
    cat = str(p.get("category", "")).strip()
    addr = str(p.get("address", "")).strip()
    parts = addr.split()
    sigungu = parts[1] if len(parts) > 1 else ""
    dong = parts[2] if len(parts) > 2 else ""
    
    if name and cat and dong:
        parsed_places.append({
            "brand": name,
            "cat": cat,
            "dong": dong,
            "sigungu": sigungu
        })
        all_dongs.append(dong)
        all_cats.append(cat)

all_dongs = list(set(all_dongs))
all_cats = list(set(all_cats))

LOCATION_ALIASES = {
    "홍대": ["서교동", "연남동", "동교동", "합정동", "상수동"],
    "강남역": ["역삼동", "서초동"],
    "이태원": ["이태원동", "한남동"],
    "종로": ["관철동", "종로1.2.3.4가동", "인사동"],
    "신촌": ["신촌동", "창천동"],
    "대학로": ["혜화동", "이화동"],
    "북촌": ["가회동", "삼청동"],
    "광화문": ["세종로"],
    "명동": ["명동", "회현동"],
    "동대문": ["신설동", "이문동"],
    "신림": ["신림동"],
    "사당": ["사당동"],
    "건대": ["화양동", "자양동"],
    "성수": ["성수동1가", "성수동2가", "성수동"],
    "샤로수길": ["봉천동", "낙성대동"],
    "가로수길": ["신사동", "압구정동"]
}
all_aliases = list(LOCATION_ALIASES.keys())

SLANG_CATEGORIES = {
    "PC방": ["피방", "피시방", "피씨방", "피시빵", "겜방", "피카", "피씨"],
    "양식": ["스파게티집", "파스타집", "피자집", "레스토랑", "양식당"],
    "카페": ["커피숍", "갬성카페", "디저트카페", "카공카페", "커피집", "디저트가게", "베이커리", "빵집"],
    "한식": ["밥집", "백반집", "한식당", "국밥집", "해장국집", "김천", "분식집"],
    "일식": ["초밥집", "스시집", "돈까스집", "일식당", "이자카야", "텐동집", "라멘집"],
    "중식": ["중국집", "짜장면집", "짬뽕집", "마라탕집", "마라샹궈집", "양꼬치집"],
    "치킨": ["치킨집", "닭집", "통닭집", "치맥", "닭강정집"],
    "주점": ["술집", "호프집", "포차", "포장마차", "바", "칵테일바", "라운지", "감주", "헌팅포차", "헌포", "루프탑"],
    "고기/구이": ["고기집", "고깃집", "삼겹살집", "돼지고기집", "소고기집", "한우집", "갈비집", "막창집", "곱창집", "특수부위집"],
    "횟집": ["회센터", "초장집", "해산물집", "스시야"],
    "미용실": ["헤어샵", "바버샵", "머리방", "미장원"],
    "당구장": ["당구장", "포켓볼장", "다마"],
    "노래방": ["코노", "동전노래방", "코인노래방", "노래연습장", "룸술집"]
}

ATTRIBUTES = [
    "조용한", "분위기 좋은", "넓은", "깨끗한", "저렴한", "가성비 좋은",
    "혼밥", "혼술", "데이트", "단체", "모임", "회식",
    "맛있는", "유명한", "인기 많은", "숨겨진", "로컬",
    "24시간", "늦게까지 하는", "새벽",
    "아늑한", "감성", "레트로", "모던한", "힙한",
    # 슬랭/신조어 대거 추가
    "개맛도리", "맛도리", "존맛", "존맛탱", "JMT", "개존맛", "짱맛", "핵존맛",
    "야르한", "미친", "폼미친", "폼미쳤다", "찢었다", "레전드", "갓성비",
    "인스타감성", "인스타용", "사진맛집", "뷰맛집", "햇살맛집", "디저트맛집",
    "핫플", "핫플레이스", "웨이팅", "오픈런", "나만아는", "단골", "노포감성",
    "야장", "루프탑", "테라스", "애견동반", "주차가능", "콜키지프리", "혼밥하기좋은",
    "가성비갑", "킹성비", "혜자", "창렬아닌", "분위기깡패", "조명맛집", "음악맛집",
    "조용한곳", "시끌벅적한", "대화하기좋은", "소개팅", "썸남이랑", "썸녀랑",
    "불금", "불토", "낮술", "혼술러", "낮술하기좋은", "밤샘", "2차", "3차", "막차"
]

SUFFIXES = [
    "", " 어디야", " 추천", " 추천해줘", " 알려줘", " 찾아줘",
    " 어디 있어", " 가고 싶어", " 위치", " 근처", " 주변", " 가는길",
    " 괜찮은 곳", " 맛집", " 갈만한 데", " 핫플",
    " 있어?", " 알아?", " 좀", " 부탁", " 검색", " 검색해봐",
    " 갈래", " 가자", " 예약", " 예약해줘", " 자리있어?"
]

def make_bio_tags(text, entities):
    tokens = []
    current = 0
    for match in re.finditer(r'\S+', text):
        token = match.group()
        start = match.start()
        end = match.end()
        
        tag = "O"
        for e_start, e_end, e_tag in entities:
            if start >= e_start and end <= e_end:
                if start == e_start:
                    tag = f"B-{e_tag}"
                else:
                    tag = f"I-{e_tag}"
                break
        
        tokens.append((token, tag))
    
    return tokens

print("학습 데이터 합성 중...")
samples = []
def add_sample(text, entities):
    tagged = make_bio_tags(text, entities)
    if tagged:
        samples.append({
            "tokens": [t[0] for t in tagged],
            "tags": [t[1] for t in tagged],
        })

sampled_places = random.sample(parsed_places, min(50000, len(parsed_places)))

def get_slang_category(cat):
    for formal, slangs in SLANG_CATEGORIES.items():
        if formal in cat or cat in formal:
            if random.random() < 0.6:  # 60% 확률로 슬랭 사용
                return random.choice(slangs)
    return cat

for row in sampled_places:
    brand = row["brand"]
    dong = row["dong"]
    sigungu = row["sigungu"]
    cat = row["cat"]
    
    if not brand or not dong or not cat:
        continue
        
    cat = get_slang_category(cat)
    
    matching_aliases = [alias for alias, dongs in LOCATION_ALIASES.items() if dong in dongs]
    
    suffix = random.choice(SUFFIXES)
    text = f"{dong} {brand}{suffix}"
    entities = [(0, len(dong), "LOC"), (len(dong)+1, len(dong)+1+len(brand), "BRAND")]
    add_sample(text, entities)
    
    if matching_aliases:
        alias = random.choice(matching_aliases)
        suffix = random.choice(SUFFIXES)
        text = f"{alias} {brand}{suffix}"
        entities = [(0, len(alias), "LOC"), (len(alias)+1, len(alias)+1+len(brand), "BRAND")]
        add_sample(text, entities)
    
    suffix = random.choice(SUFFIXES)
    text = f"{dong} {cat}{suffix}"
    entities = [(0, len(dong), "LOC"), (len(dong)+1, len(dong)+1+len(cat), "CAT")]
    add_sample(text, entities)
    
    suffix = random.choice(SUFFIXES)
    text = f"{sigungu} {cat}{suffix}"
    entities = [(0, len(sigungu), "LOC"), (len(sigungu)+1, len(sigungu)+1+len(cat), "CAT")]
    add_sample(text, entities)
    
    if random.random() < 0.5:  # 속성 추가 확률 증가
        attr = random.choice(ATTRIBUTES)
        suffix = random.choice(SUFFIXES)
        text = f"{attr} {cat} {dong}{suffix}"
        entities = [
            (0, len(attr), "ATTR"),
            (len(attr)+1, len(attr)+1+len(cat), "CAT"),
            (len(attr)+1+len(cat)+1, len(attr)+1+len(cat)+1+len(dong), "LOC"),
        ]
        add_sample(text, entities)
        
        # 순서가 반대인 경우 (홍대 개맛도리 고기집)
        loc_term = random.choice(matching_aliases) if matching_aliases and random.random() < 0.5 else dong
        text = f"{loc_term} {attr} {cat}{suffix}"
        entities = [
            (0, len(loc_term), "LOC"),
            (len(loc_term)+1, len(loc_term)+1+len(attr), "ATTR"),
            (len(loc_term)+1+len(attr)+1, len(loc_term)+1+len(attr)+1+len(cat), "CAT"),
        ]
        add_sample(text, entities)
    
    if random.random() < 0.2:
        suffix = random.choice(SUFFIXES)
        text = f"{brand}{suffix}"
        entities = [(0, len(brand), "BRAND")]
        add_sample(text, entities)
    
    if matching_aliases and random.random() < 0.3:
        alias = random.choice(matching_aliases)
        attr = random.choice(ATTRIBUTES)
        suffix = random.choice(SUFFIXES)
        text = f"{alias} {attr} {cat}{suffix}"
        entities = [
            (0, len(alias), "LOC"),
            (len(alias)+1, len(alias)+1+len(attr), "ATTR"),
            (len(alias)+1+len(attr)+1, len(alias)+1+len(attr)+1+len(cat), "CAT"),
        ]
        add_sample(text, entities)
    
    if random.random() < 0.2:
        text = f"{cat} {dong} 근처"
        entities = [
            (0, len(cat), "CAT"),
            (len(cat)+1, len(cat)+1+len(dong), "LOC"),
        ]
        add_sample(text, entities)
    
    if random.random() < 0.15:
        text = f"{brand} {sigungu}에 있어?"
        entities = [
            (0, len(brand), "BRAND"),
            (len(brand)+1, len(brand)+1+len(sigungu), "LOC"),
        ]
        add_sample(text, entities)

for cat in random.sample(all_cats, min(500, len(all_cats))):
    cat_slang = get_slang_category(cat)
    for _ in range(3):
        suffix = random.choice(SUFFIXES)
        text = f"{cat_slang}{suffix}"
        entities = [(0, len(cat_slang), "CAT")]
        add_sample(text, entities)

for loc in random.sample(all_dongs + all_aliases, min(300, len(all_dongs) + len(all_aliases))):
    suffix = random.choice(SUFFIXES)
    text = f"{loc}{suffix}"
    entities = [(0, len(loc), "LOC")]
    add_sample(text, entities)

# 하드코어 슬랭 명시적 테스트셋 추가 주입 (학습 데이터)
hardcore_samples = [
    ("신촌 피시빵", [("신촌", "LOC"), ("피시빵", "CAT")]),
    ("홍대 개맛도리 고기집", [("홍대", "LOC"), ("개맛도리", "ATTR"), ("고기집", "CAT")]),
    ("연남동 야르한 스파게티", [("연남동", "LOC"), ("야르한", "ATTR"), ("스파게티", "CAT")]),
    ("개맛도리 횟집 찾아줘", [("개맛도리", "ATTR"), ("횟집", "CAT"), ("찾아줘", "O")]),
    ("종로 피씨방 추천", [("종로", "LOC"), ("피씨방", "CAT"), ("추천", "O")]),
    ("피방 어디있어", [("피방", "CAT"), ("어디있어", "O")])
]

for text, entity_tuples in hardcore_samples:
    entities = []
    idx = 0
    for word, tag in entity_tuples:
        idx = text.find(word, idx)
        if idx != -1 and tag != "O":
            entities.append((idx, idx + len(word), tag))
            idx += len(word)
    add_sample(text, entities)

print(f"생성 완료: {len(samples):,}개 학습 샘플")

random.shuffle(samples)
split_idx = int(len(samples) * 0.9)
train_data = samples[:split_idx]
val_data = samples[split_idx:]

output = {
    "label_list": ["O", "B-LOC", "I-LOC", "B-BRAND", "I-BRAND", "B-CAT", "I-CAT", "B-ATTR", "I-ATTR"],
    "train": train_data,
    "validation": val_data,
}

os.makedirs("data", exist_ok=True)
with open("data/ner_training_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print(f"저장 완료: data/ner_training_data.json")
