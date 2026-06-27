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
}
all_aliases = list(LOCATION_ALIASES.keys())

ATTRIBUTES = [
    "조용한", "분위기 좋은", "넓은", "깨끗한", "저렴한", "가성비 좋은",
    "혼밥", "혼술", "데이트", "단체", "모임", "회식",
    "맛있는", "유명한", "인기 많은", "숨겨진", "로컬",
    "24시간", "늦게까지 하는", "새벽",
    "아늑한", "감성", "레트로", "모던한", "힙한",
]

SUFFIXES = [
    "", " 어디야", " 추천", " 추천해줘", " 알려줘", " 찾아줘",
    " 어디 있어", " 가고 싶어", " 위치", " 근처",
    " 괜찮은 곳", " 맛집", " 갈만한 데",
    " 있어?", " 알아?", " 좀", " 부탁",
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

sampled_places = random.sample(parsed_places, min(30000, len(parsed_places)))

for row in sampled_places:
    brand = row["brand"]
    dong = row["dong"]
    sigungu = row["sigungu"]
    cat = row["cat"]
    
    if not brand or not dong or not cat:
        continue
    
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
    
    if random.random() < 0.3:
        attr = random.choice(ATTRIBUTES)
        suffix = random.choice(SUFFIXES)
        text = f"{attr} {cat} {dong}{suffix}"
        entities = [
            (0, len(attr), "ATTR"),
            (len(attr)+1, len(attr)+1+len(cat), "CAT"),
            (len(attr)+1+len(cat)+1, len(attr)+1+len(cat)+1+len(dong), "LOC"),
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
    for _ in range(3):
        suffix = random.choice(SUFFIXES)
        text = f"{cat}{suffix}"
        entities = [(0, len(cat), "CAT")]
        add_sample(text, entities)

for loc in random.sample(all_dongs + all_aliases, min(300, len(all_dongs) + len(all_aliases))):
    suffix = random.choice(SUFFIXES)
    text = f"{loc}{suffix}"
    entities = [(0, len(loc), "LOC")]
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
