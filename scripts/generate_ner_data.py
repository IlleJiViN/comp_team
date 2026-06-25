"""
소상공인 CSV 데이터로부터 NER 학습 데이터를 자동 합성하는 스크립트.
출력: ner_training_data.json (Colab 업로드용)

태그 체계:
  B-LOC / I-LOC   : 지역 (행정동, 시군구, 별칭)
  B-BRAND / I-BRAND: 상호명/브랜드
  B-CAT / I-CAT    : 업종 카테고리
  B-ATTR / I-ATTR  : 속성/분위기
  O                : 기타
"""

import pandas as pd
import json
import random
import re
from collections import defaultdict

random.seed(42)

print("[1/5] 소상공인 CSV 로딩...")
df = pd.read_csv(
    "data/소상공인시장진흥공단_상가(상권)정보_서울_202603.csv",
    encoding="utf-8",
    usecols=["상호명", "상권업종소분류명", "상권업종중분류명", "행정동명", "시군구명"],
    low_memory=False,
)
df = df.dropna(subset=["상호명", "행정동명", "상권업종소분류명"])
print(f"  → 로드 완료: {len(df):,}건")

# ============================================================
# 사전 구축
# ============================================================
print("[2/5] 엔티티 사전 구축...")

# 상호명 사전 (출현 빈도 상위 5000개 + 랜덤 5000개)
brand_counts = df["상호명"].value_counts()
top_brands = list(brand_counts.head(5000).index)
rare_brands = list(brand_counts.tail(max(0, len(brand_counts) - 5000)).sample(min(5000, len(brand_counts) - 5000), random_state=42).index)
all_brands = list(set(top_brands + rare_brands))
print(f"  → 상호명: {len(all_brands):,}개")

# 행정동 사전
all_dongs = list(df["행정동명"].unique())
print(f"  → 행정동: {len(all_dongs)}개")

# 시군구 사전
all_sigungus = list(df["시군구명"].unique())
print(f"  → 시군구: {len(all_sigungus)}개")

# 업종 소분류 사전
all_cats = list(df["상권업종소분류명"].unique())
print(f"  → 업종 소분류: {len(all_cats)}개")

# 업종 중분류 사전
all_mid_cats = list(df["상권업종중분류명"].dropna().unique())
print(f"  → 업종 중분류: {len(all_mid_cats)}개")

# 지역 별칭 매핑 (별칭 → 실제 행정동 리스트)
LOCATION_ALIASES = {
    "홍대": ["서교동", "동교동"],
    "합정": ["합정동"],
    "망원": ["망원동"],
    "연남": ["연남동"],
    "상수": ["상수동"],
    "신촌": ["노고산동", "대흥동", "창천동"],
    "공덕": ["공덕동"],
    "마포": ["도화동", "마포동"],
    "상암": ["상암동"],
    "이태원": ["이태원동", "한남동"],
    "강남": ["역삼동", "논현동", "삼성동"],
    "건대": ["화양동", "자양동"],
    "성수": ["성수동1가", "성수동2가"],
    "을지로": ["을지로동", "입정동"],
    "종로": ["종로1·2·3·4가동", "종로5·6가동"],
    "압구정": ["압구정동", "신사동"],
    "잠실": ["잠실동", "잠실본동"],
    "여의도": ["여의도동"],
    "용산": ["용산동", "한강로동"],
    "혜화": ["혜화동"],
    "대학로": ["혜화동", "이화동"],
    "북촌": ["가회동", "삼청동"],
    "광화문": ["세종로"],
    "명동": ["명동", "회현동"],
    "동대문": ["신설동", "이문동"],
    "신림": ["신림동"],
    "사당": ["사당동"],
}
all_aliases = list(LOCATION_ALIASES.keys())

# 분위기/속성 사전
ATTRIBUTES = [
    "조용한", "분위기 좋은", "넓은", "깨끗한", "저렴한", "가성비 좋은",
    "혼밥", "혼술", "데이트", "단체", "모임", "회식",
    "맛있는", "유명한", "인기 많은", "숨겨진", "로컬",
    "24시간", "늦게까지 하는", "새벽",
    "아늑한", "감성", "레트로", "모던한", "힙한",
]

# 후행 표현 사전 (O 태그)
SUFFIXES = [
    "", " 어디야", " 추천", " 추천해줘", " 알려줘", " 찾아줘",
    " 어디 있어", " 가고 싶어", " 위치", " 근처",
    " 괜찮은 곳", " 맛집", " 갈만한 데",
    " 있어?", " 알아?", " 좀", " 부탁",
]

# ============================================================
# BIO 태깅 함수
# ============================================================
def make_bio_tags(text, entities):
    """
    text: 전체 문장
    entities: [(start, end, tag), ...] — 문자 인덱스 기반
    
    Returns: [(token, tag), ...]
    """
    # 공백 기준 토큰화
    tokens = []
    current = 0
    for match in re.finditer(r'\S+', text):
        token = match.group()
        start = match.start()
        end = match.end()
        
        # 이 토큰에 해당하는 엔티티 태그 찾기
        tag = "O"
        for e_start, e_end, e_tag in entities:
            # 토큰이 엔티티 범위 안에 있는지 확인
            if start >= e_start and end <= e_end:
                if start == e_start:
                    tag = f"B-{e_tag}"
                else:
                    tag = f"I-{e_tag}"
                break
        
        tokens.append((token, tag))
    
    return tokens


# ============================================================
# 쿼리 합성 템플릿
# ============================================================
print("[3/5] 학습 데이터 합성 중...")

samples = []

def add_sample(text, entities):
    tagged = make_bio_tags(text, entities)
    if tagged:
        samples.append({
            "tokens": [t[0] for t in tagged],
            "tags": [t[1] for t in tagged],
        })


# 행 샘플링 (전체 53만건 중 충분한 양 사용)
sampled_df = df.sample(min(50000, len(df)), random_state=42)

for _, row in sampled_df.iterrows():
    brand = str(row["상호명"]).strip()
    dong = str(row["행정동명"]).strip()
    sigungu = str(row["시군구명"]).strip()
    cat = str(row["상권업종소분류명"]).strip()
    mid_cat = str(row["상권업종중분류명"]).strip() if pd.notna(row.get("상권업종중분류명")) else ""
    
    if not brand or not dong or not cat:
        continue
    
    # 이 행에 매칭되는 별칭 찾기
    matching_aliases = [alias for alias, dongs in LOCATION_ALIASES.items() if dong in dongs]
    
    # --- 템플릿 1: "{동} {상호명}" ---
    suffix = random.choice(SUFFIXES)
    text = f"{dong} {brand}{suffix}"
    entities = [(0, len(dong), "LOC"), (len(dong)+1, len(dong)+1+len(brand), "BRAND")]
    add_sample(text, entities)
    
    # --- 템플릿 2: "{별칭} {상호명}" (별칭이 있을 때) ---
    if matching_aliases:
        alias = random.choice(matching_aliases)
        suffix = random.choice(SUFFIXES)
        text = f"{alias} {brand}{suffix}"
        entities = [(0, len(alias), "LOC"), (len(alias)+1, len(alias)+1+len(brand), "BRAND")]
        add_sample(text, entities)
    
    # --- 템플릿 3: "{동} {업종}" ---
    suffix = random.choice(SUFFIXES)
    text = f"{dong} {cat}{suffix}"
    entities = [(0, len(dong), "LOC"), (len(dong)+1, len(dong)+1+len(cat), "CAT")]
    add_sample(text, entities)
    
    # --- 템플릿 4: "{시군구} {업종}" ---
    suffix = random.choice(SUFFIXES)
    text = f"{sigungu} {cat}{suffix}"
    entities = [(0, len(sigungu), "LOC"), (len(sigungu)+1, len(sigungu)+1+len(cat), "CAT")]
    add_sample(text, entities)
    
    # --- 템플릿 5: "{속성} {업종} {동}" ---
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
    
    # --- 템플릿 6: "{상호명}" (상호명만) ---
    if random.random() < 0.2:
        suffix = random.choice(SUFFIXES)
        text = f"{brand}{suffix}"
        entities = [(0, len(brand), "BRAND")]
        add_sample(text, entities)
    
    # --- 템플릿 7: "{별칭} {속성} {업종}" ---
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
    
    # --- 템플릿 8: "{업종} {동} 근처" ---
    if random.random() < 0.2:
        text = f"{cat} {dong} 근처"
        entities = [
            (0, len(cat), "CAT"),
            (len(cat)+1, len(cat)+1+len(dong), "LOC"),
        ]
        add_sample(text, entities)
    
    # --- 템플릿 9: "{상호명} {시군구}에 있어?" ---
    if random.random() < 0.15:
        text = f"{brand} {sigungu}에 있어?"
        entities = [
            (0, len(brand), "BRAND"),
            (len(brand)+1, len(brand)+1+len(sigungu), "LOC"),
        ]
        add_sample(text, entities)

# --- 추가: 순수 카테고리만 검색하는 패턴 ---
for cat in random.sample(all_cats, min(500, len(all_cats))):
    for _ in range(3):
        suffix = random.choice(SUFFIXES)
        text = f"{cat}{suffix}"
        entities = [(0, len(cat), "CAT")]
        add_sample(text, entities)

# --- 추가: 순수 지역만 검색하는 패턴 ---
for loc in random.sample(all_dongs + all_aliases, min(300, len(all_dongs) + len(all_aliases))):
    suffix = random.choice(SUFFIXES)
    text = f"{loc}{suffix}"
    entities = [(0, len(loc), "LOC")]
    add_sample(text, entities)

print(f"  → 생성 완료: {len(samples):,}개 학습 샘플")

# ============================================================
# 태그 분포 확인
# ============================================================
print("[4/5] 태그 분포 확인...")
tag_counts = defaultdict(int)
for s in samples:
    for tag in s["tags"]:
        tag_counts[tag] += 1

for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
    print(f"  {tag}: {count:,}")

# ============================================================
# 저장
# ============================================================
print("[5/5] 저장 중...")

# Train / Validation 분할 (90:10)
random.shuffle(samples)
split_idx = int(len(samples) * 0.9)
train_data = samples[:split_idx]
val_data = samples[split_idx:]

output = {
    "label_list": ["O", "B-LOC", "I-LOC", "B-BRAND", "I-BRAND", "B-CAT", "I-CAT", "B-ATTR", "I-ATTR"],
    "train": train_data,
    "validation": val_data,
}

with open("data/ner_training_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

size_mb = len(json.dumps(output, ensure_ascii=False).encode("utf-8")) / 1024 / 1024
print(f"\n✅ 저장 완료: data/ner_training_data.json ({size_mb:.1f} MB)")
print(f"   Train: {len(train_data):,}개")
print(f"   Validation: {len(val_data):,}개")
print(f"   Labels: {output['label_list']}")
