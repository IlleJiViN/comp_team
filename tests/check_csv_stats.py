import pandas as pd

df = pd.read_csv("data/소상공인시장진흥공단_상가(상권)정보_서울_202603.csv", nrows=5, encoding="utf-8")
for i, row in df.iterrows():
    print(f"--- Row {i} ---")
    print(f"  상호명: {row['상호명']}")
    print(f"  업종대분류: {row['상권업종대분류명']}")
    print(f"  업종중분류: {row['상권업종중분류명']}")
    print(f"  업종소분류: {row['상권업종소분류명']}")
    print(f"  시군구: {row['시군구명']}")
    print(f"  행정동: {row['행정동명']}")
    print(f"  도로명주소: {row['도로명주소']}")
    print()

# 카운트 확인
print(f"\n전체 서울 CSV 행 수: {len(pd.read_csv('data/소상공인시장진흥공단_상가(상권)정보_서울_202603.csv', encoding='utf-8'))}")

# 업종 소분류 유니크 수
df_full = pd.read_csv("data/소상공인시장진흥공단_상가(상권)정보_서울_202603.csv", encoding="utf-8", usecols=["상권업종소분류명", "행정동명", "시군구명"])
print(f"업종 소분류 유니크 수: {df_full['상권업종소분류명'].nunique()}")
print(f"행정동 유니크 수: {df_full['행정동명'].nunique()}")
print(f"시군구 유니크 수: {df_full['시군구명'].nunique()}")
