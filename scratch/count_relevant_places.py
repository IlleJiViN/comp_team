from sqlalchemy import create_engine, text
import pandas as pd

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

THEME_MAP = {
    "cafe_study": ["카페", "독서실/스터디 카페", "빵/도넛", "토스트/샌드위치/샐러드", "아이스크림/빙수", "서점"],
    "gaming": ["PC방", "전자 게임장", "바둑/장기/체스 경기 운영업"],
    "music_practice": ["노래방", "음악학원", "합주실", "연습실"],
    "fast_food": ["버거", "피자", "치킨", "토스트/샌드위치/샐러드", "그 외 기타 간이 음식점"],
    "korean_food": [
        "백반/한정식", "국/탕/찌개류", "국수/칼국수", "김밥/분식", "분식", "김밥/만두/분식", "닭/오리고기 구이/찜", 
        "돼지고기 구이/찜", "소고기 구이/찜", "곱창 전골/구이", "족발/보쌈", "해산물 구이/찜", 
        "횟집", "냉면/밀면", "전/부침개", "기타 한식 음식점", "떡/한과", "정육점"
    ],
    "foreign_food": [
        "중국집", "파스타/스테이크", "일식 면 요리", "일식 카레/돈가스/덮밥", "일식 회/초밥", 
        "기타 서양식 음식점", "기타 일식 음식점", "베트남식 전문", "기타 동남아식 전문", 
        "분류 안된 외국식 음식점", "뷔페"
    ],
    "nightlife": ["요리 주점", "생맥주 전문", "일반 유흥 주점", "무도 유흥 주점", "주류 소매업"],
    "sports_fitness": [
        "헬스장", "요가/필라테스 학원", "체형/비만 관리", "종합 스포츠시설", 
        "당구장", "탁구장", "볼링장", "수영장", "골프 연습장", "테니스장"
    ]
}

try:
    engine = create_engine(DATABASE_URL)
    
    # Let's count how many total places in the entire 671,650 Gyeonggi-do dataset are matching these categories!
    # Since category in PostgreSQL contains the subcategory name, we can do a query.
    all_subcategories = []
    for subcats in THEME_MAP.values():
        all_subcategories.extend(subcats)
        
    print(f"Total unique subcategories we care about: {len(all_subcategories)}")
    
    with engine.connect() as conn:
        # We query the places table to see how many rows match these subcategories
        sql = text("""
            SELECT category, count(*) as count
            FROM places
            WHERE category IN :subcats
            GROUP BY category
            ORDER BY count DESC
        """)
        
        df = pd.read_sql(sql, conn, params={"subcats": tuple(all_subcategories)})
        
        print("\n--- Relevant Places by Subcategory in DB ---")
        print(df)
        print(f"\nTotal Relevant Places: {df['count'].sum()}")
        
        # Let's also check how many are already embedded for V3
        embedded_count = conn.execute(text("""
            SELECT count(*) 
            FROM places 
            WHERE embedding_vector_v3 IS NOT NULL AND category IN :subcats
        """), {"subcats": tuple(all_subcategories)}).scalar()
        print(f"Total Embedded Relevant Places: {embedded_count}")
        
except Exception as e:
    print("Error:", e)
