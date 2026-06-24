import json
import os
import psycopg2
from psycopg2.extras import DictCursor

# PostgreSQL Database Configuration
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
OUTPUT_JSONL_FILE = "data/spotsync_summaries.jsonl"

# Prompt template for the 5-dimensional standardized summary
SUMMARIZATION_PROMPT_TEMPLATE = """
당신은 대한민국 최고의 공간 및 장소 추천 전문 AI 에이전트입니다.
아래 제공되는 장소의 기본 정보와 방문자/블로그 리뷰들을 분석하여, 지침을 절대적으로 준수해 400~500자 분량의 '단 하나의 정규화된 표준 요약본'을 작성하십시오.

### 🏢 장소 기본 정보
- **장소명**: {place_name}
- **카테고리**: {place_category}
- **지리적 위치(주소)**: {place_address}

### 📋 분석 및 요약 지침
1. **5대 핵심 차원 분석**: 리뷰에서 아래 5가지 요소를 명확히 파악하십시오.
   - 분위기 (Atmosphere)
   - 맛/품질 (Taste/Quality)
   - 가격대 (Price Range)
   - 서비스 (Service)
   - 추천 대상 (Target Audience)
2. **구조적 재구성**: 5대 차원 분석 결과를 바탕으로 줄글 형태의 매끄럽고 매력적인 400~500자 요약본을 작성하십시오. 개별 차원을 나열하는 대신, 하나의 완성도 높은 문맥을 가진 단락으로 병합해야 합니다.
3. **할루시네이션 가드레일**: 오직 제공된 리뷰 정보에만 근거하십시오. 리뷰 데이터가 부족하여 특정 차원을 분석할 수 없는 경우, 소설을 쓰지 말고 해당 요소에 대해 "정보 부족"임을 간략히 명시하십시오. 만약 전체 리뷰 수가 너무 적거나 유의미한 정보가 없다면 요약본을 "정보 부족으로 인한 요약 불가"로 설정하십시오.
4. **절대 편향 금지**: 리뷰 개수가 많든 적든 상관없이, 제공된 텍스트 범위 내에서 장소 고유의 특징(DNA)을 깊이 있게 담아내어 동등한 품질의 요약을 생성해야 합니다.

### 📥 수집된 최신 리뷰 데이터 (출처 및 텍스트)
{reviews}

### 📤 결과물 (400~500자 한글 줄글 요약):
"""

def generate_summary_mock(name, category, reviews):
    """
    Mock function representing the API call to Vertex AI / Google GenAI.
    In the real implementation, this will call Vertex AI Grounded Generation API
    or Gemini API routed via GenAI App Builder.
    """
    # Simple mock check for sparse data
    if not reviews or len(reviews.strip()) < 10:
        return "정보 부족으로 인한 요약 불가"
        
    # Example mock output structure (showing how the 5 dimensions are combined)
    return (
       f"[{name}]은(는) {category} 장소로, 포근하고 아늑한 분위기가 돋보이는 곳입니다. "
       f"방문자들은 특히 이곳의 뛰어난 맛과 재료의 품질에 대해 입을 모아 칭찬합니다. "
       f"가격대는 다소 높은 편이지만 그에 걸맞은 높은 수준의 친절한 서비스와 전문적인 응대가 제공됩니다. "
       f"조용한 대화나 모임, 혹은 특별한 날 데이트를 즐기고자 하는 추천대상에게 안성맞춤인 공간입니다."
    )

def main():
    print("Connecting to local PostgreSQL database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=DictCursor)
    
    # 1. Fetch places that have been enriched with reviews but not yet summarized
    # For now, we will select places that have reviews (description is not empty)
    cur.execute("""
        SELECT id, place_id, name, category, address, description 
        FROM places 
        WHERE address LIKE '서울%' 
          AND is_enriched = TRUE 
          AND description IS NOT NULL 
          AND description != ''
        LIMIT 10;
    """)
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} enriched places to process...")
    
    os.makedirs(os.path.dirname(OUTPUT_JSONL_FILE), exist_ok=True)
    
    # 2. Open JSONL file for writing Vertex AI Search ingestion format
    with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            place_id = row['place_id']
            name = row['name']
            category = row['category']
            address = row['address']
            reviews = row['description']
            
            # Format the prompt
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
                place_name=name,
                place_category=category,
                place_address=address,
                reviews=reviews
            )
            
            print(f"Generating summary for: {name} (ID: {row['id']})...")
            
            # 3. Call the model/API (Mocked here, replace with real SDK call)
            # Example SDK Call setup (Vertex AI):
            #
            # from google.cloud import aiplatform
            # from vertexai.generative_models import GenerativeModel
            # model = GenerativeModel("gemini-1.5-pro")
            # response = model.generate_content(prompt)
            # summary = response.text
            
            summary = generate_summary_mock(name, category, reviews)
            
            # 4. Construct the structured JSON payload for Vertex AI Search Data Store
            record = {
                "id": place_id,
                "title": name,
                "category": category,
                "address": address,
                "normalized_summary": summary,
                "raw_reviews": reviews
            }
            
            # Write to JSONL (one valid JSON object per line)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Processing complete! JSONL output saved to: {OUTPUT_JSONL_FILE}")
    conn.close()

if __name__ == "__main__":
    main()
