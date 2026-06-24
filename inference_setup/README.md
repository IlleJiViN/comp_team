# SpotSync Inference Setup

이 디렉토리는 다른 개발자(친구)가 **DB 없이도** 즉시 벡터 데이터를 구축하고 추론 모델 테스트를 진행할 수 있도록 준비된 패키지입니다.

## 파일 준비
먼저 아래 두 파일을 이 디렉토리(`inference_setup/`) 안에 복사해 넣으세요:
1. `places_data.json` (메타데이터, `embedding_pipeline/places_data.json` 복사)
2. `rich_place_embeddings.pt` (1024차원 BGE-M3 임베딩, 약 1GB)

## 데이터베이스 초기화 (PostgreSQL)

PostgreSQL(with `pgvector`)이 설치되어 있어야 합니다. (Docker 등을 이용해 쉽게 띄울 수 있습니다)

```bash
# 필요한 패키지 설치
pip install psycopg2-binary torch numpy
```

```bash
# DB 초기화 스크립트 실행
python init_postgres.py
```
위 스크립트를 실행하면 `spotsync` 데이터베이스 내에 `places` 테이블을 자동으로 생성하고 `id`, `name`, `address` 등의 메타데이터와 `embedding_vector_bge_m3` (1024차원 vector)를 전부 매핑하여 Insert 합니다.

## 요약 (Q&A)
- **Q. 로컬 DB가 아예 없어도 되나요?**
  A. 네, 제공된 스크립트(`init_postgres.py`)가 테이블 생성부터 확장 프로그램(pgvector) 활성화, 메타데이터 연동 및 임베딩 삽입까지 전부 자동으로 수행합니다. 빈 PostgreSQL 서버만 있으면 됩니다!
- **Q. Elasticsearch 도 써야 하나요?**
  A. 만약 ES 기반 RAG를 테스트하고 싶다면 메인 디렉토리의 `index_es_rich.py` 로직을 참고하여 ES에 붓기만 하면 됩니다. 
