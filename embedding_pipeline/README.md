# SpotSync Rich Embedding Worker (GPU)

이 디렉토리는 18만 개의 장소 + 14만 개의 리뷰 텍스트를 하나로 뭉쳐서(Sentence Transformer) 벡터 임베딩을 생성하는 GPU 전용 스크립트를 포함하고 있습니다.
CPU 환경에서는 1시간 이상 소요되므로, GPU가 탑재된 PC에서 실행하는 것을 권장합니다.

## 1. 사전 준비 (Prerequisites)

1. **Python 3.9+** 설치 확인
2. 해당 디렉토리에서 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```
3. DB 연결 정보를 담은 `.env` 파일이 상위 폴더(또는 이 폴더)에 있어야 합니다.
   * `DATABASE_URL=postgresql://postgres:password@host:5432/spotsync`
   * 만약 DB 서버가 외부에 있다면 해당 주소로 설정하세요.

## 2. 임베딩 실행

터미널에서 아래 명령어를 실행하세요.

```bash
python gpu_embedding_worker.py
```

### 실행 과정
1. DB에 연결하여 전체 184,080개의 장소와 수집된 리뷰 텍스트를 불러옵니다.
2. 데이터를 다음 형식으로 조합합니다:
   `"장소명: ... | 카테고리: ... | 주소: ... | 특징 및 리뷰: ..."`
3. GPU(또는 MPS/CPU)를 사용해 `jhgan/ko-sroberta-multitask` 모델로 벡터화합니다.
4. 생성이 완료되면 `rich_place_embeddings.pt` 파일이 생성됩니다.

## 3. 결과물 전달

생성된 `rich_place_embeddings.pt` 파일(수백 MB)을 서버로 옮기거나, 메인 브랜치 작업자에게 전달해 주세요.
(서버가 시작될 때 해당 파일을 로드하여 벡터 검색에 사용하게 됩니다.)
