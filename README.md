# Gravity (SpotSync AI)

## 📌 프로젝트 설정 가이드 (팀원용)

### 1. 환경 변수 설정
이 저장소를 클론한 후, 가장 먼저 환경 변수 파일을 설정해야 합니다.
`Google API Key` 및 `Kakao API Key`가 필요합니다.

```bash
# .env.example 파일을 복사하여 .env 파일을 생성합니다.
cp .env.example .env
```
생성된 `.env` 파일에 각자의 API 키를 입력하세요.

### 2. 가상환경 세팅 및 패키지 설치
Python 가상환경을 만들고 필요한 라이브러리를 설치합니다.
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 모델 가중치(Models) 다운로드 및 배치
이 프로젝트는 AI 모델 검색을 위해 **로컬 ONNX 모델**을 사용합니다. 모델 가중치 파일은 용량이 크기 때문에 Git 추적에서 제외(`.gitignore`)되어 있습니다. 
서버 실행 전 반드시 아래 모델들을 다운로드하여 위치시켜야 합니다:

1. **다운로드할 모델**:
   - `bge-m3-onnx` (텍스트 임베딩 모델)
   - `spotsync-ner-onnx` (개체명 인식 모델)
2. **배치 경로**:
   - 프로젝트 루트 디렉토리에 `models` 폴더를 생성하고 각각 아래와 같이 폴더째로 배치합니다:
     ```text
     gravity/
     └── models/
         ├── bge-m3-onnx/
         └── spotsync-ner-onnx/
     ```
   - *다운로드 링크:*
     - **Hugging Face 모델 허브**에서 직접 다운로드하거나 테스트할 수 있습니다:
       - 🔗 [SpotSync NER (원본 모델)](https://huggingface.co/ille255/spotsync-ner)
       - 🔗 [SpotSync NER ONNX (경량화 모델)](https://huggingface.co/ille255/spotsync-ner-onnx)
       - 🔗 [BGE-M3 ONNX Embedding (경량화 텍스트 임베딩)](https://huggingface.co/ille255/bge-m3-onnx)

### 4. 데이터베이스 (PostGIS) 복원 가이드
이 프로젝트는 **PostGIS 및 pgvector** 데이터베이스를 사용합니다. 
로컬 데이터베이스를 처음부터 구축하면 시간과 비용이 많이 들기 때문에, 팀에서 공유하는 **DB 백업(Dump) 파일**을 복원하여 사용합니다.

1. **Docker 컨테이너 빌드 및 실행**
   - 아래 명령어로 커스텀 DB 이미지 빌드 및 컨테이너를 백그라운드에서 실행합니다:
     ```bash
     docker-compose up -d postgis
     ```
2. **DB 덤프 파일 다운로드**
   - *팀 구글 드라이브 링크: `[팀 구글 드라이브 DB 백업 파일 링크 삽입]`*
   - 다운로드받은 `spotsync_db_backup.dump` 파일을 프로젝트 루트 폴더에 위치시킵니다.
3. **DB 복원 명령어 실행**
   ```bash
   # 도커 컨테이너 내부로 덤프 파일을 복사한 뒤 복원합니다.
   docker cp spotsync_db_backup.dump spotsync-postgis:/tmp/
   docker exec -it spotsync-postgis pg_restore -U postgres -d spotsync -1 /tmp/spotsync_db_backup.dump
   ```

### 5. 프론트엔드 패키지 설치
`frontend` 폴더로 이동하여 npm 패키지들을 설치해 줍니다:
```bash
cd frontend
npm install
cd ..
```

### 6. 애플리케이션 원클릭 통합 관리 (실행/종료)
FastAPI 백엔드, Vite 프론트엔드, Docker DB를 매번 각각 켜고 끄는 번거로움을 해결하기 위해 **통합 서비스 관리 유틸리티**가 기본 포함되어 있습니다.

- **전체 서비스 시작 (DB 컨테이너, FastAPI 백엔드, Vite 프론트엔드 일괄 실행 및 자동 연결 검증)**:
  ```bash
  .venv\Scripts\python.exe .agents/skills/manage-server/scripts/manage_server.py start
  ```
- **전체 서비스 종료**:
  ```bash
  .venv\Scripts\python.exe .agents/skills/manage-server/scripts/manage_server.py stop
  ```
- **현재 가동 상태 확인 (PID 및 컨테이너 동작 상태)**:
  ```bash
  .venv\Scripts\python.exe .agents/skills/manage-server/scripts/manage_server.py status
  ```
- **개별 수동 실행 (필요한 경우)**:
  - Backend: `.venv\Scripts\python.exe -m uvicorn ai_search_v10:app --host 0.0.0.0 --port 8000`
  - Frontend: `frontend` 폴더에서 `npm run dev` 실행

