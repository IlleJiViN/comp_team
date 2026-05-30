# SpotSync AI Semantic Search Benchmark Report

Generated at: 2026-05-30 13:09:22

## 🖥️ System & Model Configurations
- **Model Name**: `jhgan/ko-sroberta-multitask`
- **Device**: `cpu`
- **FAISS Index Type**: `IndexFlatIP` (Cosine Similarity via L2 Normalization)
- **PyTorch Thread Limit**: `4 threads`
- **Total Indexed Mock Places**: `5`

## 📊 Performance Metrics Summary
- **Overall Status**: **🎉 SUCCESS** (Tail latency is well under the 200ms threshold.)
- **Matching Accuracy**: **4/4** (100.0%)
- **Average Search Latency**: `143.95 ms`
- **Tail Latency (p90)**: `177.82 ms`
- **Peak Latency (p99)**: `226.08 ms`

## 🧪 Detailed Verification Scenarios
| # | Query | Expected Place | Matched Place | Similarity Score | Latency (ms) | Status |
|---|---|---|---|---|---|---|
| 1 | 드럼이랑 마이크 성능 좋은 방음 잘되는 음악 합주실 | 싱크사운드 신촌점 | 싱크사운드 신촌점 | 0.8069 | 163.08 | ✅ MATCH |
| 2 | 노트북 들고 공부하기 편한 조용하고 편안한 카페 | 카페 조용한 공간 | 카페 조용한 공간 | 0.7592 | 134.05 | ✅ MATCH |
| 3 | 컴퓨터 그래픽카드 최고 사양 게이밍 모니터 넓은 피씨방 | 아이린 PC방 연세대점 | 아이린 PC방 연세대점 | 0.7635 | 124.41 | ✅ MATCH |
| 4 | 혼자 가서 보컬 연습하고 마이크 녹음하기 조용한 스튜디오 | 스타 보컬 스튜디오 | 스타 보컬 스튜디오 | 0.7027 | 125.38 | ✅ MATCH |
