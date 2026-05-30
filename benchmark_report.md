# SpotSync AI v4 - Decomposed Category Similarity Benchmark Report

Generated at: 2026-05-30 14:32:10

## 🏗️ Architecture: Decomposed Category Similarity (v4)
- **Approach**: Query → Category Prototype Matching → PostGIS Category Filter → Name Similarity → Weighted Score
- **Scoring Formula**: `final = 0.6 × category_sim + 0.4 × name_sim`
- **Category Threshold**: `0.45`

## 🖥️ System & Model Configurations
- **Model Name**: `jhgan/ko-sroberta-multitask`
- **Device**: `cpu`
- **Category Prototypes**: `70` categories pre-cached
- **PyTorch Thread Limit**: `4 threads`
- **Total Mock Places**: `5`

## 📊 Performance Metrics Summary
- **Overall Status**: **🎉 SUCCESS** (Tail latency is well under the 200ms threshold.)
- **Matching Accuracy**: **4/4** (100.0%)
- **Average Search Latency**: `74.99 ms`
- **Tail Latency (p90)**: `98.58 ms`
- **Peak Latency (p99)**: `135.94 ms`

## 🧪 Detailed Verification Scenarios
| # | Query | Expected | Matched | Category | Score | Latency (ms) | Status |
|---|-------|----------|---------|----------|-------|-------------|--------|
| 1 | 드럼이랑 마이크 성능 좋은 방음 잘되는 음악 합주실 | 싱크사운드 신촌점 | 싱크사운드 신촌점 | 합주실, 연습실, 노래방 | 0.6557 | 62.63 | ✅ MATCH |
| 2 | 노트북 들고 공부하기 편한 조용하고 편안한 카페 | 카페 조용한 공간 | 카페 조용한 공간 | 독서실/스터디 카페, 카페, 연습실 | 0.5896 | 60.01 | ✅ MATCH |
| 3 | 컴퓨터 그래픽카드 최고 사양 게이밍 모니터 넓은 피씨방 | 아이린 PC방 연세대점 | 아이린 PC방 연세대점 | PC방, 전자 게임장, 노래방 | 0.6837 | 149.33 | ✅ MATCH |
| 4 | 혼자 가서 보컬 연습하고 마이크 녹음하기 조용한 스튜디오 | 스타 보컬 스튜디오 | 스타 보컬 스튜디오 | 연습실 | 0.6101 | 86.91 | ✅ MATCH |
