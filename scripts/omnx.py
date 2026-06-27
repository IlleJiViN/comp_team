from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

model_id = "BAAI/bge-m3"
onnx_path = Path("bge_m3_onnx_int8")

# 1. 원본 모델을 ONNX 포맷으로 로드 (자동 변환 및 로컬 저장)
# file_name="model.onnx"로 기본 지정됨
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 로드된 ONNX 모델을 INT8(정수 8비트)로 동적 양자화하여 복사본 저장
# CPU 연산 아키텍처에 맞게 내부 가중치를 변환합니다.
model.config.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)

# optimum 인프라를 통해 최적화 툴 장착 후 양자화 모델 파일로 저장
from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer

# 양자화 설정 (CPU 환경용 동적 양자화)
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = QuantizationConfig(is_static=False, weight_type="int8")

# 변환 실행 후 최종 저장
quantizer.quantize(quantization_config=qconfig, save_dir=onnx_path)