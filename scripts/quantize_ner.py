import json
import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

model_id = "models/spotsync-ner"
onnx_path = Path("models/spotsync-ner-onnx")

print("Loading original NER model and converting to ONNX...")
# export=True forces HuggingFace to convert PyTorch weights to ONNX format
model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save the original ONNX (fp32)
model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)

# Copy label_config.json as well
with open(os.path.join(model_id, "label_config.json"), "r", encoding="utf-8") as f:
    label_config = json.load(f)
with open(os.path.join(onnx_path, "label_config.json"), "w", encoding="utf-8") as f:
    json.dump(label_config, f, ensure_ascii=False)

print("Quantizing ONNX model to INT8...")
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# Apply dynamic INT8 quantization for CPU environments
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx2(is_static=False)

# This will generate model_quantized.onnx in the target directory
quantizer.quantize(quantization_config=qconfig, save_dir=onnx_path)

print("Quantization complete! Model saved to", onnx_path)
