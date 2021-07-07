from onnxruntime.quantization import quantize_qat, QuantType

model_fp32 = 'yolov3.onnx'
model_quant = 'yolov3_int8.onnx'
quantized_model = quantize_qat(model_fp32, model_quant)