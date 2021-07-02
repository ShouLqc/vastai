import onnx 
import onnxoptimizer
from onnxsim import simplify
from onnxsim.onnx_simplifier import check

model = onnx.load('yolov3_b1.onnx')
model, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"
model = onnxoptimizer.optimize(model)
onnx.save(model, 'yolov3_optimized.onnx')
print('success')