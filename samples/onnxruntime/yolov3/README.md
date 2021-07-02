# Object detection with yolov3.onnx

## Description
This sample, yolov3_onnx, implements a full ONNX-based pipeline for performing inference with the YOLOv3 network, with an input size of 608 x 608 pixels, including pre and post-processing. This sample is based on the [YOLOv3-608](https://pjreddie.com/media/files/papers/YOLOv3.pdf) paper.

## How it works
Firstly, a [yolov3.onnx]() should be downloaded.

Run
```bash 
python3 run_onnx.py
```
for easy start.

Run
```bash
python3 quantize_model.py 
```
to generate yolov3_int8.onnx
