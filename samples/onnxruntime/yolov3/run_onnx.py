import numpy as np
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
from PIL import ImageDraw
import onnxruntime


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.
    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


def main():
    input_image_path = './dog.jpeg'
    output_image_path = 'dog_bboxes.png'
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (608, 608)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]

    # Sess_options
    sess_options = onnxruntime.SessionOptions()
    # Wheather enable profiling 
    sess_options.enable_profiling = True
    # Wheather enable graph optimization and choose optimization level
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = onnxruntime.InferenceSession("yolov3_b1.onnx", sess_options)

    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]

    print("inputs name:",inname,"outputs name:",outname)

    outputs = session.run(outname, {inname[0]:image})

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(outputs, output_shapes)]

    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                            "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                            (59, 119), (116, 90), (156, 198), (373, 326)],
                            "obj_threshold": 0.6,                                               # Threshold for object coverage, float value between 0 and 1
                            "nms_threshold": 0.5,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                            "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    obj_detected_img.save(output_image_path, 'PNG')
    print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

if __name__=='__main__':
    main()