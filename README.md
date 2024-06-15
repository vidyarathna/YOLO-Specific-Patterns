# YOLO-Specific-Patterns

This repository contains a script for detecting specific patterns (class labels) in images using the YOLO object detection model. The detected objects are then saved as separate image files.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python 3.x
- OpenCV (cv2)
- YOLO v3 model files (`yolov3.weights`, `yolov3.cfg`, and `coco.names`)

## Installation

Clone the repository:

```sh
git clone https://github.com/vidyarathna/YOLO-Specific-Patterns.git
cd YOLO-Specific-Patterns
```

Install dependencies:

```sh
pip install -r requirements.txt
```

## Usage

To detect specific patterns in images using YOLO, run the script `yolo_detect_specific_patterns.py` with the following command:

```sh
python yolo_detect_specific_patterns.py --input path/to/input/images --yolo path/to/yolo-coco --output path/to/output --patterns person dog --confidence 0.5 --threshold 0.3
```

### Arguments

- `-i`, `--input`: Path to the input directory of images.
- `-y`, `--yolo`: Base path to the YOLO directory containing `coco.names`, `yolov3.weights`, and `yolov3.cfg`.
- `-o`, `--output`: Path to the output directory where detected objects will be saved.
- `-p`, `--patterns`: List of patterns (class labels) to detect and save.
- `-c`, `--confidence`: Minimum probability to filter weak detections (default is 0.5).
- `-t`, `--threshold`: Threshold when applying non-maxima suppression (default is 0.3).

### Example

```sh
python yolo_detect_specific_patterns.py --input ./input_images --yolo ./yolo-coco --output ./output_objects --patterns person dog --confidence 0.5 --threshold 0.3
```

This command will process all images in the `./input_images` directory, detect objects matching the "person" and "dog" class labels using the YOLO model, and save the detected objects in the `./output_objects` directory.

### Script Details

The script `yolo_detect_specific_patterns.py` performs the following steps:

1. Loads the COCO class labels YOLO was trained on.
2. Initializes colors to represent each class label.
3. Loads the YOLO model configuration and weights.
4. Processes each image in the input directory:
   - Loads the image and prepares it for detection.
   - Performs object detection using YOLO.
   - Filters detections based on confidence and specified class labels.
   - Applies non-maxima suppression to remove redundant overlapping boxes.
   - Extracts and saves the detected objects as separate images.

### Notes

- Make sure the input image directory contains images that can be read by OpenCV.
- Ensure the YOLO directory contains the `coco.names`, `yolov3.weights`, and `yolov3.cfg` files.
- Adjust the confidence and threshold parameters as needed to improve detection accuracy.
