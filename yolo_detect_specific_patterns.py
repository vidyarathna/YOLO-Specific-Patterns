import numpy as np
import argparse
import time
import cv2
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-o", "--output", required=True, help="path to output directory for cut objects")
ap.add_argument("-p", "--patterns", required=True, nargs='+', help="list of patterns (class labels) to detect and save")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Ensure the output directory exists
os.makedirs(args["output"], exist_ok=True)

# Get the indices of the patterns to detect
pattern_indices = [LABELS.index(pattern) for pattern in args["patterns"]]

# Process each image in the input directory
for image_filename in os.listdir(args["input"]):
    image_path = os.path.join(args["input"], image_filename)
    
    # Load the input image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"[WARNING] Failed to load image {image_path}")
        continue
    
    (H, W) = image.shape[:2]

    # Determine only the output layer names that we need from YOLO
    layerIDs = net.getUnconnectedOutLayers()
    ln = [net.getLayerNames()[i - 1] for i in layerIDs]

    # Construct a blob from the input image and then perform a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Show timing information on YOLO
    print(f"[INFO] YOLO took {end - start:.6f} seconds to process {image_filename}")

    # Initialize our lists of detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            # Extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > args["confidence"] and classID in pattern_indices:
                # Scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # Ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Extract the detected object
            object_crop = image[y:y+h, x:x+w]
            object_label = LABELS[classIDs[i]]
            # Use the original image filename and object class for the cropped object filename
            base_filename = os.path.splitext(image_filename)[0]
            output_path = os.path.join(args["output"], f"{base_filename}_{object_label}_{i}.png")
            cv2.imwrite(output_path, object_crop)

    # Optionally, save the image with bounding boxes drawn (uncomment if needed)
    # output_image_path = os.path.join(args["output"], f"{base_filename}_detected.png")
    # cv2.imwrite(output_image_path, image)

    # Show the output image with detected objects (uncomment if needed)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    
    
    
