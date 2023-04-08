import cv2
import numpy as np
import torch

# Define the expected sizes and colors for each cube
cube_sizes = {'red': (10, 10), 'green': (20, 20), 'blue': (30, 30)}
cube_colors = {'red': ([0, 0, 100], [50, 50, 255]),
               'green': ([0, 100, 0], [50, 255, 50]),
               'blue': ([100, 0, 0], [255, 50, 50])}

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Start the main loop
while True:
    # Read an image from the camera
    ret, img = cap.read()
    if not ret:
        break
    
    # Perform object detection using YOLOv5
    results = model(img)
    
    # Extract the bounding boxes and class probabilities from the results
    bboxes = results.xyxy[0].cpu().numpy()
    probs = results.xyxy[0][:, 4].cpu().numpy()
    classes = results.xyxy[0][:, 5].cpu().numpy()
    
    # Filter out low confidence detections
    mask = probs > 0.5
    bboxes = bboxes[mask]
    classes = classes[mask]
    
    # Filter out detections that do not match the expected size and color for each cube
    detected_cubes = {}
    for color, size in cube_sizes.items():
        color_range = cube_colors[color]
        mask = np.logical_and.reduce((bboxes[:, 2] - bboxes[:, 0] >= size[0],
                                       bboxes[:, 3] - bboxes[:, 1] >= size[1],
                                       np.all(bboxes[:, 5:8] >= np.array(color_range[0]), axis=1),
                                       np.all(bboxes[:, 5:8] <= np.array(color_range[1]), axis=1)))
        color_bboxes = bboxes[mask]
        color_probs = probs[mask]
        color_classes = classes[mask]
        
        # Find the center of mass for each detected cube
        centers = []
        for bbox in color_bboxes:
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int((bbox[1] + bbox[3]) / 2)
            centers.append((x_center, y_center))
        
        # Add the detected cubes to the dictionary
        detected_cubes[color] = list(zip(centers, color_bboxes.tolist(), color_probs.tolist(), color_classes.tolist()))
    
    # Display the image with the detected cubes
    for color, cubes in detected_cubes.items():
        for center, bbox, prob, cls in cubes:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{color} {prob:.2f}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, center, 5, (0, 255, 0), -1)
    
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

