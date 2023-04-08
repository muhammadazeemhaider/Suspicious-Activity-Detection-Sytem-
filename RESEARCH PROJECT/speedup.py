import cv2
import numpy as np

cap = cv2.VideoCapture('1.mp4')

speed_factor = 4  # Increase the speed by a factor of 2
frame_skip = speed_factor - 1  # Number of frames to skip

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_file.mp4', fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0  # Keep track of the number of frames processed

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Only write every (speed_factor)th frame
        if frame_count % speed_factor == 0:
            out.write(frame)
        frame_count += 1
    else:
        break
    # Skip (frame_skip) frames on the next iteration
    for _ in range(frame_skip):
        ret, _ = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
