import cv2
import os

video_path = "/Users/sarahhill/Downloads/IMG_6400.MOV"
output_folder = "/Users/sarahhill/Documents/Workspaces/Stuffy-Identifier/RawFrames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
frame_skip = 10  # change this if you want more/less images

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        filename = f"{output_folder}/drake{frame_count:04d}.jpg"
        cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()
print("Done!")