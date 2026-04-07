import cv2
import json
from ultralytics import YOLO

# Load model
model = YOLO("/Users/sarahhill/Documents/Workspaces/Stuffy-Identifier/runs/detect/train/weights/best.pt")

# Load metadata
with open("stuffy_metadata.json", "r") as f:
    metadata = json.load(f)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cv2.namedWindow("Webcam Detection", cv2.WINDOW_NORMAL)

frame_count = 0
annotated_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1

    # Run detection every 5 frames
    if frame_count % 5 == 0:
        results = model(frame, conf=0.1, verbose=False)

        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            key = str(cls)

            if key in metadata:
                info = metadata[key]

                print("----- DETECTED -----")
                print(f"Name: {info['name']}")
                print(f"Species: {info['species']}")
                print("Colors: " + ", ".join(info['colors']))
                print(f"Confidence: {conf:.2f}")
                print("--------------------")

        # Draw boxes on frame
        annotated_frame = results[0].plot()

    # If no new detection, reuse last frame
    if annotated_frame is None:
        annotated_frame = frame

    # Show frame
    cv2.imshow("Webcam Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()