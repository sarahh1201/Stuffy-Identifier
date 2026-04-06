from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

## Test 1: Load model and perform inference on an image

# Load a trained model
model = YOLO("/Users/sarahhill/Documents/Workspaces/Stuffy-Identifier/runs/detect/train4/weights/best.pt")

# Perform inference on an image
results = model.predict(
    source="/Users/sarahhill/Documents/Workspaces/Stuffy-Identifier/dataset/images/val/test1.jpg",
    conf=0.1,
    save=True
)

## Test 2: Visualize bounding boxes from label file
# Paths
image_path = "/Users/sarahhill/Documents/Workspaces/Stuffy-Identifier/dataset/images/train/test14.jpg"
label_path = "/Users/sarahhill/Documents/Workspaces/Stuffy-Identifier/dataset/labels/train/test14.txt"

# Load image
img = Image.open(image_path)
img_width, img_height = img.size

# Read label file
with open(label_path, "r") as f:
    lines = f.readlines()

# Create plot
fig, ax = plt.subplots()
ax.imshow(img)

# Draw each bounding box
for line in lines:
    class_id, x_center, y_center, width, height = map(float, line.split())

    # Convert from normalized YOLO format to pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Calculate top-left corner
    x = x_center - width / 2
    y = y_center - height / 2

    # Create rectangle
    rect = patches.Rectangle(
        (x, y),
        width,
        height,
        linewidth=2
        ,fill=False
    )

    ax.add_patch(rect)
    ax.text(x, y, f"class {int(class_id)}", color='red', fontsize=10)

plt.axis('off')
plt.show()