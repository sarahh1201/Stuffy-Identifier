from flask import Flask, redirect, render_template, Response, request, url_for
import json
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load model
model = YOLO("/Users/sarahhill/Documents/Workspaces/Stuffy-Identifier/runs/detect/train/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Load metadata
with open("dataset/stuffy_metadata.json", "r") as f:
    metadata = json.load(f)

# Catalogue page
@app.route('/')
def index():
    return render_template('index.html', metadata=metadata)

# Detection page
@app.route('/detect')
def detect_page():
    return render_template('detect.html')

# Add new stuffed animal page
@app.route('/add', methods=['GET', 'POST'])
def add_animal_page():
    if request.method == 'POST':
        name = request.form.get('name')
        species = request.form.get('species')
        colours_raw = request.form.get('colours', '')
        acquired = request.form.get('acquired')
        size = request.form.get('size')

        colours = [c.strip() for c in colours_raw.split(',') if c.strip()]

        new_key = str(max(map(int, metadata.keys())) + 1)

        metadata[new_key] = {
            "name": name,
            "acquired": acquired,
            "size": size,
            "colours": colours,
            "species": species,
            "detectable": False
        }

        with open("dataset/stuffy_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return redirect(url_for('index'))

    return render_template('add_animal.html')

# Animal detail page
@app.route('/animal/<key>')
def animal_detail(key):
    item = metadata.get(key)
    if not item:
        return "Animal not found", 404
    return render_template('animal_detail.html', animal=item)

# Webcam stream
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Video feed route
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)