import argparse
import json
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs/detect/train/weights/best.pt"
METADATA_PATH = BASE_DIR / "dataset/stuffy_metadata.json"

SPECIES_EMOJI = {
    "Fox": "🦊",
    "Bear Mix": "🐻",
    "Dinosaur": "🦕",
    "Dragon": "🐉",
    "Cow": "🐮",
    "Monkey": "🐵",
}

CARD_COLORS = ["#FAEDCB", "#C9E4DE", "#C6DEF1", "#DBCDF0", "#F2C6DE", "#F7D9C4", "#E8D5F2"]

app = Flask(__name__)
model = YOLO(str(MODEL_PATH))

cap = None
latest_detections = []


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


metadata = load_metadata()


def save_metadata():
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def get_cap():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap


def release_cap():
    global cap
    if cap is not None:
        cap.release()
        cap = None


def metadata_key_for_name(name):
    for key, item in metadata.items():
        if item["name"].lower() == name.lower():
            return key
    return None


@app.context_processor
def inject_helpers():
    page_map = {
        "index": "catalogue",
        "detect_page": "detect",
        "add_animal_page": "add",
        "animal_detail": "catalogue",
    }
    return {
        "species_emoji": lambda species: SPECIES_EMOJI.get(species, "🧸"),
        "card_color": lambda index: CARD_COLORS[index % len(CARD_COLORS)],
        "active_page": page_map.get(request.endpoint, ""),
    }


@app.route("/")
def index():
    return render_template("index.html", metadata=metadata)


@app.route("/detect")
def detect_page():
    return render_template("detect.html")


@app.route("/add", methods=["GET", "POST"])
def add_animal_page():
    global metadata

    if request.method == "POST":
        name = request.form.get("name")
        species = request.form.get("species")
        colours_raw = request.form.get("colours", "")
        acquired = request.form.get("acquired")
        size = request.form.get("size")

        colours = [c.strip() for c in colours_raw.split(",") if c.strip()]
        new_key = str(max(map(int, metadata.keys())) + 1)

        metadata[new_key] = {
            "name": name,
            "acquired": acquired or None,
            "size": size,
            "colours": colours,
            "species": species,
            "detectable": False,
        }
        save_metadata()

        return redirect(url_for("index"))

    return render_template("add_animal.html")


@app.route("/animal/<key>")
def animal_detail(key):
    item = metadata.get(key)
    if not item:
        return "Animal not found", 404
    return render_template("animal_detail.html", animal=item, key=key)


@app.route("/api/detections")
def api_detections():
    return jsonify(latest_detections)


def generate_frames():
    global latest_detections

    while True:
        success, frame = get_cap().read()
        if not success:
            break

        results = model(frame, verbose=False)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[cls_id]
            detections.append(
                {
                    "name": name,
                    "confidence": round(confidence, 2),
                    "key": metadata_key_for_name(name),
                }
            )

        latest_detections = detections
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        release_cap()
