# Stuffy Identifier

Identify and catalogue stuffed animals with a YOLO detection model and a cute desktop UI.

## Tools Used

- [makesense.ai](https://www.makesense.ai) for labelling
- OpenCV for image processing
- Ultralytics for model training
- Flask for the local backend
- Electron for the desktop app shell

## Setup

Activate the Python environment:

```bash
source venv/bin/activate
```

Install Electron dependencies (one time):

```bash
npm install
```

## Run the Desktop App

```bash
npm start
```

This launches Electron, starts the Flask backend automatically, and opens the Stuffy Identifier window.

## Run Flask Only (Browser)

```bash
source venv/bin/activate
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Pages

- **Catalogue** — browse all stuffies with species, colours, and detectable status
- **Detect** — live webcam feed with YOLO bounding boxes and a sidebar of detected friends
- **Add New** — add catalogue entries for stuffies not yet in the model

## Project Layout

- `app.py` — Flask backend, YOLO inference, metadata API
- `desktop/main.js` — Electron main process (spawns Flask, opens window)
- `dataset/stuffy_metadata.json` — stuffed animal catalogue data
- `dataset/data.yaml` — YOLO class names for training
- `templates/` — HTML pages
- `static/` — CSS and frontend JavaScript
