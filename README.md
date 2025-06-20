# Cleaing AI POC

This is a proof-of-concept AI system that detects object movement and changes between two images using a combination of YOLOv8, CLIP, and basic color analysis. It is designed to help identify whether items in a space (like a room) have been moved, added, or removed — potentially useful in contexts like cleaning validation or inventory tracking.

---

## Features

-  **YOLOv8 Object Detection**: Detects and labels objects in both images.
-  **Color Labeling**: Assigns approximate color labels to each object using dominant color estimation.
-  **CLIP Embedding Comparison**: Matches objects between images based on semantic similarity.
-  **Distance Tracking**: Measures how far matched objects have moved.
- **Visual Overlays**: 
  - Orange: Moved objects
  - Green: Newly added
  - Red: Removed

---

## Tech Stack

- Python
- [YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8m.pt`)
- [CLIP (ViT-B/32)](https://github.com/openai/CLIP)
- OpenCV (cv2)
- NumPy
- Flask (for demo UI)

---

## Workflow

1. User uploads **before** and **after** images via the Flask frontend.
2. YOLO detects objects and generates bounding boxes + labels.
3. Dominant color of each object is estimated.
4. CLIP embeddings are extracted for each object crop.
5. Objects are matched across images using cosine similarity and spatial distance.
6. Visual labels are added to the "after" image to show changes.

---

## Folder Structure
Cleaning-AI-POC/
├── detect.py # Core detection logic
├── app.py # Flask server (if included)
├── templates/
│ └── index.html # Upload form UI
├── static/
│ └── output.jpg # Result image
├── uploads/ # User-uploaded images
├── requirements.txt
└── README.md


---

## Setup

```bash
# Clone repo
git clone https://github.com/UndergradMartinC/Cleaing-AI-POC.git
cd Cleaing-AI-POC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download YOLO weights if not already included
# Place yolov8m.pt in the project root


## Usage

```bash
# Run the Flask app
python app.py
```

1. Visit `http://localhost:5000` in your browser.
2. Upload your **before** and **after** images (JPG format recommended).
3. The output image will display with:
   -  Moved items
   -  New items
   -  Removed items
4. Result is saved to `/static/output.jpg` (can be changed).

---

## Limitations

- Color labeling is done via naive average RGB matching.
- No true object tracking; relies on CLIP embedding similarity.
- Cannot detect changes for heavily occluded or rare objects.
- Scene context is missing (e.g., cannot infer furniture arrangement changes).
- Accuracy varies depending on lighting and image quality.


---

## Author
Martin Cook