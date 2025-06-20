import torch
import clip
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# === Load models ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
yolo = YOLO("yolov8m.pt")

COLOR_LABELS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'orange': (0, 165, 255),
    'brown': (42, 42, 165)
}

def get_dominant_color_name(image_crop):
    small = cv2.resize(image_crop, (20, 20), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    avg = np.mean(pixels, axis=0)
    def color_distance(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))
    return min(COLOR_LABELS.keys(), key=lambda name: color_distance(avg, COLOR_LABELS[name]))

def get_yolo_detections(image_path):
    image = cv2.imread(image_path)
    results = yolo(image_path)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label_id = int(box.cls[0])
        base_label = yolo.names[label_id]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        color = get_dominant_color_name(crop)
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        clip_input = clip_preprocess(pil_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = clip_model.encode_image(clip_input).cpu().numpy().flatten()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detections.append({
            "label": base_label,
            "color": color,
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
            "embedding": embedding
        })
    return detections, image

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_detections(before, after, threshold=0.8, max_distance=100):
    matched = []
    unmatched_after = after.copy()
    unmatched_before = before.copy()
    global_id = 1
    id_map = {}
    for b in before:
        best_match = None
        best_score = -1
        best_dist = float('inf')
        for a in unmatched_after:
            if a["label"] != b["label"]:
                continue
            dx = a["center"][0] - b["center"][0]
            dy = a["center"][1] - b["center"][1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > max_distance:
                continue
            sim = cosine_similarity(a["embedding"], b["embedding"])
            if sim >= threshold and sim > best_score:
                best_score = sim
                best_match = a
                best_dist = dist
            elif sim >= 0.65 and dist < 20:
                best_score = sim
                best_match = a
                best_dist = dist
        if best_match:
            id_map[id(best_match)] = global_id
            id_map[id(b)] = global_id
            matched.append((b, best_match, global_id))
            unmatched_after.remove(best_match)
            unmatched_before.remove(b)
            global_id += 1
    return matched, unmatched_after, unmatched_before, id_map, global_id

def cleanup_files(*file_paths):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted: {path}")
        except Exception as e:
            print(f"Error deleting {path}: {e}")


def run_detection(before_path, after_path, output_path):
    before_dets, _ = get_yolo_detections(before_path)
    after_dets, after_img = get_yolo_detections(after_path)
    matched, new_objects, removed_objects, id_map, next_id = match_detections(before_dets, after_dets)
    for b, a, obj_id in matched:
        dist = int(np.sqrt((a["center"][0] - b["center"][0]) ** 2 + (a["center"][1] - b["center"][1]) ** 2))
        x1, y1, x2, y2 = a["bbox"]
        label = f"{a['color']} {a['label']} #{obj_id} d={dist}px"
        cv2.rectangle(after_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(after_img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    for obj in new_objects:
        x1, y1, x2, y2 = obj["bbox"]
        obj_id = next_id
        next_id += 1
        label = f"{obj['color']} {obj['label']} #{obj_id} (new)"
        cv2.rectangle(after_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(after_img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for obj in removed_objects:
        x1, y1, x2, y2 = obj["bbox"]
        obj_id = next_id
        next_id += 1
        label = f"{obj['color']} {obj['label']} #{obj_id} (removed)"
        cv2.rectangle(after_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(after_img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(output_path, after_img)
    print("Detection complete. Output saved.")
