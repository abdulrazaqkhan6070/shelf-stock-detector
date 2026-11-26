from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
SAMPLES_DIR = Path("samples")
OUTPUT_DIR = Path("outputs_rows")
IMGSZ = 1280
CONF = 0.15
MAX_SIZE_FACTOR = 3.0   # drop boxes bigger than 3x the median width/height
MIN_SIZE_FACTOR = 0.25  # drop boxes smaller than 0.25x the median width/height
ROW_GAP_FACTOR = 0.6    # new row when a box center is > 0.6 x median height below the row average
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ROW_COLORS = [
    (0, 200, 0), (255, 120, 0), (200, 0, 200), (0, 180, 255),
    (0, 0, 220), (150, 150, 0), (255, 0, 120), (0, 130, 90),
]
GRAY = (150, 150, 150)


def detect_boxes(model, frame):
    """Run the model and return boxes as an N x 4 array of [x1, y1, x2, y2]."""
    result = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    return result.boxes.xyxy.cpu().numpy()


def filter_boxes(boxes):
    """Split boxes into (kept, removed) by size relative to the median box."""
    if len(boxes) == 0:
        return boxes, boxes
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    med_w, med_h = np.median(w), np.median(h)
    keep = (
        (w <= MAX_SIZE_FACTOR * med_w)
        & (h <= MAX_SIZE_FACTOR * med_h)
        & (w >= MIN_SIZE_FACTOR * med_w)
        & (h >= MIN_SIZE_FACTOR * med_h)
    )
    return boxes[keep], boxes[~keep]


def estimate_slope(cx, cy, med_h):
    """Estimate shelf tilt (dy/dx) from each box and its nearest right-hand neighbor."""
    slopes = []
    for i in range(len(cx)):
        same_level = (cx > cx[i]) & (np.abs(cy - cy[i]) < 0.8 * med_h)
        if not same_level.any():
            continue
        j = np.where(same_level)[0][np.argmin(cx[same_level] - cx[i])]
        slopes.append((cy[j] - cy[i]) / (cx[j] - cx[i]))
    if not slopes:
        return 0.0
    return float(np.clip(np.median(slopes), -0.5, 0.5))


def cluster_rows(boxes):
    """Group boxes into shelf rows (tilt-corrected), top to bottom, each sorted left to right."""
    if len(boxes) == 0:
        return [], 0.0
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    med_h = np.median(boxes[:, 3] - boxes[:, 1])

    slope = estimate_slope(cx, cy, med_h)
    cy = cy - slope * (cx - cx.mean())  # straighten tilted shelves
    order = np.argsort(cy)

    groups, current = [], [order[0]]
    for idx in order[1:]:
        if cy[idx] - np.mean(cy[current]) > ROW_GAP_FACTOR * med_h:
            groups.append(current)
            current = [idx]
        else:
            current.append(idx)
    groups.append(current)

    rows = []
    for g in groups:
        row = boxes[g]
        rows.append(row[np.argsort(row[:, 0])])  # left to right
    return rows, slope


def draw_result(frame, rows, removed):
    canvas = frame.copy()
    scale = max(1.0, frame.shape[1] / 1000)
    thick = max(2, int(2 * scale))

    for x1, y1, x2, y2 in removed.astype(int):  # unknown zones, thin gray
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GRAY, 1)

    for i, row in enumerate(rows):
        color = ROW_COLORS[i % len(ROW_COLORS)]
        for x1, y1, x2, y2 in row.astype(int):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
        x1, y1 = int(row[0, 0]), int(row[0, 1])
        cv2.putText(canvas, f"R{i + 1} ({len(row)})", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, color, thick)
    return canvas


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)
    images = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"{img_path.name}: could not read, skipped")
            continue

        boxes = detect_boxes(model, frame)
        kept, removed = filter_boxes(boxes)
        rows, slope = cluster_rows(kept)

        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_rows.jpg"), draw_result(frame, rows, removed))
        print(f"{img_path.name}: {len(kept)} kept, {len(removed)} unknown, "
                f"{len(rows)} rows {[len(r) for r in rows]} tilt={slope:+.3f}")

if __name__ == "__main__":
    main()