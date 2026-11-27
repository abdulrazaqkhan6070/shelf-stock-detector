import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from shelf_rows import EXTENSIONS, MODEL_PATH, SAMPLES_DIR, cluster_rows, detect_boxes, filter_boxes

OUTPUT_DIR = Path("outputs_gaps")
GAP_FACTOR = 1.2          # gap must be wider than 1.2 x median product width
MIN_ROW_BOXES = 3         # rows with fewer boxes are not analyzed
UNKNOWN_OVERLAP = 0.2     # skip a gap if over 20% of it overlaps an unknown zone
OCCUPIED_OVERLAP = 0.3    # skip a gap if over 30% of it is covered by any detected box
RECHECK_CONF = 0.10       # confidence used when re-running the detector on a candidate slot
RECHECK_MIN_CONF = 0.10   # a re-check detection must reach this to rescue a slot (same as before)
RESTOCK_THRESHOLD = 85.0  # restock_alert is True below this fill rate (%)

RED = (0, 0, 255)
GREEN = (0, 200, 0)
GRAY = (150, 150, 150)
YELLOW = (0, 215, 255)


def overlap_ratio(gap, boxes):
    """Largest share of the gap rectangle covered by any single box."""
    gx1, gy1, gx2, gy2 = gap
    area = max(1.0, (gx2 - gx1) * (gy2 - gy1))
    best = 0.0
    for bx1, by1, bx2, by2 in boxes:
        iw = min(gx2, bx2) - max(gx1, bx1)
        ih = min(gy2, by2) - max(gy1, by1)
        if iw > 0 and ih > 0:
            best = max(best, iw * ih / area)
    return best


def covered_fraction(slot, boxes):
    """Share of the slot area covered by the union of all given boxes."""
    sx1, sy1, sx2, sy2 = int(slot[0]), int(slot[1]), int(slot[2]), int(slot[3])
    w, h = sx2 - sx1, sy2 - sy1
    if w <= 0 or h <= 0 or len(boxes) == 0:
        return 0.0
    mask = np.zeros((h, w), dtype=bool)
    for bx1, by1, bx2, by2 in boxes:
        x1, x2 = max(sx1, int(bx1)) - sx1, min(sx2, int(bx2)) - sx1
        y1, y2 = max(sy1, int(by1)) - sy1, min(sy2, int(by2)) - sy1
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return float(mask.mean())


def slot_has_product(frame, slot, model, med_w):
    """Re-run the detector on a zoomed crop around a candidate slot.

    Returns the best confidence of a qualifying detection centered in the slot,
    or None when nothing qualifies.
    """
    x1, y1, x2, y2 = (int(v) for v in slot)
    pad = int(max(med_w, x2 - x1))  # extra context on each side
    cx1, cx2 = max(0, x1 - pad), min(frame.shape[1], x2 + pad)
    cy1, cy2 = max(0, y1), min(frame.shape[0], y2)
    if cx2 - cx1 < 20 or cy2 - cy1 < 20:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    result = model.predict(crop, conf=RECHECK_CONF, imgsz=640, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    best = None
    for (bx1, by1, bx2, by2), c in zip(boxes, confs):
        center_x = cx1 + (bx1 + bx2) / 2
        if x1 <= center_x <= x2 and (bx2 - bx1) >= 0.3 * med_w and c >= RECHECK_MIN_CONF:
            best = float(c) if best is None else max(best, float(c))
    return best


def find_gaps(rows, kept, unknown, med_w, frame, model):
    """Return (empty_slots, uncertain_slots, total_row_span).

    empty_slots:     list of (row_index, slot)
    uncertain_slots: list of (row_index, slot, reason, conf) with reason in {unk, cov, chk}
    """
    empty, uncertain, total_span = [], [], 0.0
    for r_idx, row in enumerate(rows):
        if len(row) < MIN_ROW_BOXES:
            continue
        total_span += row[:, 2].max() - row[:, 0].min()
        band_y1 = float(np.median(row[:, 1]))  # slot height = this row's typical box height
        band_y2 = float(np.median(row[:, 3]))
        reach = row[0]  # box with the furthest right edge so far
        for box in row[1:]:
            gap_w = box[0] - reach[2]
            if gap_w > GAP_FACTOR * med_w:
                n = max(1, int(round(gap_w / med_w)))  # split into product-sized slots
                step = gap_w / n
                for k in range(n):
                    sx1 = reach[2] + k * step
                    slot = (sx1, band_y1, sx1 + step, band_y2)
                    if len(unknown) and overlap_ratio(slot, unknown) > UNKNOWN_OVERLAP:
                        uncertain.append((r_idx, slot, "unk", None))
                    elif covered_fraction(slot, kept) > OCCUPIED_OVERLAP:
                        uncertain.append((r_idx, slot, "cov", None))
                    else:
                        conf = slot_has_product(frame, slot, model, med_w)
                        if conf is not None:
                            uncertain.append((r_idx, slot, "chk", conf))
                        else:
                            empty.append((r_idx, slot))
            if box[2] > reach[2]:
                reach = box
    return empty, uncertain, total_span


def analyze(frame, model, details=None):
    """Return (annotated_frame, metrics_dict) for one shelf image."""
    boxes = detect_boxes(model, frame)
    kept, removed = filter_boxes(boxes)
    rows, _ = cluster_rows(kept)

    empty, uncertain, total_span = [], [], 0.0
    if len(kept):
        med_w = float(np.median(kept[:, 2] - kept[:, 0]))
        empty, uncertain, total_span = find_gaps(rows, kept, removed, med_w, frame, model)

    empty_w = sum(s[2] - s[0] for _, s in empty)
    if total_span > 0:
        fill = round(float(np.clip(100 * (1 - empty_w / total_span), 0, 100)), 1)
        alert = bool(fill < RESTOCK_THRESHOLD)
    else:
        fill, alert = None, False  # nothing analyzable in this image

    metrics = {
        "fill_rate_pct": fill,
        "empty_slots": [
            {"row": r + 1, "x1": int(s[0]), "y1": int(s[1]), "x2": int(s[2]), "y2": int(s[3])}
            for r, s in empty
        ],
        "restock_alert": alert,
        "uncertain_gaps": len(uncertain),
    }

    if details is not None:  # diagnostic printout, does not affect the metrics
        for r, s in empty:
            details.append(f"  EMPTY  row {r + 1}  x {int(s[0])}-{int(s[2])}  y {int(s[1])}-{int(s[3])}")
        for r, s, reason, conf in uncertain:
            if reason == "chk":
                details.append(f"  chk    row {r + 1}  x {int(s[0])}-{int(s[2])}  y {int(s[1])}-{int(s[3])}  conf {conf:.2f}")
        counts = {k: sum(1 for u in uncertain if u[2] == k) for k in ("unk", "cov", "chk")}
        details.append(f"  uncertain by rule: {counts}")

    return draw(frame, kept, removed, empty, uncertain, metrics), metrics


def draw(frame, kept, removed, empty, uncertain, metrics):
    canvas = frame.copy()
    scale = max(1.0, frame.shape[1] / 1000)
    thick = max(2, int(2 * scale))

    for x1, y1, x2, y2 in removed.astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GRAY, 1)
    for x1, y1, x2, y2 in kept.astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 1)

    # diagnostic only: slots the code rejected, labelled with the rule (and confidence for chk)
    for _, (x1, y1, x2, y2), reason, conf in uncertain:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), YELLOW, 1)
        text = reason if conf is None else f"{reason} {conf:.2f}"
        cv2.putText(canvas, text, (int(x1) + 2, int(y1) + int(14 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, YELLOW, 1)

    overlay = canvas.copy()
    for _, (x1, y1, x2, y2) in empty:
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), RED, -1)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    for _, (x1, y1, x2, y2) in empty:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), RED, thick)
        cv2.putText(canvas, "EMPTY SLOT", (int(x1), max(20, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, RED, max(1, thick - 1))

    fill = metrics["fill_rate_pct"]
    label = "Fill: n/a" if fill is None else f"Fill: {fill}%"
    label += f" | {len(empty)} empty"
    if metrics["restock_alert"]:
        label += " | RESTOCK"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, thick)
    cv2.rectangle(canvas, (0, 0), (tw + 20, th + 24), (0, 0, 0), -1)
    cv2.putText(canvas, label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale,
                (255, 255, 255), thick)
    return canvas


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)
    wanted = set(sys.argv[1:])  # optional: image names without extension, e.g. test_1803
    images = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)
    if wanted:
        images = [p for p in images if p.stem in wanted]

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"{img_path.name}: could not read, skipped")
            continue

        details = [] if wanted else None
        annotated, metrics = analyze(frame, model, details)
        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_gaps.jpg"), annotated)
        (OUTPUT_DIR / f"{img_path.stem}_gaps.json").write_text(json.dumps(metrics, indent=2))
        print(f"{img_path.name}: fill={metrics['fill_rate_pct']} "
              f"empty={len(metrics['empty_slots'])} uncertain={metrics['uncertain_gaps']} "
              f"restock={metrics['restock_alert']}")
        for line in details or []:
            print(line)


if __name__ == "__main__":
    main()