from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"  # shelf-trained single-class model (class: item)
SAMPLES_DIR = Path("samples")
OUTPUT_DIR = Path("outputs")
CONF = 0.15
IMGSZ = 1280
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)

    images = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)
    print(f"Model classes: {model.names}")
    print(f"Found {len(images)} images\n")

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"{img_path.name}: could not read, skipped")
            continue

        result = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        annotated = result.plot(labels=False)  # boxes only, less clutter on dense shelves

        out_path = OUTPUT_DIR / f"{img_path.stem}_detected.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"{img_path.name}: {len(result.boxes)} products -> {out_path.name}")


if __name__ == "__main__":
    main()