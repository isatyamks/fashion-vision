import cv2
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np

def is_similar(img1, img2, threshold=0.4):
    # Compare color histograms of two images
    img1 = img1.resize((100, 100)).convert("RGB")
    img2 = img2.resize((100, 100)).convert("RGB")
    h1 = cv2.calcHist([np.array(img1)], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    h2 = cv2.calcHist([np.array(img2)], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return similarity > threshold

def main():
    model = YOLO("weights\\best.pt")
    names = model.names
    conf_threshold = 0.6
    output_dir = "video_crops_unique1"
    os.makedirs(output_dir, exist_ok=True)

    video_path = "data\\videos\\2025-05-31_14-01-37_UTC.mp4"
    cap = cv2.VideoCapture(video_path)

    frame_skip = 5  # Process every 5th frame
    frame_count = 0
    saved_crops = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(frame)[0]

        for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            cropped = pil_image.crop((x1, y1, x2, y2))
            class_id = int(cls.item())
            class_name = names[class_id]

            # Compare with previous crops
            is_dup = False
            for saved in saved_crops:
                if is_similar(cropped, saved):
                    is_dup = True
                    break
            if is_dup:
                continue

            saved_crops.append(cropped)
            crop_filename = f"{output_dir}/frame{frame_count}_obj{i}_{class_name}_conf{conf:.2f}.jpg"
            cropped.save(crop_filename)
            print(f"Saved unique: {crop_filename}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
