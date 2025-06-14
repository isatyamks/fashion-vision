import cv2
from ultralytics import YOLO
from PIL import Image
import os

def main():
    model = YOLO("..\\..\\data_20\\runs\\detect\\fashion-v12\\weights\\best.pt")
    names = model.names  # Class labels
    conf_threshold = 0.5
    output_dir = "video_crops"
    os.makedirs(output_dir, exist_ok=True)

    video_path = "..\\data\\videos\\2025-05-31_14-01-37_UTC.mp4"
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_count}...")

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(frame)[0]

        print("Boxes:", results.boxes)
        print("Coords:", results.boxes.xyxy)
        print("Confidences:", results.boxes.conf)
        print("Classes:", results.boxes.cls)

        for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            cropped = pil_image.crop((x1, y1, x2, y2))
            class_id = int(cls.item())
            class_name = names[class_id]
            crop_filename = f"{output_dir}/frame{frame_count}_obj{i}_{class_name}_conf{conf:.2f}.jpg"
            cropped.save(crop_filename)
            print(f"Saved: {crop_filename}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete!")

if __name__ == "__main__":
    main()
