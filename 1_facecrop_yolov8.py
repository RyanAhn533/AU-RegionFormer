import os
import shutil
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image, ImageFile
import cv2

# Pillow에서 truncated 이미지 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

# YOLOv8 Face Detection 모델 다운로드 및 로드
print("[INFO] Downloading YOLOv8 Face Detection model...")
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)
print("[INFO] Model loaded.")

# 원본 데이터 경로
src_root = "/media/ajy/EXTERNAL_USB/한국인 감정인식을 위한 복합 영상/Training"
# 저장할 경로
dst_root = "/home/ajy/AI_hub_250704/data"

def load_image(image_path):
    """
    안전하게 이미지를 읽는다: 
    Pillow → 실패시 OpenCV fallback
    """
    try:
        print(f"[DEBUG] Trying PIL open: {image_path}")
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path).convert("RGB")
        print(f"[DEBUG] PIL success: {image_path}")
        return img
    except Exception as e:
        print(f"[WARN] PIL failed: {e} -> trying cv2 for {image_path}")
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            print(f"[ERROR] cv2 cannot read image: {image_path}")
            return None
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        print(f"[DEBUG] cv2 fallback success: {image_path}")
        return img

def crop_face(image_path):
    img = load_image(image_path)
    if img is None:
        return None
    try:
        results = model(img)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None
        box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2 = map(int, box)
        cropped = img.crop((x1, y1, x2, y2))
        return cropped
    except Exception as e:
        print(f"[ERROR] YOLO detection failed: {e} on {image_path}")
        return None

# 디렉토리 순회
for root, dirs, files in os.walk(src_root):
    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        src_file_path = os.path.join(root, file)

        # 대상 디렉토리 구조 유지
        relative_path = os.path.relpath(root, src_root)
        dst_dir = os.path.join(dst_root, relative_path)
        os.makedirs(dst_dir, exist_ok=True)

        cropped_face = crop_face(src_file_path)
        if cropped_face is not None:
            dst_file_path = os.path.join(dst_dir, file)
            cropped_face.save(dst_file_path)
            print(f"[INFO] Saved cropped face to {dst_file_path}")
        else:
            print(f"[WARN] No face detected or failed to process: {src_file_path}")
