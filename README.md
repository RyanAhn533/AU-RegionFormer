AU-Patch Attention FER: An Action-Unit-Aware Facial Expression Recognition Pipeline

<img width="1536" height="1024" alt="ChatGPT Image 2026년 2월 24일 오후 03_31_47" src="https://github.com/user-attachments/assets/4be31431-0bca-4908-b411-c09abf4fcb04" />


A fully end-to-end FER pipeline that extracts region-specific AU patches using MediaPipe FaceMesh and fuses them with global facial features via a lightweight Attention-based architecture.

Overview

This project uses the AI Hub Korean Facial Expression dataset and combines global face representations with localized Action Unit (AU) regions.
Rather than relying solely on full-face images, the system isolates six expression-critical facial regions—forehead, eyes, nose, cheeks, mouth, chin—and feeds their patches into a transformer-based fusion module.

The key idea is simple:
Global context + AU-focused local evidence → significantly more robust FER performance.

End-to-End Pipeline
Original Image
   │
   ├─ [Step 0] YOLOv8 Face Crop (optional)
   │       → Clean face-only images in cluttered scenes
   │
   ├─ [Step 1] AU Patch Coordinate Extraction (MediaPipe FaceMesh)
   │       → Resize → auto-rotate correction → detect 468 landmarks → generate AU CSV
   │
   ├─ [Step 2] CSV Validation
   │       → NaN/inf scan, path existence checks
   │
   └─ [Step 3] AU Fusion Attention Training
           → Global Encoder + Patch Encoder → Fusion Attention → FER Head
File Structure
File	Stage	Description
1_facecrop_yolov8.py	Step 0	YOLOv8-based face detection and cropping
250809_AU_crop_csv_copy_3.py	Step 1	AU patch coordinate extraction using MediaPipe FaceMesh
250810_csv_check_nan.py	Step 2	CSV integrity validation (NaN/inf/path)
250820_AU_attention1_250926best.py	Step 3	End-to-end AU + Global Fusion Attention training script
Step 0 — Face Cropping (Optional)

File: 1_facecrop_yolov8.py
This step is optional but useful when images contain multiple people or noisy backgrounds.

Key features:

Loads arnabdhar/YOLOv8-Face-Detection from HuggingFace Hub

PIL → OpenCV fallback for corrupted images

Selects the largest bounding box if multiple faces are detected

Preserves original directory structure

python 1_facecrop_yolov8.py
Step 1 — AU Patch Coordinate Extraction

File: 250809_AU_crop_csv_copy_3.py
This is the core preprocessing stage.

AU Region Definitions
Region	FaceMesh Landmark Index	Description
forehead	69, 299, 9	Left, right, center
eyes	159, 386	Left, right
nose	195	Nose ridge
cheeks	186, 410	Left, right
mouth	13	Upper lip
chin	18	Chin

→ Total 11 patches (multi-landmark regions expand to multiple patches)

Main Processing Steps

Unified resizing: Shorter side resized to --work-short (default 800 px)

Auto 180° correction: Choose orientation with larger inter-eye distance

CSV-only output: Instead of saving crops, store only coordinates to reduce disk load

Multiprocessing: Per-worker MediaPipe initialization using spawn mode

CSV Schema
path, label, rot_deg, work_w, work_h, patch_in, patch_out,
{au}_cx, {au}_cy,
{au}_wx1, {au}_wy1, {au}_wx2, {au}_wy2,
{au}_cx1, {au}_cy1, {au}_cx2, {au}_cy2
Usage
python 250809_AU_crop_csv_copy_3.py \
    --train /path/to/train \
    --val /path/to/val \
    --outdir /path/to/output \
    --work-short 800 \
    --patch-in 256 \
    --patch-out 256

Export actual patches from CSV:

python 250809_AU_crop_csv_copy_3.py --export-from-csv
Step 2 — CSV Validation

File: 250810_csv_check_nan.py

Checks:

Basic dataframe integrity (df.info(), df.head())

Column-wise NaN detection

Inf values in numeric fields

File existence check for path

Step 3 — AU Fusion Attention Model

File: 250820_AU_attention1_250926best.py

Model: AUFusionModel
Input Image ──→ GlobalEncoder (MobileNetV3 + DWConv Tail) ──→ g [B, d]
                                                                        │
AU patches × 11 ──→ PatchEncoder (MobileNetV3 + Linear) ──→ au [B, K, d]│
                             + AU Positional Encoding                   │
                                                                        │
                      ┌─────────────────────────────────────────────────┘
                      ▼
            [CLS] + g + au_tokens ──→ FusionBlock
                                      ├─ Self-Attention
                                      └─ Cross-Attention (Q=[CLS,g], KV=AU, gated by γ)
                                              │
                                              ▼
                                       FERHead (MLP)
                                            ↓
                                       logits [B, C]
Module Breakdown
Module	Purpose	Architecture
PatchEncoder	Encodes AU patches	MobileNetV3 → Linear → d
GlobalEncoder	Encodes full face	MobileNetV3 → 1×1 Conv → DWConv → 1×1 Conv → GAP → d
FusionBlock	Fuses global + AU tokens	Self-Attn + gated Cross-Attn
FERHead	Classification	LN → Linear → SiLU → Dropout → Linear
Training Strategy

Backbone warm-up: Freeze MobileNetV3 for first 5 epochs

Optimizer: AdamW (lr=2e-4, wd=0.05)

Scheduler: 5% warmup + cosine decay

Loss: CE + Label Smoothing (0.05) + class weights

AMP Enabled via torch.cuda.amp

Grad Clipping: 5.0

Model Selection: best macro-F1

Data Augmentation
Input	Train	Val
Global (224×224)	HFlip, ColorJitter	Resize
Patch (128×128)	HFlip	Resize
Both	ImageNet normalization	Same
Output Directory
output_dir/
  ├── best.pth
  ├── last.pth
  ├── confusion_best.png
  ├── report_best.txt
  └── history.jsonl

Usage:

python 250820_AU_attention1_250926best.py
Hyperparameters
Parameter	Value	Description
EPOCHS	300	Max epochs
BATCH_SIZE	32	Batch size
BASE_LR	2e-4	Initial LR
WEIGHT_DECAY	0.05	AdamW decay
LABEL_SMOOTHING	0.05	Smoothing factor
D_EMB	512	Embedding dim
N_HEAD	8	Attention heads
N_LAYERS	2	Transformer layers
GLOBAL_IMG_SIZE	224	Global input size
PATCH_OUT_SIZE	128	Patch size
FREEZE_BACKBONE_EPOCHS	5	Freeze duration
Dependencies
torch>=2.0
timm
torchvision
ultralytics
mediapipe
opencv-python
Pillow
numpy
pandas
scikit-learn
matplotlib
tqdm
huggingface_hub
Dataset

Source: AI Hub Korean Facial Expression Dataset

Structure: {split}/{emotion_label}/{image}

Labels are derived automatically from directory names.

License

TBD

Acknowledgments

AI Hub Korean Affect Dataset

MediaPipe FaceMesh

YOLOv8 Face Detection

timm PyTorch Image Models
