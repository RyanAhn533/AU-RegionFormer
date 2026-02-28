
AU-Based Facial Expression Recognition
End-to-End Pipeline Analysis & Documentation

AU Patch + Global Encoder + Cross-Attention Fusion Architecture


 ![Uploading Architecture.png…]()


Project Files Overview
Stage	File Name	Description
Stage 1	1_facecrop_yolov8.py	YOLOv8 face detection → crop and save face images from raw dataset
Stage 2-A	250809_AU_crop_csv.py	MediaPipe FaceMesh → AU landmark coordinates → CSV/NPZ (initial version)
Stage 2-B	2_AU_crop_csv_copy_3.py	Improved AU coordinate extraction: auto-orient, expanded AU, work resolution normalization
Stage 3	3_csv_check_nan.py	Data quality validation: NaN/inf check, missing file detection
Stage 4	4_AU_attention1_250926best.py	Full training: AU Patch Encoder + Global Encoder + Cross-Attention Fusion → FER classification

1. Pipeline Overview
This document describes the complete end-to-end pipeline for AU (Action Unit)-based Facial Expression Recognition (FER). The system takes raw face images from the AI Hub Korean Emotion Recognition dataset and produces emotion classification results through a multi-stage processing pipeline.

1.1 High-Level Flow
The pipeline consists of four sequential stages that transform raw images into trained emotion classification models:

Stage 1 Face Crop
YOLOv8	→	Stage 2 AU Extract
MediaPipe	→	Stage 3 Data QC
Pandas	→	Stage 4 Train
PyTorch


2. Stage 1: Face Detection & Cropping
File: 1_facecrop_yolov8.py
Purpose: Extract face regions from raw images using YOLOv8 face detection and save cropped face images while preserving the original directory structure.

2.1 Processing Flow
Load YOLOv8-Face-Detection model from HuggingFace (arnabdhar/YOLOv8-Face-Detection)
Walk through source directory tree recursively
For each image file (.jpg, .jpeg, .png): attempt to load via PIL, fallback to OpenCV if corrupted
Run YOLOv8 inference to detect face bounding boxes
Select the largest face (by area) if multiple faces detected
Crop the face region and save to destination with identical directory structure

2.2 Key Design Decisions
Dual image loader: PIL primary with OpenCV fallback handles truncated/corrupted images gracefully
Largest face selection: When multiple faces are detected, the largest bounding box is chosen, assuming the primary subject is most prominent
Directory structure preservation: os.path.relpath maintains the class/subfolder hierarchy for downstream label inference

2.3 Input / Output
Item	Details
Input	Raw images from AI Hub dataset (src_root directory tree, class-organized folders)
Output	Cropped face images with same directory/filename structure (dst_root)
Model	YOLOv8-Face-Detection (arnabdhar/YOLOv8-Face-Detection on HuggingFace)

3. Stage 2: AU Landmark Extraction & CSV Generation
This stage has two versions. Version A (250809_AU_crop_csv.py) is the initial implementation, and Version B (2_AU_crop_csv_copy_3.py) is the improved production version with auto-orientation, expanded AU landmarks, and work-resolution normalization.

3.1 Version A (Initial): 250809_AU_crop_csv.py
Core Logic: Force-resize images to a fixed size (224×224), run MediaPipe FaceMesh, compute AU center coordinates as pixel positions, and generate bounding boxes for each AU region.

AU Region Definitions (Version A)
AU Region	Landmark Indices	Description
forehead	(69, 10, 151, 299)	Average of 4 forehead landmarks
eyes	(159, 386)	Average of left/right eye landmarks
nose	5	Single nose tip landmark
cheeks	(205, 425)	Average of left/right cheek landmarks
mouth	13	Single upper lip center landmark
chin	200	Single chin landmark

Version A Output: CSV columns: path, label, ux0–ux5, uy0–uy5 (center pixel coordinates) + optional bx/by box coordinates. Also outputs NPZ with numpy arrays.

3.2 Version B (Improved): 2_AU_crop_csv_copy_3.py
This is the production version used for actual training data preparation. Key improvements over Version A:

Key Improvements
Work resolution normalization: Instead of forcing 224×224, images are resized by short-side to a configurable work_short (default 800px). This preserves aspect ratio and produces higher-quality landmark detection.
Auto-orientation (0°/180°): Tries both 0° and 180° rotation, selects the orientation with the largest inter-eye distance. This handles upside-down images from the dataset.
Expanded AU landmarks: Tuple-based landmarks are expanded into individual entries (e.g., forehead_69, forehead_299, forehead_9). This provides more fine-grained spatial information.
Rich CSV metadata: Stores rot_deg, work_w, work_h, patch_in, patch_out alongside per-AU coordinates (cx, cy, wx1, wy1, wx2, wy2, cx1, cy1, cx2, cy2).
spawn multiprocessing: Uses spawn instead of fork to avoid MediaPipe GPU context issues in child processes.
Preview mode: Saves sample overlay images for visual verification before processing the full dataset.

Expanded AU Definitions (Version B)
AU Region	Landmarks	Expanded Names
forehead	(69, 299, 9)	forehead_69, forehead_299, forehead_9
eyes	(159, 386)	eyes_159, eyes_386
nose	195	nose (single)
cheeks	(186, 410)	cheeks_186, cheeks_410
mouth	13	mouth (single)
chin	18	chin (single)

This results in a total of K=10 AU regions (3 + 2 + 1 + 2 + 1 + 1), each with its own coordinate columns in the CSV.

4. Stage 3: Data Quality Check
File: 3_csv_check_nan.py
Purpose: Validate the generated CSV before training to catch data issues early.

4.1 Validation Steps
Schema check: df.info() and df.head() to verify column types and sample data
NaN detection: df.isna().sum() per column to identify missing coordinates
Infinity check: np.isinf() on numeric columns to catch overflow values
File existence: Verify every image path in the CSV actually exists on disk

This stage is critical because NaN or missing values in AU coordinates will cause training to fail silently or produce degraded results. Non-existent paths will cause PIL.Image.open failures during data loading.

5. Stage 4: Model Training (AU Attention FER)
File: 4_AU_attention1_250926best.py
Purpose: Train the full AU-based facial expression recognition model using a dual-encoder architecture with cross-attention fusion.

5.1 Architecture Overview

Figure 1. AU Patch + Global Encoder + Cross-Attention Fusion Architecture
The model consists of four main components that work together in a hierarchical fusion scheme:

5.2 Component Details
5.2.1 Patch Encoder
Backbone: MobileNetV3-Small (pretrained on ImageNet via timm, model: mobilenetv3_small_100.lamb_in1k)
Input: K AU patches, each 128×128×3 (resized from CSV crop coordinates)
Process: Each patch is independently processed through MobileNetV3 feature extractor → global average pooling over spatial dimensions → linear projection from 576-dim to D_EMB (512). All K patches in a batch are processed together as B×K flattened inputs.
Output: [B, K, 512] tensor of AU patch embeddings

5.2.2 Global Encoder
Backbone: MobileNetV3-Small (same architecture, separate weights)
Input: Full face image 224×224×3
Tail CNN: After MobileNetV3 features (576-dim, 7×7 spatial), applies a series of 1×1 conv (576→1024), depthwise 7×7 conv (1024 groups), and 1×1 conv (1024→512), each with BatchNorm + SiLU activation. This captures global facial structure beyond individual AU regions.
Output: [B, 512] tensor of global face embedding

5.2.3 Fusion Block (Cross-Attention)
This is the core innovation of the architecture. It fuses local AU information with global face context through a two-stage attention mechanism:

Token construction: Concatenates [CLS token, Global embedding, AU_1, AU_2, ..., AU_K] into a sequence of K+2 tokens, each of dimension 512.
Self-Attention: Standard Transformer Encoder (N_LAYERS=2 layers, N_HEAD=8 heads) processes all tokens jointly. This allows AU patches to attend to each other and to the global context.
Cross-Attention: Query = [CLS, Global] (first 2 tokens), Key/Value = AU tokens (remaining K tokens). This lets the classification token and global representation explicitly query information from specific AU regions.
Gated Residual: Cross-attention output is added to the self-attention output via a learnable sigmoid gate (gamma parameter, initialized to 0). This allows the model to gradually learn to incorporate cross-attention information during training.
Layer Norm: Final LayerNorm normalizes the fused token representations.

5.2.4 FER Head (Classification)
Input: CLS token (first token from Fusion Block output, [B, 512])
Architecture: LayerNorm → Linear(512→512) → SiLU → Dropout(0.2) → Linear(512→num_classes)
Output: [B, num_classes] logits for emotion classification


5.3 Training Configuration
Parameter	Value / Description
Epochs	300
Batch Size	32
Base Learning Rate	2e-4
Optimizer	AdamW (weight_decay=0.05)
Scheduler	Warmup (5%) + Cosine Decay (min_lr=1e-6)
Label Smoothing	0.05
Gradient Clipping	Max norm = 5.0
Backbone Freeze	First 5 epochs (both Patch & Global MobileNetV3 backbones frozen)
Mixed Precision	AMP (torch.cuda.amp) on CUDA
Class Weighting	Inverse frequency weighting, normalized to mean=1
D_EMB	512 (embedding dimension)
N_HEAD	8 (attention heads)
N_LAYERS	2 (transformer encoder layers)
Global Image Size	224 × 224
Patch Size	128 × 128

5.4 Data Augmentation
Training augmentations are applied separately to global images and AU patches:

Global image (train): Resize(224) → RandomHorizontalFlip(0.5) → ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02) → ToTensor → Normalize
Patches (train): Resize(128) → RandomHorizontalFlip(0.5) → ToTensor → Normalize
Validation: Resize only → ToTensor → Normalize (no augmentation)
Normalization: Uses timm model-specific mean/std from mobilenetv3_small_100.lamb_in1k data config

5.5 Training Strategy
Backbone freezing: For the first 5 epochs, both MobileNetV3 backbones are frozen. Only the projection layers, tail CNN, fusion block, AU embeddings, CLS token, and classification head are trained. This stabilizes the randomly initialized components before fine-tuning the pretrained features.
Optimizer reset: At epoch 6, backbones are unfrozen, and a new optimizer/scheduler is created with all parameters. This effectively restarts the learning rate schedule for fine-tuning.
Best model selection: Based on validation macro-F1 score. Both best.pth and last.pth are saved each epoch.
Logging: JSONL history file, confusion matrix PNG, and classification report text file for the best epoch.


6. End-to-End Data Flow
Below is the complete data transformation chain from raw dataset to trained model, showing exact data formats at each stage boundary:

Stage	Input Format	Output Format
Stage 1	Raw images (variable size, class-organized dirs)	Cropped face images (variable size, same dir structure)
Stage 2	Cropped face images (from Stage 1 output)	CSV: path, label, rot_deg, work_w/h, patch_in/out, per-AU (cx, cy, wx1/wy1/wx2/wy2, cx1/cy1/cx2/cy2)
Stage 3	CSV from Stage 2	Validated CSV (same format, errors reported)
Stage 4	CSV + original images (runtime crop)	Trained model (best.pth, last.pth), confusion matrix, classification report, training history

6.1 Critical Coordinate System
Understanding the coordinate system is essential for pipeline consistency. The CSV stores coordinates in the work resolution coordinate space, not the original image resolution:

work_w, work_h: The actual dimensions of the image after resize-to-short-side (e.g., 800px short side). All coordinates are in this space.
wx1, wy1, wx2, wy2: The ideal crop window centered on the AU landmark. May extend outside image bounds.
cx1, cy1, cx2, cy2: The clamped (clipped to image bounds) crop window. Actual pixel region to extract.
At training time: The Dataset class reads the original image, resizes it to (work_w, work_h) to match the coordinate system, then crops using wx1/wy1/wx2/wy2 coordinates. This ensures pixel-perfect alignment between preprocessing and training.


7. Unified Pipeline Script Design
To run the entire pipeline end-to-end as a single continuous process, the following unified script structure is recommended. Each stage feeds directly into the next with proper error handling and checkpointing.

7.1 Recommended Unified Pipeline Structure

# ========== unified_pipeline.py ==========
 
def stage1_face_crop(src_root, dst_root):
    # Load YOLOv8 model
    # Walk src_root, detect & crop faces
    # Save to dst_root preserving dir structure
    # Return: count of processed, failed images
 
def stage2_au_extract(face_root_train, face_root_val, outdir,
                       work_short=800, patch_in=256, patch_out=256):
    # Initialize MediaPipe FaceMesh (spawn multiprocessing)
    # Auto-orient each image (0/180 degrees)
    # Extract expanded AU landmarks
    # Generate index_train.csv, index_val.csv
    # Optional: preview overlay images
    # Return: paths to generated CSVs
 
def stage3_validate(csv_path):
    # Check NaN, inf in coordinate columns
    # Verify all image paths exist
    # Report statistics, remove invalid rows
    # Return: cleaned CSV path, stats dict
 
def stage4_train(train_csv, val_csv, output_dir, config):
    # Build datasets and dataloaders
    # Initialize AUFusionModel
    # Freeze backbone -> warm up -> unfreeze -> train
    # Save best/last checkpoints, confusion matrix
    # Return: best_f1, best_epoch, model_path
 
def main():
    cfg = load_config('pipeline_config.yaml')
 
    # Stage 1
    stage1_face_crop(cfg.raw_data, cfg.cropped_data)
 
    # Stage 2
    csvs = stage2_au_extract(
        cfg.cropped_train, cfg.cropped_val, cfg.au_dir)
 
    # Stage 3
    clean_train = stage3_validate(csvs['train'])
    clean_val   = stage3_validate(csvs['val'])
 
    # Stage 4
    result = stage4_train(
        clean_train, clean_val, cfg.output_dir, cfg)
    print(f'Best F1: {result.best_f1}')

7.2 Pipeline Configuration Recommendations

Parameter	Recommended Value	Rationale
work_short	800	High enough for accurate FaceMesh, low enough for speed
patch_in / patch_out	256 / 128	Crop at 256 for context, resize to 128 for training efficiency
GLOBAL_IMG_SIZE	224	Standard MobileNetV3 input size
D_EMB	512	Balance between capacity and GPU memory
FREEZE_BACKBONE	5 epochs	Stabilize random heads before fine-tuning backbone
multiprocessing	spawn mode	Required for MediaPipe GPU context safety


8. Key Considerations & Potential Issues

8.1 Coordinate System Alignment
The most critical aspect of this pipeline is maintaining coordinate consistency between Stage 2 (AU extraction) and Stage 4 (training). The CSV stores coordinates in the work resolution space (e.g., after resize to 800px short side). At training time, the image must be resized to exactly (work_w, work_h) before applying the crop coordinates. Any mismatch will cause patches to be extracted from wrong regions.

8.2 Version Mismatch Risk
Version A (250809) and Version B (copy_3) use different landmark indices and coordinate schemes. The training script (Stage 4) dynamically discovers AU regions from CSV column names using the _wx1 suffix pattern, so it automatically adapts to either version. However, mixing CSVs from different versions for train/val would cause misaligned AU regions.

8.3 Memory & Performance
Stage 1: YOLOv8 inference is GPU-intensive. Process images sequentially to avoid OOM.
Stage 2: MediaPipe FaceMesh is CPU-only (GPU explicitly disabled). Uses multiprocessing with configurable worker count.
Stage 4: B×K patches are processed as a single batch (B*K forward passes through Patch Encoder). With B=32 and K=10, this means 320 forward passes per iteration, which is the memory bottleneck.

8.4 Failure Handling
Stage 1: Images with no detected faces are skipped (logged as WARN). Corrupted images fallback to OpenCV.
Stage 2-A: Failed FaceMesh detection uses random patch coordinates as fallback.
Stage 2-B: Failed detection returns None, row is excluded from CSV entirely.
Stage 4: Missing image paths generate zero-filled dummy tensors with a warning counter.
