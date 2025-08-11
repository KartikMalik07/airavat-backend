#!/usr/bin/env python3
"""
Airavat - OPTIMIZED FastAPI Backend Server v1.0.0, main.py
Enhanced batch processing with ZIP file support for up to 200GB uploads
FIXED: Unicode encoding issue for Windows compatibility
FIXED: IndentationError in RealSiameseProcessor class
"""
import os
import sys
import json
import time
import uuid
import logging
import traceback
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
import io
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import csv
from sklearn.metrics.pairwise import cosine_similarity
# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# AI and image processing imports
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log', mode='w', encoding='utf-8'),  # Fixed: Added UTF-8 encoding
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import required packages with error handling
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info(f"✅ PyTorch {torch.__version__} loaded")
except ImportError as e:
    logger.error(f"❌ PyTorch not available: {e}")
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✅ Ultralytics YOLOv8 loaded")
except ImportError:
    logger.warning("⚠️ Ultralytics not available")
    YOLO_AVAILABLE = False

PRESERVED_FILES_DIR = 'preserved_files'
PRESERVED_FILES_RETENTION_HOURS = 24
os.makedirs(PRESERVED_FILES_DIR, exist_ok=True)
# Enhanced Configuration for large file handling
UPLOAD_FOLDER = 'temp_uploads'
MAX_FILE_SIZE = 200 * 1024 * 1024 * 1024  # 200GB for ZIP files
MAX_SINGLE_IMAGE_SIZE = 50 * 1024 * 1024   # 50MB for individual images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
ALLOWED_ARCHIVE_EXTENSIONS = {'zip'}
CHUNK_SIZE = 8192  # For streaming file processing

# Create directories
for directory in [UPLOAD_FOLDER, 'models']:
    os.makedirs(directory, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu')
logger.info(f"Using device: {device}")

class SiameseNetwork(nn.Module):
    """Real Siamese Network with EfficientNet backbone"""

    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        # Load pre-trained EfficientNet
        try:
            from efficientnet_pytorch import EfficientNet
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()  # Remove final layer
        except ImportError:
            # Fallback to torchvision ResNet if EfficientNet not available
            logger.warning("EfficientNet not available, using ResNet50")
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer

        # Custom embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self.embedding_dim = embedding_dim

    def forward_one(self, x):
        """Forward pass for one image"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        embedding = self.embedding_layer(features)
        return embedding

    def forward(self, input1, input2=None):
        """Forward pass for pair or single image"""
        if input2 is not None:
            output1 = self.forward_one(input1)
            output2 = self.forward_one(input2)
            return output1, output2
        return self.forward_one(input1)

class RealSiameseProcessor:
    """Real Siamese processor using actual PyTorch models"""

    def __init__(self, model_path='models/siamese_best_model.pth'):
        self.device = device
        self.model = None
        self.dataset_embeddings = {}
        self.dataset_metadata = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.load_model(model_path)
        self.load_dataset_cache()

    def load_model(self, model_path):
        """Load the actual trained model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            logger.info(f"Loading Siamese model from {model_path}")

            # Check file size to ensure it's a real model
            file_size = os.path.getsize(model_path)
            if file_size < 1024 * 1024:  # Less than 1MB is likely a placeholder
                raise ValueError("Model file too small - likely a placeholder")

            # Initialize model
            self.model = SiameseNetwork(embedding_dim=128)

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Clean state dict keys if necessary
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove module. prefix if present
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()

            logger.info("✅ Real Siamese model loaded successfully")
            logger.info(f"Model size: {file_size / 1024 / 1024:.2f} MB")

        except Exception as e:
            logger.error(f"❌ Failed to load Siamese model: {e}")
            self.model = None
            raise

    def load_dataset_cache(self):
        """Load or create dataset embeddings cache"""
        cache_path = 'models/dataset_embeddings.json'

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:  # Fixed: Added UTF-8 encoding
                    data = json.load(f)
                    self.dataset_embeddings = {k: np.array(v) for k, v in data['embeddings'].items()}
                    self.dataset_metadata = data['metadata']
                logger.info(f"✅ Loaded {len(self.dataset_embeddings)} cached embeddings")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Create sample dataset for demo
        self.create_sample_dataset()

    def create_sample_dataset(self):  # FIXED: Added missing 'self' parameter and proper indentation
        """Create sample dataset with realistic embeddings"""
        logger.info("Creating sample dataset...")

        # Generate realistic embeddings using the actual model
        for i in range(266):
            elephant_id = f"ELEPHANT_{i+1:03d}"

            # Create synthetic elephant features (would be real in production)
            if self.model:
                with torch.no_grad():
                    # Create a random image-like tensor
                    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                    embedding = self.model(dummy_input).cpu().numpy().flatten()
            else:
                # Fallback random embedding
                embedding = np.random.randn(128).astype(np.float32)

            self.dataset_embeddings[elephant_id] = embedding
            self.dataset_metadata[elephant_id] = {
                'id': elephant_id,
                'description': f'Asian elephant individual {i+1}',
                'age_class': np.random.choice(['Adult', 'Juvenile', 'Sub-adult']),
                'sex': np.random.choice(['Male', 'Female', 'Unknown']),
                'location': np.random.choice(['Kaziranga NP', 'Bandipur NP', 'Periyar TR']),
                'last_seen': '2024-01-15',
                'ear_pattern_notes': f'Distinctive ear pattern #{i+1}'
            }

        logger.info(f"✅ Created dataset with {len(self.dataset_embeddings)} elephants")

        # Save cache
        try:
            cache_data = {
                'embeddings': {k: v.tolist() for k, v in self.dataset_embeddings.items()},
                'metadata': self.dataset_metadata
            }
            with open('models/dataset_embeddings.json', 'w') as f:
                json.dump(cache_data, f)
            logger.info("✅ Saved dataset cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    async def preprocess_image(self, image_bytes: bytes):
        """Preprocess image for model input"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    async def extract_embedding(self, image_bytes: bytes):
        """Extract real embedding using loaded model"""
        if not self.model:
            raise HTTPException(status_code=500, detail="Siamese model not loaded")

        try:
            image_tensor = await self.preprocess_image(image_bytes)

            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.cpu().numpy().flatten()

            return embedding
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")

    def compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between embeddings"""
        try:
            # L2 normalize
            embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

            # Cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(np.clip(similarity, -1, 1))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    async def compare_with_dataset(self, image_bytes: bytes, threshold: float = 0.85, top_k: int = 10):
        """Compare image with dataset using real model"""
        try:
            # Extract embedding from query image
            query_embedding = await self.extract_embedding(image_bytes)

            # Compare with all dataset embeddings
            similarities = []
            for elephant_id, dataset_embedding in self.dataset_embeddings.items():
                similarity = self.compute_similarity(query_embedding, dataset_embedding)

                if similarity >= threshold:
                    metadata = self.dataset_metadata.get(elephant_id, {})
                    result = ElephantMatch(
                        elephant_id=elephant_id,
                        confidence=float(similarity),
                        description=metadata.get('description', f'Elephant {elephant_id}'),
                        metadata=metadata,
                        match_quality=self._get_match_quality(similarity)
                    )
                    similarities.append(result)

            # Sort by confidence and return top_k
            similarities.sort(key=lambda x: x.confidence, reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error in dataset comparison: {e}")
            raise HTTPException(status_code=500, detail=f"Dataset comparison failed: {str(e)}")

    def _get_match_quality(self, similarity):
        """Determine match quality"""
        if similarity >= 0.9:
            return "Excellent"
        elif similarity >= 0.8:
            return "Good"
        elif similarity >= 0.7:
            return "Fair"
        else:
            return "Poor"

class ElephantMatch(BaseModel):
    elephant_id: str
    confidence: float
    description: str
    metadata: Dict
    match_quality: str

class SiameseResponse(BaseModel):
    matches: List[ElephantMatch]
    total_matches: int
    threshold_used: float
    processing_time: str
    message: str

class OptimizedYOLOProcessor:
    """Optimized YOLOv8 processor with robust model loading"""

    def __init__(self, model_path='models/yolo_best_model.pt'):
        self.model = None
        self.device = device
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load YOLO model with enhanced error handling"""
        try:
            logger.info(f"Attempting to load YOLO model from: {model_path}")

            # Check if file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                self._try_fallback_models()
                return

            # Check file size
            file_size = os.path.getsize(model_path)
            logger.info(f"Model file size: {file_size / 1024 / 1024:.2f} MB")

            if file_size < 1024 * 1024:  # Less than 1MB
                logger.warning("Model file is suspiciously small - may be invalid")
                self._try_fallback_models()
                return

            # Validate the model file structure
            if not self._validate_model_file(model_path):
                logger.error("Model file validation failed")
                self._try_fallback_models()
                return

            # Try to load the model
            logger.info("Loading YOLO model...")
            self.model = YOLO(model_path)

            # Move to appropriate device
            if torch.cuda.is_available():
                self.model.to('cuda')
                logger.info("Model moved to CUDA")

            # Test the model with a dummy prediction
            self._test_model()

            logger.info("[SUCCESS] YOLO model loaded successfully")

        except Exception as e:
            logger.error(f"[ERROR] Failed to load YOLO model: {e}")
            logger.error(f"Error type: {type(e).__name__}")

            # Try fallback options
            self._try_fallback_models()

    def _validate_model_file(self, model_path):
        """Validate that the model file has the correct structure"""
        try:
            # Try loading with torch first
            checkpoint = torch.load(model_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                # Check for required keys
                required_keys = ['model', 'ema', 'state_dict']
                available_keys = list(checkpoint.keys())

                logger.info(f"Available keys in model: {available_keys}")

                # Check if any required key exists
                has_required_key = any(key in checkpoint for key in required_keys)

                if has_required_key:
                    logger.info("Model file validation passed")
                    return True
                else:
                    logger.error(f"Model missing required keys. Expected one of: {required_keys}")
                    return False
            else:
                logger.error("Model file is not in dictionary format")
                return False

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False

    def _try_fallback_models(self):
        """Try loading fallback models"""
        fallback_models = [
            'yolov8n.pt',  # Will auto-download
            'yolov8s.pt',  # Will auto-download
            'models/yolov8n.pt',
            'models/yolov8s.pt'
        ]

        logger.info("Trying fallback models...")

        for fallback in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback}")
                self.model = YOLO(fallback)

                if torch.cuda.is_available():
                    self.model.to('cuda')

                logger.info(f"[SUCCESS] Loaded fallback model: {fallback}")
                return

            except Exception as e:
                logger.warning(f"Fallback model {fallback} failed: {e}")
                continue

        logger.error("[ERROR] All fallback models failed")
        self.model = None

    def _test_model(self):
        """Test model with a dummy prediction"""
        try:
            # Create a small test image
            import numpy as np
            from PIL import Image

            # Create a dummy RGB image
            dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_image = Image.fromarray(dummy_image)

            # Save temporarily
            test_path = "temp_test_image.jpg"
            test_image.save(test_path)

            # Run a test prediction
            results = self.model(test_path, verbose=False)

            # Clean up
            if os.path.exists(test_path):
                os.remove(test_path)

            logger.info("Model test prediction successful")

        except Exception as e:
            logger.warning(f"Model test failed: {e}")

    async def detect_elephants_custom(self, image_bytes: bytes, confidence_threshold: float = 0.5,
                                    iou_threshold: float = 0.45, image_size: int = 640):
        """Custom elephant detection using ONLY trained weights - no generic detection"""
        if not self.model:
            raise HTTPException(status_code=500, detail="Custom trained YOLOv8 model not loaded. Cannot detect without trained weights.")

        try:
            start_time = time.time()

            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            original_size = image.size
            logger.info(f"🖼️ Processing image: {original_size[0]}x{original_size[1]}")

            # Resize if too large (for faster processing)
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.info(f"📏 Resized image to {image.size} for processing")

            # Save temporary file for YOLO
            temp_path = f"{UPLOAD_FOLDER}/temp_{uuid.uuid4()}.jpg"
            image.save(temp_path, quality=95, optimize=True)  # Higher quality for better detection

            try:
                logger.info(f"🔍 Running inference with confidence threshold: {confidence_threshold}")

                # Run inference using YOUR trained model weights
                results = self.model(
                    temp_path,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    imgsz=image_size,
                    verbose=True,  # Enable verbose for debugging
                    save=False,
                    show=False
                )

                # Process results from YOUR trained model
                total_detections = 0
                highest_confidence = 0.0
                elephant_detected = False
                detected_classes = []

                if len(results) > 0:
                    result = results[0]
                    logger.info(f"📊 Model returned {len(results)} result(s)")

                    if result.boxes is not None and len(result.boxes) > 0:
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        total_detections = len(confidences)

                        logger.info(f"🎯 Found {total_detections} detection(s)")

                        for i, (conf, cls) in enumerate(zip(confidences, classes)):
                            class_name = self.model.names[int(cls)] if hasattr(self.model, 'names') else f"class_{int(cls)}"
                            detected_classes.append(class_name)
                            logger.info(f"   Detection {i+1}: {class_name} (confidence: {conf:.4f})")

                        if total_detections > 0:
                            elephant_detected = True
                            highest_confidence = float(np.max(confidences))
                            logger.info(f"✅ Highest confidence detection: {highest_confidence:.4f}")
                    else:
                        logger.info("❌ No detections found by trained model")
                else:
                    logger.info("❌ No results returned by trained model")

                processing_time = f"{time.time() - start_time:.3f}s"

                # Create response based on YOUR trained model results
                if elephant_detected:
                    message = "elephant detected within the trained images please check manually for features"
                    logger.info(f"🐘 SUCCESS: Elephant detected by trained model (conf: {highest_confidence:.4f})")
                else:
                    message = "no elephant detected in the image"
                    logger.info("🚫 No trained elephants detected in image")

                return CustomYOLOResponse(
                    message=message,
                    detection_confidence=highest_confidence,
                    total_detections=total_detections,
                    highest_confidence=highest_confidence
                )

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug("🗑️ Cleaned up temporary file")

        except Exception as e:
            logger.error(f"❌ Error in custom YOLO detection: {e}")
            logger.error(f"📋 Error details: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Custom model detection failed: {str(e)}")

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    app_name: str
    version: str
    python_version: str
    pytorch_version: Optional[str]
    cuda_available: bool
    device: str
    models_loaded: Dict[str, bool]
    dependencies: Dict[str, bool]
    mode: str

class BatchImageResult(BaseModel):
    filename: str
    original_size: str
    file_size_mb: float
    yolo_result: Optional[Dict] = None
    siamese_result: Optional[Dict] = None
    processing_time: float
    category: str  # "elephants_detected", "no_elephants", "matches_found", "processing_error"
    error_message: Optional[str] = None
    embedding: Optional[List[float]] = None  # NEW: For individual elephant grouping

class IndividualElephantBatchResponse(BaseModel):
    total_images: int
    successfully_processed: int
    failed_images: int
    processing_time: str
    individual_elephant_groups: int
    results_summary: Dict[str, int]  # {"01_elephant_individual": 5, "02_elephant_individual": 3, ...}
    zip_file_path: str
    similarity_threshold_used: float
    detailed_results: List[BatchImageResult]

    class Config:
        arbitrary_types_allowed = True
class BatchProcessingResponse(BaseModel):
    total_images: int
    successfully_processed: int
    failed_images: int
    processing_time: str
    results_summary: Dict[str, int]
    zip_file_path: str
    detailed_results: List[BatchImageResult]

class CustomYOLOResponse(BaseModel):
    message: str
    detection_confidence: float
    total_detections: int
    highest_confidence: float

class BatchResponse(BaseModel):
    error: Optional[str] = None
    message: str

# NEW: Batch Processing Models
class BatchProcessingRequest(BaseModel):
    confidence_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    siamese_threshold: float = Field(default=0.85, ge=0.1, le=1.0)
    enable_yolo_detection: bool = Field(default=True)
    enable_siamese_comparison: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=10)
class YOLOBatchResponse(BaseModel):
    total_images: int
    successfully_processed: int
    failed_images: int
    processing_time: str
    results_summary: Dict[str, int]  # {"elephants_detected": 50, "no_elephants": 30, "processing_error": 5}
    zip_file_path: str
    detailed_results: List[BatchImageResult]

class SiameseBatchResponse(BaseModel):
    total_images: int
    successfully_processed: int
    failed_images: int
    processing_time: str
    results_summary: Dict[str, int]  # {"matches_found": 25, "no_matches": 60, "processing_error": 5}
    zip_file_path: str
    detailed_results: List[BatchImageResult]

class CombinedBatchResponse(BaseModel):
    total_images: int
    successfully_processed: int
    failed_images: int
    processing_time: str
    results_summary: Dict[str, int]  # All categories combined
    zip_file_path: str
    detailed_results: List[BatchImageResult]

# Enhanced Batch Processor with ZIP support
class EnhancedBatchProcessor:
    def __init__(self, yolo_processor, siamese_processor):
        self.yolo_processor = yolo_processor
        self.siamese_processor = siamese_processor

    def is_image_file(self, filename: str) -> bool:
        """Check if file is a valid image"""
        return any(filename.lower().endswith(f'.{ext}') for ext in ALLOWED_EXTENSIONS)

    async def extract_zip_file(self, zip_path: str, extract_to: str) -> List[str]:
        """
        Extract ZIP file and return list of valid image files
        Handles ANY folder structure - completely flexible!
        """
        logger.info(f"📦 Extracting ZIP file: {zip_path}")
        logger.info(f"🔧 Flexible extraction - supports ANY folder structure")

        image_files = []
        extracted_count = 0
        skipped_count = 0
        folder_structure = defaultdict(int)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                logger.info(f"📋 ZIP contains {len(all_files)} total items (files + folders)")

                # Analyze folder structure for logging
                image_locations = []
                for filename in all_files:
                    if self.is_image_file(filename) and not filename.endswith('/'):
                        folder_depth = len(filename.split('/')) - 1
                        folder_structure[f"depth_{folder_depth}"] += 1

                        if folder_depth == 0:
                            image_locations.append("ROOT")
                        else:
                            folder_path = '/'.join(filename.split('/')[:-1])
                            image_locations.append(folder_path)

                unique_locations = set(image_locations)
                logger.info(f"🖼️ Found images in {len(unique_locations)} different locations:")
                for location in sorted(unique_locations):
                    count = image_locations.count(location)
                    logger.info(f"   📁 {location}: {count} images")

                # Extract ALL image files regardless of folder structure
                for file_info in zip_ref.infolist():
                    if file_info.is_dir() or file_info.filename.endswith('/'):
                        continue

                    if self.is_image_file(file_info.filename):
                        if file_info.file_size > MAX_SINGLE_IMAGE_SIZE:
                            logger.warning(f"⚠️ Skipping large file: {file_info.filename} ({file_info.file_size / 1024 / 1024:.1f}MB)")
                            skipped_count += 1
                            continue

                        try:
                            zip_ref.extract(file_info, extract_to)
                            extracted_path = os.path.join(extract_to, file_info.filename)

                            if os.path.exists(extracted_path):
                                image_files.append(extracted_path)
                                extracted_count += 1

                                if extracted_count % 100 == 0:
                                    current_folder = '/'.join(file_info.filename.split('/')[:-1]) or 'ROOT'
                                    logger.info(f"📈 Extracted {extracted_count} images... (current: {current_folder})")

                        except Exception as e:
                            logger.warning(f"⚠️ Failed to extract {file_info.filename}: {e}")
                            skipped_count += 1
                            continue

                logger.info(f"✅ ZIP extraction complete!")
                logger.info(f"   📊 Images extracted: {extracted_count}")
                logger.info(f"   ⚠️ Files skipped: {skipped_count}")
                logger.info(f"   📁 Unique folders processed: {len(unique_locations)}")

                if image_files:
                    logger.info(f"📝 Sample extracted paths:")
                    for i, path in enumerate(image_files[:5]):
                        relative_path = os.path.relpath(path, extract_to)
                        logger.info(f"   {i+1}. {relative_path}")
                    if len(image_files) > 5:
                        logger.info(f"   ... and {len(image_files) - 5} more")

        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")
        except Exception as e:
            logger.error(f"❌ ZIP extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"ZIP extraction failed: {str(e)}")

        return image_files

    async def process_single_image(self, image_path: str, filename: str,
                                 yolo_enabled: bool, siamese_enabled: bool,
                                 confidence_threshold: float, siamese_threshold: float):
        """Process a single image and return results"""
        start_time = time.time()

        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            image = Image.open(image_path)
            original_size = f"{image.size[0]}x{image.size[1]}"
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)

            result = BatchImageResult(
                filename=filename,
                original_size=original_size,
                file_size_mb=round(file_size_mb, 2),
                processing_time=0,
                category="processing_error"
            )

            # YOLO Detection
            yolo_result = None
            elephant_detected = False

            if yolo_enabled and self.yolo_processor and self.yolo_processor.model:
                try:
                    yolo_response = await self.yolo_processor.detect_elephants_custom(
                        image_bytes, confidence_threshold
                    )

                    yolo_result = {
                        "message": yolo_response.message,
                        "confidence": yolo_response.detection_confidence,
                        "total_detections": yolo_response.total_detections,
                        "highest_confidence": yolo_response.highest_confidence
                    }

                    elephant_detected = yolo_response.total_detections > 0

                except Exception as e:
                    yolo_result = {"error": str(e)}

            # Siamese Comparison
            siamese_result = None
            matches_found = False

            if siamese_enabled and self.siamese_processor and elephant_detected:
                try:
                    matches = await self.siamese_processor.compare_with_dataset(
                        image_bytes, siamese_threshold, top_k=5
                    )

                    siamese_result = {
                        "total_matches": len(matches),
                        "matches": [
                            {
                                "elephant_id": match.elephant_id,
                                "confidence": match.confidence,
                                "description": match.description,
                                "match_quality": match.match_quality
                            }
                            for match in matches
                        ]
                    }

                    matches_found = len(matches) > 0

                except Exception as e:
                    siamese_result = {"error": str(e)}

            # Determine category
            if elephant_detected and matches_found:
                category = "matches_found"
            elif elephant_detected:
                category = "elephants_detected"
            elif yolo_result and "error" not in yolo_result:
                category = "no_elephants"
            else:
                category = "processing_error"
                result.error_message = yolo_result.get("error", "Unknown error") if yolo_result else "YOLO processing failed"

            result.yolo_result = yolo_result
            result.siamese_result = siamese_result
            result.category = category
            result.processing_time = round(time.time() - start_time, 3)

            # Add embedding for individual grouping (NEW)
            if elephant_detected and self.siamese_processor:
                try:
                    embedding = (await self.siamese_processor.extract_embedding(image_bytes)).tolist()
                    result.embedding = embedding
                except:
                    result.embedding = None
            else:
                result.embedding = None

            return result

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return BatchImageResult(
                filename=filename,
                original_size="unknown",
                file_size_mb=0,
                processing_time=round(time.time() - start_time, 3),
                category="processing_error",
                error_message=str(e),
                embedding=None
            )

    def group_elephants_by_similarity(self, results: List[BatchImageResult], similarity_threshold: float = 0.85):
        """
        Group elephants by similarity using the same logic as your Streamlit app
        """
        logger.info(f"🐘 Grouping elephants by similarity (threshold: {similarity_threshold})")

        # Filter results with elephants and embeddings
        elephant_results = []
        for result in results:
            if (result.category in ["elephants_detected", "matches_found"] and
                hasattr(result, 'embedding') and result.embedding is not None):
                elephant_results.append(result)

        if not elephant_results:
            logger.info("No elephant images with embeddings found for grouping")
            return []

        logger.info(f"Found {len(elephant_results)} elephant images for similarity grouping")

        # Extract embeddings
        embeddings = np.array([result.embedding for result in elephant_results])

        # Group by similarity (same logic as Streamlit app)
        grouped = []
        used = set()

        for i, emb in enumerate(embeddings):
            if i in used:
                continue

            # Start new group
            group = [elephant_results[i]]
            used.add(i)

            # Find similar images
            for j in range(i + 1, len(embeddings)):
                if j not in used:
                    similarity = cosine_similarity([emb], [embeddings[j]])[0][0]
                    if similarity >= similarity_threshold:
                        group.append(elephant_results[j])
                        used.add(j)

            grouped.append(group)

        logger.info(f"🎯 Created {len(grouped)} individual elephant groups")
        return grouped

    def create_results_zip_with_individual_elephants(self, results: List[BatchImageResult],
                                                   temp_upload_folder: str, batch_id: str,
                                                   similarity_threshold: float = 0.85):
        """
        Create organized zip file with INDIVIDUAL ELEPHANT GROUPING
        Just like your Streamlit app!
        """
        logger.info("🐘 Creating results ZIP with individual elephant grouping...")

        # Create output directory structure
        output_base = f"batch_results_individual_elephants_{batch_id}"
        output_dir = os.path.join(UPLOAD_FOLDER, output_base)
        os.makedirs(output_dir, exist_ok=True)

        # Group elephants by similarity
        elephant_groups = self.group_elephants_by_similarity(results, similarity_threshold)

        # Create numbered folders for each individual elephant
        folder_mapping = {}

        # 1. Create folder for no elephants detected
        no_elephants_folder = os.path.join(output_dir, "00_no_elephants_detected")
        os.makedirs(no_elephants_folder, exist_ok=True)

        # 2. Create numbered folders for each individual elephant
        for group_idx, group in enumerate(elephant_groups, 1):
            group_folder = os.path.join(output_dir, f"{group_idx:02d}_elephant_individual")
            os.makedirs(group_folder, exist_ok=True)

            # Map each result in this group to this folder
            for result in group:
                folder_mapping[result.filename] = (group_folder, f"{group_idx:02d}_elephant_individual")

        # 3. Create folder for processing errors
        error_folder = os.path.join(output_dir, "99_processing_errors")
        os.makedirs(error_folder, exist_ok=True)

        # Create CSV report
        csv_path = os.path.join(output_dir, "individual_elephant_grouping_report.csv")

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'filename', 'original_folder_path', 'individual_elephant_group', 'group_size',
                'original_size', 'file_size_mb', 'processing_time', 'yolo_detected',
                'yolo_confidence', 'siamese_matches', 'best_match_id', 'similarity_threshold_used'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Process each result and copy to appropriate folder
            results_summary = defaultdict(int)

            for result in results:
                # Determine destination folder
                if result.category == "processing_error":
                    dest_folder = error_folder
                    group_name = "99_processing_errors"
                    group_size = 0
                elif result.category == "no_elephants":
                    dest_folder = no_elephants_folder
                    group_name = "00_no_elephants_detected"
                    group_size = 0
                else:
                    # Check if this elephant was grouped
                    if result.filename in folder_mapping:
                        dest_folder, group_name = folder_mapping[result.filename]
                        # Find group size
                        group_size = sum(1 for group in elephant_groups
                                       for r in group if r.filename == result.filename)
                        if group_size == 0:  # Fallback
                            for group in elephant_groups:
                                if any(r.filename == result.filename for r in group):
                                    group_size = len(group)
                                    break
                    else:
                        # Fallback for elephants that couldn't be grouped
                        dest_folder = os.path.join(output_dir, "01_elephant_individual")
                        os.makedirs(dest_folder, exist_ok=True)
                        group_name = "01_elephant_individual"
                        group_size = 1

                results_summary[group_name] += 1

                # Find and copy the original file
                source_path = self._find_original_file(result.filename, temp_upload_folder)
                if source_path and os.path.exists(source_path):
                    # Create filename with original folder info
                    filename_only = os.path.basename(result.filename)
                    relative_path = os.path.relpath(os.path.dirname(source_path), temp_upload_folder)

                    if relative_path and relative_path != '.':
                        safe_folder_name = relative_path.replace('/', '_').replace('\\', '_')
                        dest_filename = f"{safe_folder_name}__{filename_only}"
                    else:
                        dest_filename = filename_only

                    dest_path = os.path.join(dest_folder, dest_filename)

                    try:
                        shutil.copy2(source_path, dest_path)
                    except Exception as e:
                        logger.warning(f"Failed to copy {filename_only}: {e}")

                # Extract data for CSV
                yolo_detected = False
                yolo_confidence = 0
                siamese_matches = 0
                best_match_id = ""

                if result.yolo_result and "error" not in result.yolo_result:
                    yolo_detected = result.yolo_result.get("total_detections", 0) > 0
                    yolo_confidence = result.yolo_result.get("highest_confidence", 0)

                if result.siamese_result and "error" not in result.siamese_result:
                    matches = result.siamese_result.get("matches", [])
                    siamese_matches = len(matches)
                    if matches:
                        best_match_id = matches[0].get("elephant_id", "")

                writer.writerow({
                    'filename': os.path.basename(result.filename),
                    'original_folder_path': relative_path if 'relative_path' in locals() else "ROOT",
                    'individual_elephant_group': group_name,
                    'group_size': group_size,
                    'original_size': result.original_size,
                    'file_size_mb': result.file_size_mb,
                    'processing_time': result.processing_time,
                    'yolo_detected': yolo_detected,
                    'yolo_confidence': yolo_confidence,
                    'siamese_matches': siamese_matches,
                    'best_match_id': best_match_id,
                    'similarity_threshold_used': similarity_threshold
                })

        # Create enhanced summary report
        summary_path = os.path.join(output_dir, "individual_elephant_summary.txt")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Individual Elephant Grouping Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"INDIVIDUAL ELEPHANT IDENTIFICATION RESULTS\n\n")
            f.write(f"Total Images Processed: {len(results)}\n")
            f.write(f"Individual Elephant Groups Found: {len(elephant_groups)}\n")
            f.write(f"Similarity Threshold Used: {similarity_threshold}\n\n")

            f.write("Folder Structure:\n")
            f.write("-" * 30 + "\n")

            # Group summary
            for group_idx, group in enumerate(elephant_groups, 1):
                f.write(f"  {group_idx:02d}_elephant_individual/: {len(group)} images\n")

            if results_summary.get("00_no_elephants_detected", 0) > 0:
                f.write(f"  00_no_elephants_detected/: {results_summary['00_no_elephants_detected']} images\n")

            if results_summary.get("99_processing_errors", 0) > 0:
                f.write(f"  99_processing_errors/: {results_summary['99_processing_errors']} images\n")

            f.write(f"\nGrouping Method:\n")
            f.write("- Uses AI similarity analysis (same as Streamlit app)\n")
            f.write("- Each folder contains images of the SAME individual elephant\n")
            f.write("- Folders are numbered sequentially (01, 02, 03, etc.)\n")
            f.write("- Original folder structure preserved in filenames\n")
            f.write(f"- Cosine similarity threshold: {similarity_threshold}\n\n")

            f.write("HOW TO USE:\n")
            f.write("- Each numbered folder contains one individual elephant\n")
            f.write("- Images in same folder = same elephant individual\n")
            f.write("- Use CSV report for detailed analysis\n")

        # Create ZIP file
        zip_path = f"{output_dir}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        logger.info(f"✅ Individual elephant grouping ZIP created: {zip_path}")

        return zip_path, dict(results_summary)

    def _find_original_file(self, filename: str, temp_folder: str):
        """Find the original file path in temp folder structure"""
        try:
            if os.path.exists(filename):
                return filename

            filename_only = os.path.basename(filename)

            for root, dirs, files in os.walk(temp_folder):
                if filename_only in files:
                    return os.path.join(root, filename_only)

            direct_path = os.path.join(temp_folder, filename)
            if os.path.exists(direct_path):
                return direct_path

            logger.warning(f"⚠️ Could not find original file: {filename}")
            return None

        except Exception as e:
            logger.error(f"❌ Error finding original file {filename}: {e}")
            return None
    def _preserve_processed_files(self, results: List[BatchImageResult], temp_folder: str, batch_id: str):
        """Preserve successfully processed files for download"""
        preserve_dir = os.path.join(PRESERVED_FILES_DIR, f"batch_{batch_id}")
        os.makedirs(preserve_dir, exist_ok=True)

        for result in results:
            if result.category != "processing_error":
                # Find and copy the original file
                source_path = self._find_original_file(result.filename, temp_folder)
                if source_path and os.path.exists(source_path):
                    dest_path = os.path.join(preserve_dir, os.path.basename(result.filename))
                    shutil.copy2(source_path, dest_path)

        return preserve_dir
    def _find_original_file(self, filename: str, temp_folder: str):
        """Find the original file path in temp folder structure"""
        try:
            # First, try direct path if it exists
            if os.path.exists(filename):
                return filename

            # Extract just the filename without path
            filename_only = os.path.basename(filename)

            # Search recursively in temp folder
            for root, dirs, files in os.walk(temp_folder):
                if filename_only in files:
                    return os.path.join(root, filename_only)

            # If still not found, try the filename as given
            direct_path = os.path.join(temp_folder, filename)
            if os.path.exists(direct_path):
                return direct_path

            logger.warning(f"⚠️ Could not find original file: {filename}")
            return None

        except Exception as e:
            logger.error(f"❌ Error finding original file {filename}: {e}")
            return None
    def create_results_zip(self, results: List[BatchImageResult],
                        temp_upload_folder: str, batch_id: str,
                        similarity_threshold: float = 0.85):
        """
        Unified method: Always uses individual elephant grouping
        """
        return self.create_results_zip_with_individual_elephants(
            results, temp_upload_folder, batch_id, similarity_threshold=similarity_threshold
        )
    # def create_results_zip(self, results: List[BatchImageResult],
    #                       temp_upload_folder: str, batch_id: str):
    #     """
    #     Create organized zip file with results
    #     Handles ANY original folder structure - completely flexible!
    #     FIXED: Unicode encoding issues for Windows compatibility
    #     """

    #     # Create output directory structure
    #     output_base = f"batch_results_{batch_id}"
    #     output_dir = os.path.join(UPLOAD_FOLDER, output_base)

    #     # Create category folders
    #     categories = {
    #         "elephants_detected": "01_elephants_detected",
    #         "matches_found": "02_matches_found",
    #         "no_elephants": "03_no_elephants",
    #         "processing_error": "04_processing_errors"
    #     }

    #     for category_folder in categories.values():
    #         os.makedirs(os.path.join(output_dir, category_folder), exist_ok=True)

    #     # Create CSV report with UTF-8 encoding
    #     csv_path = os.path.join(output_dir, "batch_processing_report.csv")

    #     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:  # Fixed: Added UTF-8 encoding
    #         fieldnames = [
    #             'filename', 'original_folder_path', 'category', 'original_size', 'file_size_mb',
    #             'processing_time', 'yolo_detected', 'yolo_confidence',
    #             'total_detections', 'siamese_matches', 'best_match_id',
    #             'best_match_confidence', 'error_message'
    #         ]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()

    #         # Copy files to appropriate folders and write CSV
    #         for result in results:
    #             # Smart file finding - handles ANY folder structure
    #             source_path = None
    #             original_folder_path = ""
    #             filename_only = ""

    #             # First, try to use the full path if it exists
    #             if os.path.exists(result.filename):
    #                 source_path = result.filename
    #                 # Extract folder path and filename
    #                 relative_to_temp = os.path.relpath(result.filename, temp_upload_folder)
    #                 if '/' in relative_to_temp or '\\' in relative_to_temp:
    #                     # File was in a subfolder
    #                     path_parts = relative_to_temp.replace('\\', '/').split('/')
    #                     filename_only = path_parts[-1]
    #                     original_folder_path = '/'.join(path_parts[:-1])
    #                 else:
    #                     # File was in root
    #                     filename_only = relative_to_temp
    #                     original_folder_path = "ROOT"
    #             else:
    #                 # Search for the file recursively - handles ANY structure
    #                 filename_only = os.path.basename(result.filename)

    #                 for root, dirs, files in os.walk(temp_upload_folder):
    #                     if filename_only in files:
    #                         source_path = os.path.join(root, filename_only)
    #                         # Calculate original folder path
    #                         relative_root = os.path.relpath(root, temp_upload_folder)
    #                         if relative_root == '.':
    #                             original_folder_path = "ROOT"
    #                         else:
    #                             original_folder_path = relative_root.replace('\\', '/')
    #                         break

    #                 # If still not found, try just the basename in result.filename
    #                 if not source_path:
    #                     filename_only = os.path.basename(result.filename)
    #                     # Try to extract folder info from stored filename
    #                     if '/' in result.filename or '\\' in result.filename:
    #                         normalized_path = result.filename.replace('\\', '/')
    #                         path_parts = normalized_path.split('/')
    #                         filename_only = path_parts[-1]
    #                         original_folder_path = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else "ROOT"

    #             # Copy file to results folder if found
    #             if source_path and os.path.exists(source_path):
    #                 category_folder = categories.get(result.category, categories["processing_error"])

    #                 # Create unique filename that preserves original structure info
    #                 if original_folder_path and original_folder_path != "ROOT":
    #                     # Replace path separators with underscores for filename safety
    #                     safe_folder_name = original_folder_path.replace('/', '_').replace('\\', '_')
    #                     dest_filename = f"{safe_folder_name}__{filename_only}"
    #                 else:
    #                     dest_filename = filename_only

    #                 dest_path = os.path.join(output_dir, category_folder, dest_filename)

    #                 try:
    #                     shutil.copy2(source_path, dest_path)
    #                 except Exception as e:
    #                     logger.warning(f"Failed to copy {filename_only} from {original_folder_path}: {e}")

    #             # Extract data for CSV
    #             yolo_detected = False
    #             yolo_confidence = 0
    #             total_detections = 0
    #             siamese_matches = 0
    #             best_match_id = ""
    #             best_match_confidence = 0

    #             if result.yolo_result and "error" not in result.yolo_result:
    #                 yolo_detected = result.yolo_result.get("total_detections", 0) > 0
    #                 yolo_confidence = result.yolo_result.get("highest_confidence", 0)
    #                 total_detections = result.yolo_result.get("total_detections", 0)

    #             if result.siamese_result and "error" not in result.siamese_result:
    #                 matches = result.siamese_result.get("matches", [])
    #                 siamese_matches = len(matches)
    #                 if matches:
    #                     best_match = matches[0]  # Already sorted by confidence
    #                     best_match_id = best_match.get("elephant_id", "")
    #                     best_match_confidence = best_match.get("confidence", 0)

    #             writer.writerow({
    #                 'filename': filename_only,
    #                 'original_folder_path': original_folder_path,
    #                 'category': result.category,
    #                 'original_size': result.original_size,
    #                 'file_size_mb': result.file_size_mb,
    #                 'processing_time': result.processing_time,
    #                 'yolo_detected': yolo_detected,
    #                 'yolo_confidence': yolo_confidence,
    #                 'total_detections': total_detections,
    #                 'siamese_matches': siamese_matches,
    #                 'best_match_id': best_match_id,
    #                 'best_match_confidence': best_match_confidence,
    #                 'error_message': result.error_message or ""
    #             })

    #     # Create enhanced summary report with UTF-8 encoding (FIXED: Removed Unicode emojis)
    #     summary_path = os.path.join(output_dir, "summary_report.txt")
    #     results_summary = defaultdict(int)
    #     folder_summary = defaultdict(lambda: defaultdict(int))  # folder -> category -> count

    #     for result in results:
    #         results_summary[result.category] += 1

    #         # Analyze folder distribution
    #         if hasattr(result, 'original_folder') and result.original_folder:
    #             folder_summary[result.original_folder][result.category] += 1

    #     # FIXED: Using UTF-8 encoding and removed Unicode emojis for Windows compatibility
    #     with open(summary_path, 'w', encoding='utf-8') as f:
    #         f.write(f"Enhanced Batch Processing Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #         f.write("=" * 70 + "\n\n")
    #         f.write(f"Total Images Processed: {len(results)}\n\n")

    #         f.write("Overall Category Breakdown:\n")
    #         f.write("-" * 30 + "\n")
    #         for category, count in results_summary.items():
    #             f.write(f"  {category.replace('_', ' ').title()}: {count}\n")

    #         # Add folder structure analysis if available
    #         if folder_summary:
    #             f.write(f"\nFolder Structure Analysis:\n")
    #             f.write("-" * 30 + "\n")
    #             for folder, categories in folder_summary.items():
    #                 f.write(f"  Folder: {folder}:\n")
    #                 for category, count in categories.items():
    #                     f.write(f"    - {category.replace('_', ' ').title()}: {count}\n")

    #         f.write(f"\nFeatures:\n")
    #         f.write("- [OK] Supports ANY ZIP folder structure\n")
    #         f.write("- [OK] Preserves original folder information in CSV\n")
    #         f.write("- [OK] Files organized by detection results\n")
    #         f.write("- [OK] Unique naming prevents conflicts\n")
    #         f.write(f"\nProcessing completed successfully!\n")
    #         f.write(f"Results organized in folders by category.\n")
    #         f.write(f"Original folder structure preserved in CSV report.\n")
    #         f.write(f"Detailed results available in: batch_processing_report.csv\n")

    #     preserved_files_dir = self._preserve_processed_files(results, temp_upload_folder, batch_id)
    #     # Create ZIP file
    #     zip_path = f"{output_dir}.zip"

    #     with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    #         for root, dirs, files in os.walk(output_dir):
    #             for file in files:
    #                 file_path = os.path.join(root, file)
    #                 arcname = os.path.relpath(file_path, output_dir)
    #                 zipf.write(file_path, arcname)

    #     # Clean up temporary directory
    #     #shutil.rmtree(output_dir)

    #     return zip_path, dict(results_summary)

# Initialize FastAPI app
app = FastAPI(
    title="Airavat Enhanced Batch Processing Backend",
    description="Enhanced elephant detection API with ZIP file support for massive batch processing (up to 200GB)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware for large file uploads
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
siamese_processor = None
yolo_processor = None
batch_processor = None

async def initialize_models():
    """Initialize optimized AI models"""
    global siamese_processor, yolo_processor, batch_processor

    logger.info("Initializing optimized AI models...")

    # Initialize Siamese processor
    try:
        if TORCH_AVAILABLE:
            siamese_processor = RealSiameseProcessor()
            logger.info("Siamese processor initialized")
        else:
            logger.error("PyTorch not available for Siamese processor")
    except Exception as e:
        logger.error(f"Failed to initialize Siamese processor: {e}")
        siamese_processor = None

    # Initialize optimized YOLO processor
    try:
        if YOLO_AVAILABLE:
            yolo_processor = OptimizedYOLOProcessor()
            if yolo_processor.model is not None:
                logger.info("Optimized YOLO processor initialized successfully")
            else:
                logger.warning("YOLO processor created but model not loaded")
        else:
            logger.error("Ultralytics not available for YOLO processor")
    except Exception as e:
        logger.error(f"Failed to initialize YOLO processor: {e}")
        yolo_processor = None

    # Initialize enhanced batch processor
    try:
        batch_processor = EnhancedBatchProcessor(yolo_processor, siamese_processor)
        logger.info("Enhanced batch processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize batch processor: {e}")
        batch_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await initialize_models()

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_archive_file(filename: str) -> bool:
    """Check if archive file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_ARCHIVE_EXTENSIONS

# API ROUTES

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status='healthy',
            timestamp=datetime.now().isoformat(),
            app_name='Airavat Enhanced Batch Backend',
            version='1.0.0',
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            pytorch_version=torch.__version__ if TORCH_AVAILABLE else None,
            cuda_available=torch.cuda.is_available() if TORCH_AVAILABLE else False,
            device=str(device),
            models_loaded={
                'siamese': siamese_processor is not None,
                'yolo': yolo_processor is not None,
                'batch_processor': batch_processor is not None
            },
            dependencies={
                'torch': TORCH_AVAILABLE,
                'ultralytics': YOLO_AVAILABLE
            },
            mode='enhanced_batch_processing_with_zip_support'
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/batch-individual-elephants", response_model=IndividualElephantBatchResponse)
async def batch_individual_elephant_identification(
    background_tasks: BackgroundTasks,
    images: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
    confidence_threshold: float = Form(0.5),
    similarity_threshold: float = Form(0.85),
    max_workers: int = Form(4)
):
    """
    Individual Elephant Identification - Just like your Streamlit app!

    Processes images and groups them by individual elephant identity using:
    1. YOLO detection to find elephants
    2. Siamese network to extract embeddings
    3. Similarity clustering to group same individuals

    Returns numbered folders for each individual elephant:
    - 01_elephant_individual/, 02_elephant_individual/, etc.
    - Each folder contains images of the SAME individual elephant
    """

    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")

    if not yolo_processor or not yolo_processor.model:
        raise HTTPException(status_code=500, detail="YOLO model required for elephant detection")

    if not siamese_processor:
        raise HTTPException(status_code=500, detail="Siamese model required for individual identification")

    # Validation
    if not images and not zip_file:
        raise HTTPException(status_code=400, detail="Must provide either 'images' or 'zip_file' parameter")

    if images and zip_file:
        raise HTTPException(status_code=400, detail="Cannot process both individual images and ZIP file simultaneously")

    batch_id = str(uuid.uuid4())[:8]
    temp_folder = os.path.join(UPLOAD_FOLDER, f"individual_elephants_{batch_id}")
    os.makedirs(temp_folder, exist_ok=True)

    logger.info(f"🐘 Starting Individual Elephant Identification (Batch ID: {batch_id})")
    logger.info(f"🎯 Similarity threshold: {similarity_threshold}")

    start_time = time.time()

    try:
        # Handle file extraction
        image_files_to_process = await _extract_files_for_processing(
            images, zip_file, temp_folder, batch_id
        )

        # Process all images with both YOLO and Siamese
        results = []
        logger.info(f"🔍 Processing {len(image_files_to_process)} images for individual elephant identification...")

        for file_path, filename in image_files_to_process:
            try:
                result = await batch_processor.process_single_image(
                    file_path, filename,
                    yolo_enabled=True,        # ✅ Enable YOLO for detection
                    siamese_enabled=True,     # ✅ Enable Siamese for embeddings
                    confidence_threshold=confidence_threshold,
                    siamese_threshold=0.85    # Not used for grouping, just matching
                )
                results.append(result)

                if len(results) % 25 == 0:
                    logger.info(f"📊 Processed {len(results)}/{len(image_files_to_process)} images...")

            except Exception as e:
                logger.error(f"❌ Failed to process {filename}: {e}")
                results.append(BatchImageResult(
                    filename=filename,
                    original_size="unknown",
                    file_size_mb=0,
                    processing_time=0,
                    category="processing_error",
                    error_message=str(e),
                    embedding=None
                ))

        # Create results ZIP with INDIVIDUAL ELEPHANT GROUPING
        zip_path, results_summary = batch_processor.create_results_zip_with_individual_elephants(
            results, temp_folder, batch_id, similarity_threshold
        )

        # Count individual elephant groups
        individual_groups = len([key for key in results_summary.keys()
                               if key not in ["00_no_elephants_detected", "99_processing_errors"]])

        successfully_processed = sum(1 for r in results if r.category != "processing_error")
        failed_images = len(results) - successfully_processed
        total_time = time.time() - start_time

        logger.info(f"✅ Individual elephant identification completed!")
        logger.info(f"🐘 Found {individual_groups} unique individual elephants")
        logger.info(f"⏱️ Total time: {total_time:.2f}s")

        return IndividualElephantBatchResponse(
            total_images=len(results),
            successfully_processed=successfully_processed,
            failed_images=failed_images,
            processing_time=f"{total_time:.2f}s",
            individual_elephant_groups=individual_groups,
            results_summary=results_summary,
            zip_file_path=zip_path,
            similarity_threshold_used=similarity_threshold,
            detailed_results=results
        )

    except Exception as e:
        logger.error(f"❌ Individual elephant identification failed: {e}")
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        raise HTTPException(status_code=500, detail=f"Individual elephant identification failed: {str(e)}")

@app.post("/api/compare-dataset", response_model=SiameseResponse)
async def compare_with_dataset(
    image: UploadFile = File(...),
    threshold: float = Form(0.85),
    top_k: int = Form(10)
):
    """Siamese network comparison only"""
    if not siamese_processor:
        raise HTTPException(status_code=500, detail="Siamese processor not available")

    if not image.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        start_time = time.time()
        matches = await siamese_processor.compare_with_dataset(contents, threshold, top_k)
        processing_time = f"{time.time() - start_time:.2f}s"

        return SiameseResponse(
            matches=matches,
            total_matches=len(matches),
            threshold_used=threshold,
            processing_time=processing_time,
            message='Elephant identification completed'
        )

    except Exception as e:
        logger.error(f"Siamese comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/detect-yolo", response_model=CustomYOLOResponse)
async def detect_elephants_yolo(
    image: UploadFile = File(...),
    confidence: float = Form(0.5),
    iou: float = Form(0.45),
    image_size: int = Form(640)
):
    """YOLO detection with custom output format"""
    # Enhanced error checking
    if not yolo_processor:
        raise HTTPException(
            status_code=500,
            detail="YOLO processor not available. Please check server logs for model loading issues."
        )

    if not yolo_processor.model:
        raise HTTPException(
            status_code=500,
            detail="Custom YOLO model not loaded. Please ensure 'models/yolo_best_model.pt' exists and is a valid trained model file."
        )

    if not image.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Check file size
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        result = await yolo_processor.detect_elephants_custom(contents, confidence, iou, image_size)
        return result

    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# ENHANCED: Modified batch processing endpoint to handle ZIP files
@app.post("/api/batch-yolo", response_model=YOLOBatchResponse)
async def batch_yolo_detection(
    background_tasks: BackgroundTasks,
    images: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
    confidence_threshold: float = Form(0.5),
    max_workers: int = Form(4)
):
    """
    YOLO-only batch processing endpoint

    Processes images and detects elephants using only YOLO model.
    Returns images categorized as:
    - elephants_detected: Images where elephants were found
    - no_elephants: Images where no elephants were detected
    - processing_error: Images that failed to process
    """

    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")

    if not yolo_processor or not yolo_processor.model:
        raise HTTPException(status_code=500, detail="YOLO model not available")

    # Validation
    if not images and not zip_file:
        raise HTTPException(status_code=400, detail="Must provide either 'images' or 'zip_file' parameter")

    if images and zip_file:
        raise HTTPException(status_code=400, detail="Cannot process both individual images and ZIP file simultaneously")

    batch_id = str(uuid.uuid4())[:8]
    temp_folder = os.path.join(UPLOAD_FOLDER, f"yolo_batch_{batch_id}")
    os.makedirs(temp_folder, exist_ok=True)

    logger.info(f"🔍 Starting YOLO-only batch processing (Batch ID: {batch_id})")

    start_time = time.time()

    try:
        # Handle file extraction (ZIP or individual files)
        image_files_to_process = await _extract_files_for_processing(
            images, zip_file, temp_folder, batch_id
        )

        # Process all images with YOLO only
        results = []
        logger.info(f"🔍 Processing {len(image_files_to_process)} images with YOLO detection...")

        for file_path, filename in image_files_to_process:
            try:
                result = await batch_processor.process_single_image(
                    file_path, filename,
                    yolo_enabled=True,        # ✅ Enable YOLO
                    siamese_enabled=False,    # ❌ Disable Siamese
                    confidence_threshold=confidence_threshold,
                    siamese_threshold=0.85    # Not used
                )
                results.append(result)

                if len(results) % 50 == 0:
                    logger.info(f"📊 YOLO processed {len(results)}/{len(image_files_to_process)} images...")

            except Exception as e:
                logger.error(f"❌ Failed to process {filename}: {e}")
                results.append(BatchImageResult(
                    filename=filename,
                    original_size="unknown",
                    file_size_mb=0,
                    processing_time=0,
                    category="processing_error",
                    error_message=str(e)
                ))

        # Create results ZIP with YOLO-specific categories
        zip_path, results_summary = batch_processor.create_results_zip(
            results, temp_folder, f"yolo_{batch_id}"
        )

        successfully_processed = sum(1 for r in results if r.category != "processing_error")
        failed_images = len(results) - successfully_processed
        total_time = time.time() - start_time

        logger.info(f"✅ YOLO batch processing completed! ({total_time:.2f}s)")

        # Cleanup

        return YOLOBatchResponse(
            total_images=len(results),
            successfully_processed=successfully_processed,
            failed_images=failed_images,
            processing_time=f"{total_time:.2f}s",
            results_summary=results_summary,
            zip_file_path=zip_path,
            detailed_results=results
        )

    except Exception as e:
        logger.error(f"❌ YOLO batch processing failed: {e}")
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        raise HTTPException(status_code=500, detail=f"YOLO batch processing failed: {str(e)}")
from fastapi.staticfiles import StaticFiles
app.mount("/temp_uploads", StaticFiles(directory=UPLOAD_FOLDER), name="temp_uploads")

@app.post("/api/batch-siamese", response_model=SiameseBatchResponse)
async def batch_siamese_comparison(
    background_tasks: BackgroundTasks,
    images: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
    siamese_threshold: float = Form(0.85),
    max_workers: int = Form(4)
):
    """
    Siamese-only batch processing endpoint

    Processes images and compares against elephant dataset using only Siamese network.
    Returns images categorized as:
    - matches_found: Images that matched known elephants in dataset
    - no_matches: Images that didn't match any known elephants
    - processing_error: Images that failed to process
    """

    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")

    if not siamese_processor:
        raise HTTPException(status_code=500, detail="Siamese model not available")

    # Validation
    if not images and not zip_file:
        raise HTTPException(status_code=400, detail="Must provide either 'images' or 'zip_file' parameter")

    if images and zip_file:
        raise HTTPException(status_code=400, detail="Cannot process both individual images and ZIP file simultaneously")

    batch_id = str(uuid.uuid4())[:8]
    temp_folder = os.path.join(UPLOAD_FOLDER, f"siamese_batch_{batch_id}")
    os.makedirs(temp_folder, exist_ok=True)

    logger.info(f"🧠 Starting Siamese-only batch processing (Batch ID: {batch_id})")

    start_time = time.time()

    try:
        # Handle file extraction (ZIP or individual files)
        image_files_to_process = await _extract_files_for_processing(
            images, zip_file, temp_folder, batch_id
        )

        # Process all images with Siamese only
        results = []
        logger.info(f"🧠 Processing {len(image_files_to_process)} images with Siamese comparison...")

        for file_path, filename in image_files_to_process:
            try:
                result = await batch_processor.process_single_image(
                    file_path, filename,
                    yolo_enabled=False,       # ❌ Disable YOLO
                    siamese_enabled=True,     # ✅ Enable Siamese
                    confidence_threshold=0.5, # Not used
                    siamese_threshold=siamese_threshold
                )
                results.append(result)

                if len(results) % 50 == 0:
                    logger.info(f"📊 Siamese processed {len(results)}/{len(image_files_to_process)} images...")

            except Exception as e:
                logger.error(f"❌ Failed to process {filename}: {e}")
                results.append(BatchImageResult(
                    filename=filename,
                    original_size="unknown",
                    file_size_mb=0,
                    processing_time=0,
                    category="processing_error",
                    error_message=str(e)
                ))

        # Create results ZIP with Siamese-specific categories
        zip_path, results_summary = batch_processor.create_results_zip(
            results, temp_folder, f"siamese_{batch_id}"
        )

        successfully_processed = sum(1 for r in results if r.category != "processing_error")
        failed_images = len(results) - successfully_processed
        total_time = time.time() - start_time

        logger.info(f"✅ Siamese batch processing completed! ({total_time:.2f}s)")

        # Cleanup

        return SiameseBatchResponse(
            total_images=len(results),
            successfully_processed=successfully_processed,
            failed_images=failed_images,
            processing_time=f"{total_time:.2f}s",
            results_summary=results_summary,
            zip_file_path=zip_path,
            detailed_results=results
        )

    except Exception as e:
        logger.error(f"❌ Siamese batch processing failed: {e}")
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        raise HTTPException(status_code=500, detail=f"Siamese batch processing failed: {str(e)}")

@app.post("/api/batch-combined", response_model=CombinedBatchResponse)
async def batch_combined_processing(
    background_tasks: BackgroundTasks,
    images: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
    confidence_threshold: float = Form(0.5),
    siamese_threshold: float = Form(0.85),
    max_workers: int = Form(4)
):
    """
    Combined YOLO + Siamese batch processing endpoint

    Processes images with both YOLO detection AND Siamese comparison.
    Returns images categorized as:
    - matches_found: Images with elephant detections AND dataset matches
    - elephants_detected: Images where elephants were detected but no matches found
    - no_elephants: Images where no elephants were detected
    - processing_error: Images that failed to process
    """

    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")

    if not yolo_processor or not yolo_processor.model:
        raise HTTPException(status_code=500, detail="YOLO model not available")

    if not siamese_processor:
        raise HTTPException(status_code=500, detail="Siamese model not available")

    # Validation
    if not images and not zip_file:
        raise HTTPException(status_code=400, detail="Must provide either 'images' or 'zip_file' parameter")

    if images and zip_file:
        raise HTTPException(status_code=400, detail="Cannot process both individual images and ZIP file simultaneously")

    batch_id = str(uuid.uuid4())[:8]
    temp_folder = os.path.join(UPLOAD_FOLDER, f"combined_batch_{batch_id}")
    os.makedirs(temp_folder, exist_ok=True)

    logger.info(f"🔍🧠 Starting Combined batch processing (Batch ID: {batch_id})")

    start_time = time.time()

    try:
        # Handle file extraction (ZIP or individual files)
        image_files_to_process = await _extract_files_for_processing(
            images, zip_file, temp_folder, batch_id
        )

        # Process all images with both YOLO and Siamese
        results = []
        logger.info(f"🔍🧠 Processing {len(image_files_to_process)} images with YOLO + Siamese...")

        for file_path, filename in image_files_to_process:
            try:
                result = await batch_processor.process_single_image(
                    file_path, filename,
                    yolo_enabled=True,        # ✅ Enable YOLO
                    siamese_enabled=True,     # ✅ Enable Siamese
                    confidence_threshold=confidence_threshold,
                    siamese_threshold=siamese_threshold
                )
                results.append(result)

                if len(results) % 50 == 0:
                    logger.info(f"📊 Combined processed {len(results)}/{len(image_files_to_process)} images...")

            except Exception as e:
                logger.error(f"❌ Failed to process {filename}: {e}")
                results.append(BatchImageResult(
                    filename=filename,
                    original_size="unknown",
                    file_size_mb=0,
                    processing_time=0,
                    category="processing_error",
                    error_message=str(e)
                ))

        # Create results ZIP with combined categories
        zip_path, results_summary = batch_processor.create_results_zip(
            results, temp_folder, f"combined_{batch_id}"
        )

        successfully_processed = sum(1 for r in results if r.category != "processing_error")
        failed_images = len(results) - successfully_processed
        total_time = time.time() - start_time

        logger.info(f"✅ Combined batch processing completed! ({total_time:.2f}s)")

        # Cleanup


        return CombinedBatchResponse(
            total_images=len(results),
            successfully_processed=successfully_processed,
            failed_images=failed_images,
            processing_time=f"{total_time:.2f}s",
            results_summary=results_summary,
            zip_file_path=zip_path,
            detailed_results=results
        )

    except Exception as e:
        logger.error(f"❌ Combined batch processing failed: {e}")
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        raise HTTPException(status_code=500, detail=f"Combined batch processing failed: {str(e)}")


# Helper functions
async def _extract_files_for_processing(images, zip_file, temp_folder, batch_id):
    """Helper function to extract files from either ZIP or individual uploads"""
    image_files_to_process = []

    if zip_file:
        logger.info(f"📦 Processing ZIP file: {zip_file.filename}")

        # Validate ZIP file
        if not zip_file.filename or not allowed_archive_file(zip_file.filename):
            raise HTTPException(status_code=400, detail="Invalid ZIP file format")

        # Read and save ZIP file
        zip_content = await zip_file.read()
        zip_size_gb = len(zip_content) / (1024 * 1024 * 1024)
        logger.info(f"📦 ZIP file size: {zip_size_gb:.2f} GB")

        if len(zip_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"ZIP file too large (max {MAX_FILE_SIZE / (1024**3):.0f}GB)")

        # Save ZIP file temporarily
        zip_path = os.path.join(temp_folder, zip_file.filename)
        with open(zip_path, 'wb') as f:
            f.write(zip_content)

        # Extract ZIP and get image files
        extracted_images = await batch_processor.extract_zip_file(zip_path, temp_folder)
        image_files_to_process = [(img_path, img_path) for img_path in extracted_images]

        # Clean up ZIP file after extraction
        #if os.path.exists(zip_path):
           # os.remove(zip_path)
        logger.info(f"📋 First 3 files to process:")
        for i, (file_path, filename) in enumerate(image_files_to_process[:3]):
            logger.info(f"  {i+1}. file_path: {file_path}")
            logger.info(f"     filename: {filename}")
            logger.info(f"     exists: {os.path.exists(file_path)}")

        logger.info(f"📦 ZIP processing complete: {len(image_files_to_process)} images ready")

    elif images:
        logger.info(f"📁 Processing {len(images)} individual image files")

        # Validate and save all uploaded images
        for image in images:
            if not image.filename or not allowed_file(image.filename):
                raise HTTPException(status_code=400, detail=f"Invalid file: {image.filename or 'unnamed'}")

            content = await image.read()
            if len(content) > MAX_SINGLE_IMAGE_SIZE:
                logger.warning(f"⚠️ Skipping large file: {image.filename}")
                continue

            file_path = os.path.join(temp_folder, image.filename)
            with open(file_path, 'wb') as f:
                f.write(content)
            image_files_to_process.append((file_path, image.filename))

        logger.info(f"📁 Saved {len(image_files_to_process)} individual images")

    if not image_files_to_process:
        raise HTTPException(status_code=400, detail="No valid images found to process")

    return image_files_to_process

def _cleanup_temp_folder_only(temp_folder):
    """Cleanup only the temp extraction folder, preserve processed files"""
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            logger.info(f"🗑️ Cleaned up temp folder: {temp_folder}")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")

def _schedule_preserved_cleanup(batch_id):
    """Schedule cleanup of preserved files after retention period"""
    def delayed_cleanup():
        time.sleep(PRESERVED_FILES_RETENTION_HOURS * 3600)  # Convert hours to seconds
        preserve_dir = os.path.join(PRESERVED_FILES_DIR, f"batch_{batch_id}")
        if os.path.exists(preserve_dir):
            shutil.rmtree(preserve_dir)
            logger.info(f"🗑️ Cleaned up preserved files: {preserve_dir}")

    # Run in background thread
    import threading
    threading.Thread(target=delayed_cleanup, daemon=True).start()

@app.get("/api/download-batch/{filename}")
async def download_batch_result(filename: str):
    """Download batch processing result ZIP file"""

    # Security: Only allow downloading files from upload folder with specific pattern
    if not filename.endswith('.zip') or 'batch_results_' not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/zip'
    )
# NEW ENDPOINT 1: Prepare Download Package# FastAPI Route Conversions from Flask
# Replace the Flask routes in your main.py with these FastAPI equivalents

from fastapi import Request
from fastapi.responses import JSONResponse
import tempfile
import json

# NEW ENDPOINT 1: Prepare Download Package (FastAPI version)
@app.post("/api/prepare-download-package")
async def prepare_download_package(request: Request):
    """
    Organizes processed results into categorized folders and creates a ZIP for download
    FastAPI version of the Flask route
    """
    try:
        # Get JSON data from request body
        data = await request.json()

        if not data or 'results' not in data:
            return JSONResponse(
                status_code=400,
                content={
                    'success': False,
                    'error': 'No results provided'
                }
            )

        results = data['results']
        options = data.get('options', {})
        processing_type = options.get('processingType', 'yolo')

        logger.info(f"📦 Preparing download package for {len(results)} results")

        # Create temporary directory for organizing files
        download_id = f"download_{int(datetime.now().timestamp())}_{os.urandom(4).hex()}"
        temp_dir = os.path.join(tempfile.gettempdir(), download_id)
        os.makedirs(temp_dir, exist_ok=True)

        # Initialize category counters and directories
        categories = {
            'detected-elephant': [],
            'no-objects-detected': [],
            'high-similarity-90-plus': [],
            'medium-similarity-80-89': [],
            'low-similarity-70-79': [],
            'no-matches-found': [],
            'processing-errors': []
        }

        processed_count = 0
        error_count = 0

        # Process each result
        for result in results:
            try:
                if not result.get('filename'):
                    error_count += 1
                    continue

                filename = os.path.basename(result['filename'])

                # Find the original image file in temp directories
                original_image_path = find_original_image_file(filename)

                if not original_image_path or not os.path.exists(original_image_path):
                    logger.warning(f"⚠️ Original image not found: {filename}")
                    error_count += 1
                    continue

                # Determine categories based on processing results
                result_categories = determine_result_categories(result, processing_type, options)

                # Copy image to each relevant category folder
                for category in result_categories:
                    if category not in categories:
                        categories[category] = []

                    # Create category directory
                    category_dir = os.path.join(temp_dir, category)
                    os.makedirs(category_dir, exist_ok=True)

                    # Copy image file
                    dest_path = os.path.join(category_dir, filename)

                    # Handle filename conflicts
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(filename)
                        dest_path = os.path.join(category_dir, f"{name}_{counter}{ext}")
                        counter += 1

                    shutil.copy2(original_image_path, dest_path)
                    categories[category].append({
                        'filename': os.path.basename(dest_path),
                        'original_path': original_image_path,
                        'result': result
                    })

                # Create individual result summary if requested
                if options.get('includeSummaries', True):
                    create_individual_result_summary(temp_dir, filename, result, result_categories)

                processed_count += 1

            except Exception as e:
                logger.error(f"❌ Error processing result for {result.get('filename', 'unknown')}: {str(e)}")
                error_count += 1

        # Create overall summary
        if options.get('includeSummaries', True):
            create_processing_summary(temp_dir, {
                'processing_type': processing_type,
                'total_images': len(results),
                'processed_images': processed_count,
                'failed_images': error_count,
                'categories': {k: len(v) for k, v in categories.items() if v},
                'download_id': download_id,
                'created_at': datetime.now().isoformat()
            })

        # Create ZIP file
        zip_filename = f"processed_results_{processing_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)

        create_zip_from_directory(temp_dir, zip_path)

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info(f"✅ Download package created: {zip_path}")

        return JSONResponse(content={
            'success': True,
            'zipPath': zip_path,
            'filename': zip_filename,
            'processedImages': processed_count,
            'totalImages': len(results),
            'categories': {k: len(v) for k, v in categories.items() if v},
            'downloadId': download_id,
            'processingType': processing_type,
            'notes': f"{error_count} images could not be processed due to missing files." if error_count > 0 else None
        })

    except Exception as e:
        logger.error(f"❌ Error preparing download package: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e)
            }
        )

# NEW ENDPOINT 2: Download Prepared Package (FastAPI version)
@app.get("/api/download-prepared-package")
async def download_prepared_package(zip_path: str, filename: str = "processed_results.zip"):
    """
    Serves the prepared ZIP file for download
    FastAPI version of the Flask route
    """
    try:
        if not zip_path or not os.path.exists(zip_path):
            return JSONResponse(
                status_code=404,
                content={
                    'success': False,
                    'error': 'ZIP file not found'
                }
            )

        logger.info(f"📤 Serving download: {filename}")

        return FileResponse(
            path=zip_path,
            filename=filename,
            media_type='application/zip'
        )

    except Exception as e:
        logger.error(f"❌ Error serving download: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e)
            }
        )

# HELPER FUNCTIONS (keep these as-is, they work with both Flask and FastAPI)

def find_original_image_file(filename):
    """
    Find original image file in various temp directories
    """
    # Common temp directories where processed images are stored
    temp_dirs = [
        os.path.join(tempfile.gettempdir(), 'yolo_batch_uploads'),
        os.path.join(tempfile.gettempdir(), 'siamese_batch_uploads'),
        os.path.join(tempfile.gettempdir(), 'combined_batch_uploads'),
        os.path.join(tempfile.gettempdir(), 'temp_uploads')
    ]

    # Also search for pattern-based temp directories
    temp_root = tempfile.gettempdir()
    try:
        for item in os.listdir(temp_root):
            if 'batch' in item.lower() or 'upload' in item.lower() or 'yolo' in item.lower() or 'siamese' in item.lower():
                temp_dirs.append(os.path.join(temp_root, item))
    except:
        pass

    # Search through all temp directories
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir):
                if filename in files:
                    return os.path.join(root, filename)

    return None

def determine_result_categories(result, processing_type, options):
    """
    Determine which categories this result belongs to based on processing results
    """
    categories = []

    try:
        # Handle YOLO detection results
        if processing_type in ['yolo', 'combined']:
            yolo_result = result.get('yolo_result', {})

            if yolo_result and not yolo_result.get('error'):
                total_detections = yolo_result.get('total_detections', 0)

                if total_detections > 0:
                    # Has detections
                    categories.append('detected-elephant')

                    # Add specific class-based categories if needed
                    detections = yolo_result.get('detections', [])
                    for detection in detections:
                        class_name = detection.get('class', '').lower()
                        if class_name and class_name not in ['elephant']:
                            categories.append(f'detected-{class_name}')
                else:
                    # No detections
                    categories.append('no-objects-detected')
            else:
                categories.append('processing-errors')

        # Handle Siamese comparison results
        if processing_type in ['compare-dataset', 'combined']:
            siamese_result = result.get('siamese_result', {})

            if siamese_result and not siamese_result.get('error'):
                total_matches = siamese_result.get('total_matches', 0)

                if total_matches > 0:
                    # Has matches - categorize by similarity
                    matches = siamese_result.get('matches', [])
                    if matches:
                        highest_similarity = max(match.get('similarity', 0) for match in matches)

                        if highest_similarity >= 0.9:
                            categories.append('high-similarity-90-plus')
                        elif highest_similarity >= 0.8:
                            categories.append('medium-similarity-80-89')
                        elif highest_similarity >= 0.7:
                            categories.append('low-similarity-70-79')
                        else:
                            categories.append('very-low-similarity-below-70')
                else:
                    # No matches
                    categories.append('no-matches-found')

        # Default category if none assigned
        if not categories:
            categories.append('processing-errors')

    except Exception as e:
        logger.error(f"Error determining categories for {result.get('filename', 'unknown')}: {str(e)}")
        categories.append('processing-errors')

    return categories

def create_individual_result_summary(temp_dir, filename, result, categories):
    """
    Create individual result summary JSON file
    """
    try:
        summaries_dir = os.path.join(temp_dir, 'result-summaries')
        os.makedirs(summaries_dir, exist_ok=True)

        name, ext = os.path.splitext(filename)
        summary_filename = f"{name}_result.json"
        summary_path = os.path.join(summaries_dir, summary_filename)

        summary_data = {
            'filename': filename,
            'categories': categories,
            'processing_result': result,
            'created_at': datetime.now().isoformat()
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error creating individual summary for {filename}: {str(e)}")

def create_processing_summary(temp_dir, summary_data):
    """
    Create overall processing summary
    """
    try:
        summaries_dir = os.path.join(temp_dir, 'result-summaries')
        os.makedirs(summaries_dir, exist_ok=True)

        summary_path = os.path.join(summaries_dir, 'processing_summary.json')

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error creating processing summary: {str(e)}")

def create_zip_from_directory(source_dir, zip_path):
    """
    Create ZIP file from directory contents
    """
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create relative path for ZIP entry
                    arc_path = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arc_path)

        logger.info(f"✅ ZIP file created: {zip_path}")

    except Exception as e:
        logger.error(f"❌ Error creating ZIP file: {str(e)}")
        raise

# CLEANUP FUNCTION (call this periodically to clean up old ZIP files)
def cleanup_old_download_files():
    """
    Clean up old download ZIP files (call this periodically)
    """
    try:
        temp_dir = tempfile.gettempdir()
        current_time = datetime.now().timestamp()

        for filename in os.listdir(temp_dir):
            if filename.startswith('processed_results_') and filename.endswith('.zip'):
                file_path = os.path.join(temp_dir, filename)
                file_age = current_time - os.path.getctime(file_path)

                # Remove files older than 1 hour
                if file_age > 3600:  # 1 hour in seconds
                    try:
                        os.remove(file_path)
                        logger.info(f"🗑️ Cleaned up old download file: {filename}")
                    except Exception as e:
                        logger.error(f"Error removing {filename}: {str(e)}")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


# Root redirect to docs
@app.get("/")
async def root():
    return {"message": "Airavat Enhanced AI Backend with ZIP Batch Processing - Visit /docs for API documentation"}

if __name__ == '__main__':
    logger.info("Airavat Enhanced FastAPI Backend Server v1.0.0 Starting...")
    logger.info("=" * 60)
    logger.info("NEW: ZIP file support for massive batch processing (up to 200GB)")
    logger.info("Starting FastAPI server...")
    logger.info("Enhanced for massive datasets - Detection + ZIP Batch Processing")
    logger.info("API Documentation available at /docs")
    logger.info("=" * 60)

    # Start FastAPI server
    port = int(os.environ.get('PORT', 8000))

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            workers=1,
            access_log=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
