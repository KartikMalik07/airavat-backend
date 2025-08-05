#!/usr/bin/env python3
"""
YOLOv8 Processor for Elephant Detection
Handles real-time elephant detection and localization
"""

import os
import base64
import logging
from typing import List, Dict, Tuple, Optional
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("⚠️ Ultralytics not installed. YOLOv8 functionality will be limited.")

logger = logging.getLogger(__name__)

class YOLOProcessor:
    """Handles YOLOv8 inference for elephant detection"""

    def __init__(self, model_path: str):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['elephant']  # Assuming single class model

        if YOLO_AVAILABLE:
            self.load_model(model_path)
        else:
            logger.error("❌ YOLOv8 not available. Please install ultralytics package.")

    def load_model(self, model_path: str):
        """Load YOLOv8 model"""
        try:
            logger.info(f"Loading YOLOv8 model from {model_path}")

            if os.path.exists(model_path):
                self.model = YOLO(model_path)

                # Move to appropriate device
                if torch.cuda.is_available():
                    self.model.to('cuda')

                logger.info("✅ YOLOv8 model loaded successfully")
            else:
                logger.warning(f"⚠️ Model file not found: {model_path}")
                # Load default YOLOv8 model for demonstration
                self.model = YOLO('yolov8n.pt')  # Nano model for demo
                logger.info("✅ Using default YOLOv8n model for demonstration")

        except Exception as e:
            logger.error(f"❌ Error loading YOLOv8 model: {e}")
            raise

    def detect_elephants(self, image_path: str, confidence_threshold: float = 0.5,
                        iou_threshold: float = 0.45, image_size: int = 640) -> Dict:
        """Detect elephants in image"""
        if not YOLO_AVAILABLE or self.model is None:
            return {'error': 'YOLOv8 model not available'}

        try:
            logger.info(f"Detecting elephants in {image_path}")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            original_height, original_width = image.shape[:2]

            # Run inference
            results = self.model(
                image_path,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=image_size,
                verbose=False
            )

            # Process results
            detections = []
            annotated_image = image.copy()

            if len(results) > 0:
                result = results[0]  # First result

                # Extract detections
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None

                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = box.astype(int)

                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)

                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'area': int(area),
                            'class': 'elephant',
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        }
                        detections.append(detection)

                        # Draw bounding box on image
                        color = (0, 255, 0)  # Green
                        thickness = 2
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

                        # Draw label
                        label = f"Elephant {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_image, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Convert annotated image to base64
            annotated_image_base64 = self._image_to_base64(annotated_image)

            return {
                'detections': detections,
                'total_detections': len(detections),
                'annotated_image_base64': annotated_image_base64,
                'image_dimensions': {
                    'width': original_width,
                    'height': original_height
                }
            }

        except Exception as e:
            logger.error(f"Error detecting elephants: {e}")
            return {'error': str(e), 'detections': []}

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)

            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=90)
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return img_str

        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""

    def batch_detect(self, image_paths: List[str], confidence_threshold: float = 0.5,
                    iou_threshold: float = 0.45, image_size: int = 640) -> Dict:
        """Batch detection on multiple images"""
        if not YOLO_AVAILABLE or self.model is None:
            return {'error': 'YOLOv8 model not available'}

        try:
            logger.info(f"Running batch detection on {len(image_paths)} images")

            all_results = []
            total_detections = 0
            processed_count = 0

            for i, image_path in enumerate(image_paths):
                try:
                    result = self.detect_elephants(
                        image_path,
                        confidence_threshold,
                        iou_threshold,
                        image_size
                    )

                    if 'error' not in result:
                        result['image_path'] = image_path
                        result['image_index'] = i
                        all_results.append(result)
                        total_detections += result['total_detections']
                        processed_count += 1

                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(image_paths)} images")

                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue

            return {
                'results': all_results,
                'summary': {
                    'total_images': len(image_paths),
                    'processed_images': processed_count,
                    'total_detections': total_detections,
                    'average_detections_per_image': total_detections / max(processed_count, 1),
                    'images_with_detections': sum(1 for r in all_results if r['total_detections'] > 0)
                }
            }

        except Exception as e:
            logger.error(f"Error in batch detection: {e}")
            return {'error': str(e)}

    def analyze_detection_patterns(self, results: List[Dict]) -> Dict:
        """Analyze detection patterns from batch results"""
        try:
            if not results:
                return {'error': 'No results to analyze'}

            # Collect statistics
            confidences = []
            areas = []
            detections_per_image = []

            for result in results:
                detections_per_image.append(result['total_detections'])

                for detection in result['detections']:
                    confidences.append(detection['confidence'])
                    areas.append(detection['area'])

            analysis = {
                'detection_statistics': {
                    'total_images_analyzed': len(results),
                    'total_detections': sum(detections_per_image),
                    'images_with_elephants': sum(1 for count in detections_per_image if count > 0),
                    'average_elephants_per_image': np.mean(detections_per_image) if detections_per_image else 0,
                    'max_elephants_in_single_image': max(detections_per_image) if detections_per_image else 0
                },
                'confidence_statistics': {
                    'average_confidence': np.mean(confidences) if confidences else 0,
                    'min_confidence': np.min(confidences) if confidences else 0,
                    'max_confidence': np.max(confidences) if confidences else 0,
                    'confidence_std': np.std(confidences) if confidences else 0
                },
                'size_statistics': {
                    'average_detection_area': np.mean(areas) if areas else 0,
                    'min_detection_area': np.min(areas) if areas else 0,
                    'max_detection_area': np.max(areas) if areas else 0,
                    'area_std': np.std(areas) if areas else 0
                }
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing detection patterns: {e}")
            return {'error': str(e)}

    def filter_detections_by_size(self, detections: List[Dict], min_area: int = 1000,
                                 max_area: int = None) -> List[Dict]:
        """Filter detections by bounding box area"""
        filtered = []

        for detection in detections:
            area = detection.get('area', 0)

            if area >= min_area:
                if max_area is None or area <= max_area:
                    filtered.append(detection)

        return filtered

    def non_max_suppression_custom(self, detections: List[Dict],
                                  iou_threshold: float = 0.5) -> List[Dict]:
        """Apply custom non-maximum suppression to detections"""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []

        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)

            # Remove overlapping detections
            remaining = []
            for detection in detections:
                iou = self._calculate_iou(current['bbox'], detection['bbox'])
                if iou < iou_threshold:
                    remaining.append(detection)

            detections = remaining

        return keep

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        try:
            x1, y1, x2, y2 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

            # Calculate intersection area
            xi1 = max(x1, x1_2)
            yi1 = max(y1, y1_2)
            xi2 = min(x2, x2_2)
            yi2 = min(y2, y2_2)

            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0

            intersection = (xi2 - xi1) * (yi2 - yi1)

            # Calculate union area
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating IoU: {e}")
            return 0.0

    def extract_elephant_crops(self, image_path: str, detections: List[Dict],
                              padding: int = 20) -> List[np.ndarray]:
        """Extract cropped elephant regions from image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []

            crops = []
            height, width = image.shape[:2]

            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']

                # Add padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)

                # Extract crop
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)

            return crops

        except Exception as e:
            logger.error(f"Error extracting crops: {e}")
            return []

    def get_model_info(self) -> Dict:
        """Get YOLOv8 model information"""
        info = {
            'model_type': 'YOLOv8',
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'ultralytics_available': YOLO_AVAILABLE
        }

        if self.model is not None:
            try:
                info.update({
                    'model_size': getattr(self.model.model, 'yaml', {}).get('nc', 'Unknown'),
                    'input_size': 640,  # Default YOLOv8 input size
                    'class_names': self.class_names
                })
            except Exception as e:
                logger.warning(f"Could not get detailed model info: {e}")

        return info

class MockYOLOProcessor:
    """Mock YOLOv8 processor for when ultralytics is not available"""

    def __init__(self, model_path: str):
        logger.warning("⚠️ Using mock YOLOv8 processor - ultralytics not available")
        self.model_path = model_path

    def detect_elephants(self, image_path: str, **kwargs) -> Dict:
        """Mock detection - returns sample data"""
        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image', 'detections': []}

            height, width = image.shape[:2]

            # Return mock detection
            mock_detection = {
                'detections': [{
                    'bbox': [width//4, height//4, 3*width//4, 3*height//4],
                    'confidence': 0.85,
                    'area': (width//2) * (height//2),
                    'class': 'elephant',
                    'center': [width//2, height//2]
                }],
                'total_detections': 1,
                'annotated_image_base64': '',
                'image_dimensions': {'width': width, 'height': height}
            }

            return mock_detection

        except Exception as e:
            return {'error': str(e), 'detections': []}

    def batch_detect(self, image_paths: List[str], **kwargs) -> Dict:
        """Mock batch detection"""
        return {
            'results': [],
            'summary': {
                'total_images': len(image_paths),
                'processed_images': 0,
                'total_detections': 0,
                'average_detections_per_image': 0,
                'images_with_detections': 0
            },
            'error': 'Mock processor - ultralytics not available'
        }

    def get_model_info(self) -> Dict:
        """Get mock model info"""
        return {
            'model_type': 'YOLOv8 (Mock)',
            'device': 'CPU',
            'model_loaded': False,
            'ultralytics_available': False,
            'error': 'ultralytics package not installed'
        }

# Use mock processor if ultralytics is not available
if not YOLO_AVAILABLE:
    YOLOProcessor = MockYOLOProcessor
