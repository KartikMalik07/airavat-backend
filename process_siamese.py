#!/usr/bin/env python3
"""
Siamese Network Processor for Elephant Identification
Handles individual elephant identification using ear pattern recognition
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)

class SiameseNetwork(nn.Module):
    """Siamese Network with EfficientNet backbone"""

    def __init__(self, embedding_dim=128, pretrained=True):
        super(SiameseNetwork, self).__init__()

        # EfficientNet backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0' if pretrained else 'efficientnet-b0',
                                                    advprop=False)

        # Replace classifier with embedding layer
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.embedding_dim = embedding_dim

    def forward_one(self, x):
        """Forward pass for one image"""
        return self.backbone(x)

    def forward(self, input1, input2=None):
        """Forward pass for pair of images or single image"""
        if input2 is not None:
            # Siamese forward pass
            output1 = self.forward_one(input1)
            output2 = self.forward_one(input2)
            return output1, output2
        else:
            # Single image forward pass
            return self.forward_one(input1)

class SiameseProcessor:
    """Handles Siamese network inference and dataset comparison"""

    def __init__(self, model_path: str, dataset_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset_embeddings = {}
        self.dataset_metadata = {}
        self.dataset_size = 0

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.load_model(model_path)
        self.load_dataset(dataset_path)

    def load_model(self, model_path: str):
        """Load the trained Siamese network"""
        try:
            logger.info(f"Loading Siamese model from {model_path}")

            # Initialize model architecture
            self.model = SiameseNetwork(embedding_dim=128, pretrained=False)

            # Load trained weights
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()
                logger.info("✅ Siamese model loaded successfully")
            else:
                logger.error(f"❌ Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")

        except Exception as e:
            logger.error(f"❌ Error loading Siamese model: {e}")
            raise

    def load_dataset(self, dataset_path: str = None):
        """Load pre-computed dataset embeddings"""
        if dataset_path is None:
            dataset_path = "dataset/embeddings.pkl"

        try:
            if os.path.exists(dataset_path):
                logger.info(f"Loading dataset embeddings from {dataset_path}")

                with open(dataset_path, 'rb') as f:
                    data = pickle.load(f)

                self.dataset_embeddings = data.get('embeddings', {})
                self.dataset_metadata = data.get('metadata', {})
                self.dataset_size = len(self.dataset_embeddings)

                logger.info(f"✅ Loaded {self.dataset_size} elephant embeddings")
            else:
                logger.warning(f"⚠️ Dataset embeddings not found at {dataset_path}")
                self._create_sample_dataset()

        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            self._create_sample_dataset()

    def _create_sample_dataset(self):
        """Create sample dataset for demonstration"""
        logger.info("Creating sample dataset for demonstration")

        # Create sample embeddings for demonstration
        for i in range(266):
            elephant_id = f"ELEPHANT_{i+1:03d}"
            # Random embedding for demonstration
            embedding = torch.randn(128).numpy()

            self.dataset_embeddings[elephant_id] = embedding
            self.dataset_metadata[elephant_id] = {
                'id': elephant_id,
                'description': f'Asian elephant individual {i+1}',
                'age_class': 'Adult' if i % 3 == 0 else 'Juvenile',
                'sex': 'Male' if i % 2 == 0 else 'Female',
                'location': 'Wildlife Reserve',
                'last_seen': '2024-01-01'
            }

        self.dataset_size = len(self.dataset_embeddings)
        logger.info(f"✅ Created sample dataset with {self.dataset_size} elephants")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')

            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding from image using Siamese network"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)

            # Extract embedding
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.cpu().numpy().flatten()

            return embedding

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

            # Compute cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def compare_with_dataset(self, image_path: str, threshold: float = 0.85,
                           top_k: int = 10) -> List[Dict]:
        """Compare input image with entire dataset"""
        try:
            # Extract embedding from input image
            query_embedding = self.extract_embedding(image_path)

            # Compare with all dataset embeddings
            similarities = []
            for elephant_id, dataset_embedding in self.dataset_embeddings.items():
                similarity = self.compute_similarity(query_embedding, dataset_embedding)

                if similarity >= threshold:
                    result = {
                        'elephant_id': elephant_id,
                        'confidence': similarity,
                        'metadata': self.dataset_metadata.get(elephant_id, {}),
                        'description': self.dataset_metadata.get(elephant_id, {}).get('description', ''),
                        'match_quality': self._get_match_quality(similarity)
                    }
                    similarities.append(result)

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['confidence'], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error comparing with dataset: {e}")
            return []

    def _get_match_quality(self, similarity: float) -> str:
        """Determine match quality based on similarity score"""
        if similarity >= 0.9:
            return "Excellent"
        elif similarity >= 0.8:
            return "Good"
        elif similarity >= 0.7:
            return "Fair"
        else:
            return "Poor"

    def compare_two_images(self, image_path1: str, image_path2: str) -> Dict:
        """Compare two images directly"""
        try:
            # Extract embeddings
            embedding1 = self.extract_embedding(image_path1)
            embedding2 = self.extract_embedding(image_path2)

            # Compute similarity
            similarity = self.compute_similarity(embedding1, embedding2)

            return {
                'similarity': similarity,
                'confidence': similarity,
                'match_quality': self._get_match_quality(similarity),
                'same_individual': similarity >= 0.85
            }

        except Exception as e:
            logger.error(f"Error comparing two images: {e}")
            return {'similarity': 0.0, 'error': str(e)}

    def batch_process_images(self, image_paths: List[str],
                           similarity_threshold: float = 0.85) -> Dict:
        """Process batch of images and group by similarity"""
        try:
            logger.info(f"Processing batch of {len(image_paths)} images")

            # Extract embeddings for all images
            embeddings = {}
            for i, image_path in enumerate(image_paths):
                try:
                    embedding = self.extract_embedding(image_path)
                    embeddings[image_path] = embedding

                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(image_paths)} images")

                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue

            # Group images by similarity
            groups = self._group_by_similarity(embeddings, similarity_threshold)

            return {
                'total_images': len(image_paths),
                'processed_images': len(embeddings),
                'groups': groups,
                'groups_created': len(groups)
            }

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {'error': str(e)}

    def _group_by_similarity(self, embeddings: Dict[str, np.ndarray],
                           threshold: float) -> List[List[str]]:
        """Group images by similarity using clustering approach"""
        try:
            image_paths = list(embeddings.keys())
            n_images = len(image_paths)

            if n_images == 0:
                return []

            # Compute similarity matrix
            similarity_matrix = np.zeros((n_images, n_images))
            for i in range(n_images):
                for j in range(i, n_images):
                    sim = self.compute_similarity(
                        embeddings[image_paths[i]],
                        embeddings[image_paths[j]]
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

            # Simple clustering based on threshold
            visited = set()
            groups = []

            for i in range(n_images):
                if i in visited:
                    continue

                # Start new group
                group = [image_paths[i]]
                visited.add(i)

                # Find similar images
                for j in range(i + 1, n_images):
                    if j not in visited and similarity_matrix[i, j] >= threshold:
                        group.append(image_paths[j])
                        visited.add(j)

                groups.append(group)

            # Sort groups by size
            groups.sort(key=len, reverse=True)
            return groups

        except Exception as e:
            logger.error(f"Error grouping by similarity: {e}")
            return []

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'Siamese Network',
            'backbone': 'EfficientNet-B0',
            'embedding_dim': getattr(self.model, 'embedding_dim', 128),
            'device': str(self.device),
            'dataset_size': self.dataset_size,
            'model_loaded': self.model is not None
        }
