#!/usr/bin/env python3
"""
Batch Processing Module for Elephant Identification
Handles large-scale processing with both Siamese and YOLOv8 models
"""

import os
import json
import shutil
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles batch processing for both Siamese and YOLOv8 models"""

    def __init__(self, siamese_processor=None, yolo_processor=None):
        self.siamese_processor = siamese_processor
        self.yolo_processor = yolo_processor
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def get_image_files(self, folder_path: str) -> List[str]:
        """Recursively find all image files in folder"""
        image_files = []

        try:
            folder_path = Path(folder_path)
            for file_path in folder_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    image_files.append(str(file_path))

            logger.info(f"Found {len(image_files)} image files in {folder_path}")
            return sorted(image_files)

        except Exception as e:
            logger.error(f"Error finding image files: {e}")
            return []

    def process_folder(self, folder_path: str, model_type: str = 'siamese',
                      similarity_threshold: float = 0.85, max_groups: Optional[int] = None,
                      output_format: str = 'zip') -> Dict:
        """Process entire folder with specified model"""
        try:
            logger.info(f"Starting batch processing: {folder_path}")
            logger.info(f"Model: {model_type}, Threshold: {similarity_threshold}")

            # Get all image files
            image_files = self.get_image_files(folder_path)
            if not image_files:
                return {'error': 'No image files found in the specified folder'}

            # Process based on model type
            if model_type == 'siamese':
                results = self._process_with_siamese(
                    image_files, similarity_threshold, max_groups
                )
            elif model_type == 'yolo':
                results = self._process_with_yolo(
                    image_files, similarity_threshold, max_groups
                )
            elif model_type == 'both':
                results = self._process_with_both_models(
                    image_files, similarity_threshold, max_groups
                )
            else:
                return {'error': f'Unknown model type: {model_type}'}

            # Generate output based on format
            if output_format == 'zip':
                output_file = self._create_zip_output(results, folder_path)
                results['output_file'] = output_file
            elif output_format == 'folder':
                output_folder = self._create_folder_output(results, folder_path)
                results['output_folder'] = output_folder
            elif output_format == 'csv':
                csv_file = self._create_csv_output(results, folder_path)
                results['csv_file'] = csv_file

            # Generate summary report
            results['summary_report'] = self._generate_summary_report(results)

            logger.info("✅ Batch processing completed successfully")
            return results

        except Exception as e:
            logger.error(f"❌ Error in batch processing: {e}")
            return {'error': str(e)}

    def _process_with_siamese(self, image_files: List[str], threshold: float,
                             max_groups: Optional[int]) -> Dict:
        """Process images using Siamese network"""
        if not self.siamese_processor:
            return {'error': 'Siamese processor not available'}

        try:
            logger.info("Processing with Siamese Network...")

            # Use Siamese processor's batch processing
            results = self.siamese_processor.batch_process_images(image_files, threshold)

            if 'error' in results:
                return results

            # Limit groups if specified
            if max_groups and len(results['groups']) > max_groups:
                # Keep largest groups
                groups = sorted(results['groups'], key=len, reverse=True)[:max_groups]
                results['groups'] = groups
                results['groups_created'] = len(groups)

            # Calculate additional statistics
            results.update(self._calculate_siamese_stats(results))

            return results

        except Exception as e:
            logger.error(f"Error processing with Siamese: {e}")
            return {'error': str(e)}

    def _process_with_yolo(self, image_files: List[str], threshold: float,
                          max_groups: Optional[int]) -> Dict:
        """Process images using YOLOv8"""
        if not self.yolo_processor:
            return {'error': 'YOLOv8 processor not available'}

        try:
            logger.info("Processing with YOLOv8...")

            # Run YOLOv8 batch detection
            yolo_results = self.yolo_processor.batch_detect(
                image_files, confidence_threshold=threshold
            )

            if 'error' in yolo_results:
                return yolo_results

            # Group images based on detection patterns
            groups = self._group_by_detection_patterns(
                yolo_results['results'], max_groups
            )

            results = {
                'total_images': len(image_files),
                'processed_images': yolo_results['summary']['processed_images'],
                'groups': groups,
                'groups_created': len(groups),
                'detection_summary': yolo_results['summary'],
                'model_type': 'yolo'
            }

            return results

        except Exception as e:
            logger.error(f"Error processing with YOLOv8: {e}")
            return {'error': str(e)}

    def _process_with_both_models(self, image_files: List[str], threshold: float,
                                 max_groups: Optional[int]) -> Dict:
        """Process images using both models"""
        try:
            logger.info("Processing with both Siamese and YOLOv8...")

            # Process with both models
            siamese_results = self._process_with_siamese(image_files, threshold, None)
            yolo_results = self._process_with_yolo(image_files, threshold, None)

            # Combine results
            combined_results = self._combine_model_results(
                siamese_results, yolo_results, max_groups
            )

            return combined_results

        except Exception as e:
            logger.error(f"Error processing with both models: {e}")
            return {'error': str(e)}

    def _group_by_detection_patterns(self, yolo_results: List[Dict],
                                   max_groups: Optional[int]) -> List[List[str]]:
        """Group images based on YOLOv8 detection patterns"""
        try:
            # Group by number of elephants detected
            groups_by_count = {}

            for result in yolo_results:
                count = result['total_detections']
                image_path = result['image_path']

                if count not in groups_by_count:
                    groups_by_count[count] = []
                groups_by_count[count].append(image_path)

            # Convert to list of groups
            groups = list(groups_by_count.values())

            # Sort by group size
            groups.sort(key=len, reverse=True)

            # Limit groups if specified
            if max_groups and len(groups) > max_groups:
                groups = groups[:max_groups]

            return groups

        except Exception as e:
            logger.error(f"Error grouping by detection patterns: {e}")
            return []

    def _combine_model_results(self, siamese_results: Dict, yolo_results: Dict,
                              max_groups: Optional[int]) -> Dict:
        """Combine results from both models"""
        try:
            # Start with Siamese groups as base
            siamese_groups = siamese_results.get('groups', [])
            yolo_groups = yolo_results.get('groups', [])

            # Cross-validate groups using both models
            validated_groups = []

            for siamese_group in siamese_groups:
                # Check if images in this group also have similar detection patterns
                detection_counts = []
                for img_path in siamese_group:
                    # Find detection count for this image
                    count = 0
                    for yolo_result in yolo_results.get('detection_summary', {}).get('results', []):
                        if yolo_result.get('image_path') == img_path:
                            count = yolo_result.get('total_detections', 0)
                            break
                    detection_counts.append(count)

                # If detection counts are similar, keep the group
                if len(set(detection_counts)) <= 2:  # Allow some variation
                    validated_groups.append(siamese_group)

            # Limit groups if specified
            if max_groups and len(validated_groups) > max_groups:
                validated_groups = validated_groups[:max_groups]

            return {
                'total_images': siamese_results.get('total_images', 0),
                'processed_images': min(
                    siamese_results.get('processed_images', 0),
                    yolo_results.get('processed_images', 0)
                ),
                'groups': validated_groups,
                'groups_created': len(validated_groups),
                'siamese_analysis': siamese_results,
                'yolo_analysis': yolo_results,
                'model_type': 'both',
                'validation_method': 'cross_model_validation'
            }

        except Exception as e:
            logger.error(f"Error combining model results: {e}")
            return {'error': str(e)}

    def _calculate_siamese_stats(self, results: Dict) -> Dict:
        """Calculate additional statistics for Siamese results"""
        try:
            groups = results.get('groups', [])

            if not groups:
                return {}

            group_sizes = [len(group) for group in groups]

            stats = {
                'average_group_size': np.mean(group_sizes),
                'largest_group_size': max(group_sizes),
                'smallest_group_size': min(group_sizes),
                'singleton_groups': sum(1 for size in group_sizes if size == 1),
                'multi_image_groups': sum(1 for size in group_sizes if size > 1)
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating Siamese stats: {e}")
            return {}

    def _create_zip_output(self, results: Dict, original_folder: str) -> str:
        """Create ZIP file with organized results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"elephant_groups_{timestamp}.zip"
            zip_path = os.path.join("temp_results", zip_filename)

            os.makedirs("temp_results", exist_ok=True)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                groups = results.get('groups', [])

                # Add grouped images
                for i, group in enumerate(groups, 1):
                    group_folder = f"Group_{i:03d}/"

                    for image_path in group:
                        # Get relative path from original folder
                        rel_path = os.path.relpath(image_path, original_folder)
                        filename = os.path.basename(image_path)

                        # Add to ZIP with group structure
                        zipf.write(image_path, group_folder + filename)

                # Add summary report
                summary_content = self._generate_text_summary(results)
                zipf.writestr("GROUPING_SUMMARY.txt", summary_content)

                # Add detailed JSON report
                json_content = json.dumps(results, indent=2, default=str)
                zipf.writestr("detailed_results.json", json_content)

            logger.info(f"Created ZIP output: {zip_path}")
            return zip_path

        except Exception as e:
            logger.error(f"Error creating ZIP output: {e}")
            return None

    def _create_folder_output(self, results: Dict, original_folder: str) -> str:
        """Create organized folder structure with results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join("temp_results", f"elephant_groups_{timestamp}")

            os.makedirs(output_folder, exist_ok=True)

            groups = results.get('groups', [])

            # Create group folders and copy images
            for i, group in enumerate(groups, 1):
                group_folder = os.path.join(output_folder, f"Group_{i:03d}")
                os.makedirs(group_folder, exist_ok=True)

                for image_path in group:
                    filename = os.path.basename(image_path)
                    dest_path = os.path.join(group_folder, filename)
                    shutil.copy2(image_path, dest_path)

            # Create summary files
            summary_path = os.path.join(output_folder, "GROUPING_SUMMARY.txt")
            with open(summary_path, 'w') as f:
                f.write(self._generate_text_summary(results))

            json_path = os.path.join(output_folder, "detailed_results.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Created folder output: {output_folder}")
            return output_folder

        except Exception as e:
            logger.error(f"Error creating folder output: {e}")
            return None

    def _create_csv_output(self, results: Dict, original_folder: str) -> str:
        """Create CSV report of results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"elephant_analysis_{timestamp}.csv"
            csv_path = os.path.join("temp_results", csv_filename)

            os.makedirs("temp_results", exist_ok=True)

            with open(csv_path, 'w') as f:
                # Write header
                f.write("Group_ID,Image_Path,Filename,Group_Size,Model_Used\n")

                # Write data
                groups = results.get('groups', [])
                model_type = results.get('model_type', 'unknown')

                for i, group in enumerate(groups, 1):
                    group_size = len(group)
                    for image_path in group:
                        filename = os.path.basename(image_path)
                        rel_path = os.path.relpath(image_path, original_folder)
                        f.write(f"{i},{rel_path},{filename},{group_size},{model_type}\n")

            logger.info(f"Created CSV output: {csv_path}")
            return csv_path

        except Exception as e:
            logger.error(f"Error creating CSV output: {e}")
            return None

    def _generate_summary_report(self, results: Dict) -> Dict:
        """Generate comprehensive summary report"""
        try:
            groups = results.get('groups', [])
            model_type = results.get('model_type', 'unknown')

            summary = {
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'model_used': model_type,
                    'total_images_found': results.get('total_images', 0),
                    'images_successfully_processed': results.get('processed_images', 0)
                },
                'grouping_results': {
                    'total_groups_created': len(groups),
                    'images_grouped': sum(len(group) for group in groups),
                    'largest_group_size': max(len(group) for group in groups) if groups else 0,
                    'average_group_size': np.mean([len(group) for group in groups]) if groups else 0
                }
            }

            # Add model-specific information
            if model_type == 'siamese':
                summary['siamese_info'] = {
                    'similarity_threshold': results.get('similarity_threshold', 'unknown'),
                    'singleton_groups': results.get('singleton_groups', 0),
                    'multi_image_groups': results.get('multi_image_groups', 0)
                }
            elif model_type == 'yolo':
                summary['yolo_info'] = {
                    'detection_summary': results.get('detection_summary', {}),
                    'grouping_method': 'detection_pattern_similarity'
                }
            elif model_type == 'both':
                summary['combined_info'] = {
                    'validation_method': results.get('validation_method', 'unknown'),
                    'siamese_groups_before_validation': len(results.get('siamese_analysis', {}).get('groups', [])),
                    'final_validated_groups': len(groups)
                }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {'error': str(e)}

    def _generate_text_summary(self, results: Dict) -> str:
        """Generate human-readable text summary"""
        try:
            summary = results.get('summary_report', {})
            groups = results.get('groups', [])

            text = "🐘 ELEPHANT IDENTIFICATION SYSTEM - BATCH PROCESSING REPORT\n"
            text += "=" * 70 + "\n\n"

            # Processing info
            proc_info = summary.get('processing_info', {})
            text += f"📅 Processing Date: {proc_info.get('timestamp', 'Unknown')}\n"
            text += f"🤖 Model Used: {proc_info.get('model_used', 'Unknown')}\n"
            text += f"📊 Total Images Found: {proc_info.get('total_images_found', 0)}\n"
            text += f"✅ Images Successfully Processed: {proc_info.get('images_successfully_processed', 0)}\n\n"

            # Grouping results
            group_info = summary.get('grouping_results', {})
            text += "📈 GROUPING RESULTS:\n"
            text += f"  • Total Groups Created: {group_info.get('total_groups_created', 0)}\n"
            text += f"  • Images Grouped: {group_info.get('images_grouped', 0)}\n"
            text += f"  • Largest Group Size: {group_info.get('largest_group_size', 0)}\n"
            text += f"  • Average Group Size: {group_info.get('average_group_size', 0):.2f}\n\n"

            # Model-specific info
            model_type = proc_info.get('model_used', '')
            if model_type == 'siamese':
                siamese_info = summary.get('siamese_info', {})
                text += "🔍 SIAMESE NETWORK DETAILS:\n"
                text += f"  • Similarity Threshold: {siamese_info.get('similarity_threshold', 'Unknown')}\n"
                text += f"  • Singleton Groups: {siamese_info.get('singleton_groups', 0)}\n"
                text += f"  • Multi-image Groups: {siamese_info.get('multi_image_groups', 0)}\n\n"

            elif model_type == 'yolo':
                yolo_info = summary.get('yolo_info', {})
                detection_summary = yolo_info.get('detection_summary', {})
                text += "🎯 YOLOv8 DETECTION DETAILS:\n"
                text += f"  • Total Detections: {detection_summary.get('total_detections', 0)}\n"
                text += f"  • Images with Elephants: {detection_summary.get('images_with_detections', 0)}\n"
                text += f"  • Avg Detections per Image: {detection_summary.get('average_detections_per_image', 0):.2f}\n\n"

            elif model_type == 'both':
                combined_info = summary.get('combined_info', {})
                text += "🔄 COMBINED MODEL ANALYSIS:\n"
                text += f"  • Validation Method: {combined_info.get('validation_method', 'Unknown')}\n"
                text += f"  • Initial Siamese Groups: {combined_info.get('siamese_groups_before_validation', 0)}\n"
                text += f"  • Final Validated Groups: {combined_info.get('final_validated_groups', 0)}\n\n"

            # Group breakdown
            text += "📋 GROUP BREAKDOWN:\n"
            for i, group in enumerate(groups[:10], 1):  # Show first 10 groups
                text += f"  Group {i}: {len(group)} images\n"
                for img_path in group[:3]:  # Show first 3 images per group
                    filename = os.path.basename(img_path)
                    text += f"    - {filename}\n"
                if len(group) > 3:
                    text += f"    ... and {len(group) - 3} more images\n"
                text += "\n"

            if len(groups) > 10:
                text += f"  ... and {len(groups) - 10} more groups\n\n"

            text += "=" * 70 + "\n"
            text += "Generated by Elephant Identification System v1.0.0\n"
            text += "For technical support, check the documentation or GitHub repository.\n"

            return text

        except Exception as e:
            logger.error(f"Error generating text summary: {e}")
            return f"Error generating summary: {str(e)}"
