#!/usr/bin/env python3
"""
FoodSAM + VLM Integration Module
Combines GPT-4o Vision bounding boxes with SAM segmentation for food analysis
"""

import cv2
import numpy as np
import json
from segment_anything import sam_model_registry, SamPredictor
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import os
try:
    from FoodSAM.consumption_calculator import ConsumptionCalculator
except ImportError:
    # ConsumptionCalculator is defined in this file
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gpt5_output_from_file(file_path: str) -> List[Dict]:
    """
    Load and parse GPT-5 output from a text file
    
    Args:
        file_path: Path to the GPT-5 output file
        
    Returns:
        List of food items with bounding boxes
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Handle markdown code blocks
    if '```json' in content:
        json_str = content.split('```json')[1].split('```')[0].strip()
    elif '```' in content:
        json_str = content.split('```')[1].split('```')[0].strip()
    else:
        json_str = content.strip()
    
    return json.loads(json_str)


class FoodSAMVLMIntegration:
    """
    Integrates Vision-Language Model (VLM) bounding boxes with 
    Segment Anything Model (SAM) for precise food segmentation
    """
    
    def __init__(self, sam_checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda"):
        """
        Initialize SAM model for prompted segmentation
        
        Args:
            sam_checkpoint_path: Path to SAM checkpoint file
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            device: Device to run on ('cuda' or 'cpu')
        """
        logger.info(f"Loading SAM model: {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.device = device
        logger.info("SAM model loaded successfully")
    
    def convert_bboxes_to_pixels(self, vlm_bboxes: Dict, width: int, height: int) -> Dict:
        """
        Convert normalized bounding boxes (0-1) to pixel coordinates
        
        Args:
            vlm_bboxes: Bounding boxes from VLM in format:
                       {'category': {'items': [[x_min, y_min, x_max, y_max], ...]}}
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Dictionary with pixel coordinates
        """
        pixel_bboxes = {}
        
        for category, data in vlm_bboxes.items():
            pixel_bboxes[category] = {'items': []}
            
            # Handle both old format (float) and new format (dict with items)
            if isinstance(data, dict) and 'items' in data:
                # New format: {'items': [[x_min, y_min, x_max, y_max], ...]}
                for bbox in data['items']:
                    if len(bbox) == 4:
                        x_min, y_min, x_max, y_max = bbox
                        pixel_bbox = [
                            int(x_min * width),
                            int(y_min * height),
                            int(x_max * width),
                            int(y_max * height)
                        ]
                        pixel_bboxes[category]['items'].append(pixel_bbox)
            else:
                # Old format: just a float consumption value - no bounding boxes available
                logger.warning(f"Category '{category}' has old format (no bounding boxes). Skipping.")
                logger.warning("Please update gpt4o_analysis.py to return bounding boxes!")
        
        return pixel_bboxes
    
    def parse_gpt5_output(self, gpt5_data: List[Dict], original_width: int = None, original_height: int = None) -> Dict:
        """
        Parse GPT-5 output format to internal format
        
        Args:
            gpt5_data: List of food items from gpt5.py in format:
                      [{"food_name": "banana", "boundary_box": [x_min, y_min, x_max, y_max]}, ...]
                      Bounding boxes are in 512x512 pixel coordinates (GPT-5 resizes images)
            original_width: Width of the original image (for scaling bounding boxes)
            original_height: Height of the original image (for scaling bounding boxes)
                      
        Returns:
            Dictionary mapping food names to bounding boxes:
            {'banana': {'items': [[x_min, y_min, x_max, y_max]]}, ...}
        """
        parsed_bboxes = {}
        
        # GPT-5 analyzes 512x512 resized images, so we need to scale bounding boxes
        scale_x = original_width / 512.0 if original_width else 1.0
        scale_y = original_height / 512.0 if original_height else 1.0
        
        logger.info(f"Scaling GPT-5 bounding boxes: {scale_x:.2f}x width, {scale_y:.2f}x height")
        
        for item in gpt5_data:
            food_name = item.get('food_name', 'unknown')
            bbox = item.get('boundary_box', [])
            
            if len(bbox) == 4:
                # Initialize food category if not exists
                if food_name not in parsed_bboxes:
                    parsed_bboxes[food_name] = {'items': []}
                
                # Scale bounding box from 512x512 to original image dimensions
                scaled_bbox = [
                    int(bbox[0] * scale_x),  # x_min
                    int(bbox[1] * scale_y),  # y_min
                    int(bbox[2] * scale_x),  # x_max
                    int(bbox[3] * scale_y)   # y_max
                ]
                
                parsed_bboxes[food_name]['items'].append(scaled_bbox)
                logger.debug(f"Added {food_name}: {bbox} -> {scaled_bbox}")
            else:
                logger.warning(f"Invalid bounding box for {food_name}: {bbox}")
        
        return parsed_bboxes
    
    def segment_from_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Segment image region given a bounding box using SAM
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            bbox: Bounding box [x_min, y_min, x_max, y_max] in pixel coordinates
            
        Returns:
            Binary segmentation mask (H, W) as boolean array
        """
        self.predictor.set_image(image)
        
        # Convert bbox to SAM format [x_min, y_min, x_max, y_max]
        input_box = np.array(bbox)
        
        # Get mask prediction
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=False  # Return single best mask
        )
        
        return masks[0]  # Return best mask
    
    def segment_image_by_categories(self, image: np.ndarray, 
                                   category_bboxes: Dict) -> Dict[str, List[np.ndarray]]:
        """
        Segment all food items by category/name using bounding box prompts
        
        Args:
            image: Input image as numpy array (H, W, 3)
            category_bboxes: Dictionary of bounding boxes per food name/category
            
        Returns:
            Dictionary mapping food names to lists of binary masks
        """
        category_masks = {}
        
        for food_name, data in category_bboxes.items():
            category_masks[food_name] = []
            
            logger.info(f"Segmenting {len(data['items'])} items for: {food_name}")
            
            for i, bbox in enumerate(data['items']):
                if len(bbox) == 4:  # Valid bbox
                    try:
                        mask = self.segment_from_bbox(image, bbox)
                        category_masks[food_name].append(mask)
                        logger.debug(f"  Item {i+1}: mask generated ({mask.sum()} pixels)")
                    except Exception as e:
                        logger.error(f"  Error segmenting item {i+1} for {food_name}: {e}")
                        # Add empty mask as placeholder
                        category_masks[food_name].append(np.zeros(image.shape[:2], dtype=bool))
        
        return category_masks
    
    def process_image_pair(self, before_image_path: str, after_image_path: str, 
                          vlm_bboxes_before: Dict, vlm_bboxes_after: Dict, is_gpt5_format: bool = False) -> Dict:
        """
        Process before/after image pair with VLM bounding boxes
        
        Args:
            before_image_path: Path to before image
            after_image_path: Path to after image
            vlm_bboxes_before: Bounding boxes from VLM for before image (normalized coordinates) OR 
                              GPT-5 format list (if is_gpt5_format=True)
            vlm_bboxes_after: Bounding boxes from VLM for after image (normalized coordinates) OR 
                             GPT-5 format list (if is_gpt5_format=True)
            is_gpt5_format: If True, vlm_bboxes is in GPT-5 format 
                          (list of dicts with food_name and boundary_box)
            
        Returns:
            Dictionary containing:
                - before_masks: food_name -> list of masks
                - after_masks: food_name -> list of masks
                - before_image: BGR image array
                - after_image: BGR image array
                - pixel_bboxes_before: bounding boxes in pixel coordinates for before image
                - pixel_bboxes_after: bounding boxes in pixel coordinates for after image
        """
        logger.info(f"Processing image pair:")
        logger.info(f"  Before: {before_image_path}")
        logger.info(f"  After: {after_image_path}")
        
        # Load images
        before_img = cv2.imread(before_image_path)
        after_img = cv2.imread(after_image_path)
        
        if before_img is None:
            raise FileNotFoundError(f"Cannot load before image: {before_image_path}")
        if after_img is None:
            raise FileNotFoundError(f"Cannot load after image: {after_image_path}")
        
        h_before, w_before = before_img.shape[:2]
        h_after, w_after = after_img.shape[:2]
        logger.info(f"  Before image dimensions: {w_before}x{h_before}")
        logger.info(f"  After image dimensions: {w_after}x{h_after}")
        
        # Parse bounding boxes based on format
        if is_gpt5_format:
            logger.info("Using GPT-5 format (food items by name)")
            pixel_bboxes_before = self.parse_gpt5_output(vlm_bboxes_before, w_before, h_before)
            pixel_bboxes_after = self.parse_gpt5_output(vlm_bboxes_after, w_after, h_after)
        else:
            logger.info("Using category-based format")
            # Convert normalized coords to pixels
            pixel_bboxes_before = self.convert_bboxes_to_pixels(vlm_bboxes_before, w_before, h_before)
            pixel_bboxes_after = self.convert_bboxes_to_pixels(vlm_bboxes_after, w_after, h_after)
        
        # Segment both images with their respective bounding boxes
        logger.info("Segmenting BEFORE image...")
        before_masks = self.segment_image_by_categories(before_img, pixel_bboxes_before)
        
        logger.info("Segmenting AFTER image...")
        after_masks = self.segment_image_by_categories(after_img, pixel_bboxes_after)
        
        return {
            'before_masks': before_masks,
            'after_masks': after_masks,
            'before_image': before_img,
            'after_image': after_img,
            'pixel_bboxes_before': pixel_bboxes_before,
            'pixel_bboxes_after': pixel_bboxes_after
        }
    
    def visualize_bboxes(self, image: np.ndarray, bboxes: Dict, 
                        output_path: Optional[str] = None) -> np.ndarray:
        """
        Draw bounding boxes on image for visualization
        
        Args:
            image: Input image
            bboxes: Bounding boxes in pixel coordinates (food_name -> {'items': [bbox, ...]})
            output_path: Optional path to save visualization
            
        Returns:
            Image with bounding boxes drawn
        """
        vis_image = image.copy()
        
        # Color map for common food categories
        # Generate colors dynamically for food names not in the map
        predefined_colors = {
            'fruit': (0, 0, 255),        # Red
            'vegetable': (0, 255, 0),     # Green
            'whole_grain': (255, 165, 0), # Orange
            'sugary_bev': (255, 0, 0),    # Blue
            'banana': (0, 255, 255),      # Yellow
            'milk': (255, 255, 255),      # White
            'muffin': (165, 42, 42),      # Brown
            'strawberry': (255, 0, 127),  # Pink
            'chicken': (0, 128, 255),     # Light Blue
            'cereal': (255, 215, 0),       # Gold
            'apple juice': (0, 255, 128), # Light Green
            'other': (128, 128, 128)      # Gray
        }
        
        for food_name, data in bboxes.items():
            # Get color or generate a random one
            if food_name in predefined_colors:
                color = predefined_colors[food_name]
            else:
                # Generate deterministic color based on food name hash
                hash_val = hash(food_name)
                color = (
                    (hash_val & 0xFF),
                    ((hash_val >> 8) & 0xFF),
                    ((hash_val >> 16) & 0xFF)
                )
            
            for i, bbox in enumerate(data['items']):
                x_min, y_min, x_max, y_max = bbox
                
                # Draw rectangle
                cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Add label
                label = f"{food_name}_{i+1}" if len(data['items']) > 1 else food_name
                cv2.putText(vis_image, label, (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Saved bbox visualization to {output_path}")
        
        return vis_image


class ConsumptionCalculator:
    """
    Calculate food consumption metrics by comparing before/after segmentation masks
    """
    
    def __init__(self, method: str = "mask_area"):
        """
        Initialize consumption calculator
        
        Args:
            method: Calculation method
                   - 'mask_area': Compare mask areas
                   - 'pixel_intensity': Compare pixel intensities within masks
        """
        self.method = method
    
    def calculate_mask_area(self, mask: np.ndarray) -> int:
        """
        Calculate area (number of pixels) in a binary mask
        
        Args:
            mask: Binary mask array
            
        Returns:
            Number of True pixels in mask
        """
        return int(mask.sum())
    
    def compare_masks(self, before_mask: np.ndarray, after_mask: np.ndarray) -> Dict:
        """
        Compare before/after masks to calculate consumption
        
        Args:
            before_mask: Binary mask from before image
            after_mask: Binary mask from after image
            
        Returns:
            Dictionary with consumption metrics
        """
        before_area = self.calculate_mask_area(before_mask)
        after_area = self.calculate_mask_area(after_mask)
        
        # Calculate consumption ratio
        if before_area == 0:
            consumption_ratio = 0.0
        else:
            area_diff = before_area - after_area
            consumption_ratio = area_diff / before_area
            consumption_ratio = np.clip(consumption_ratio, 0.0, 1.0)
        
        return {
            'consumption_ratio': float(consumption_ratio),
            'before_area_px': before_area,
            'after_area_px': after_area,
            'area_diff_px': before_area - after_area,
            'percent_consumed': float(consumption_ratio * 100)
        }
    
    def analyze_category_consumption(self, segmentation_results: Dict, 
                                    food_name: str) -> List[Dict]:
        """
        Analyze consumption for all items of a specific food name
        
        Args:
            segmentation_results: Output from FoodSAMVLMIntegration.process_image_pair()
            food_name: Food name (e.g., 'banana', 'milk', 'muffin')
            
        Returns:
            List of consumption metrics for each item of this food
        """
        food_analysis = []
        
        before_masks = segmentation_results['before_masks'].get(food_name, [])
        after_masks = segmentation_results['after_masks'].get(food_name, [])
        
        # Handle cases where items are in before but not after (100% consumed)
        # or different numbers of items detected
        num_before = len(before_masks)
        num_after = len(after_masks)
        
        # Process items that exist in both before and after
        num_matched = min(num_before, num_after)
        for i in range(num_matched):
            metrics = self.compare_masks(before_masks[i], after_masks[i])
            metrics['item_index'] = i
            metrics['food_name'] = food_name
            food_analysis.append(metrics)
        
        # Handle items that exist in before but not in after (completely consumed)
        if num_before > num_after:
            for i in range(num_after, num_before):
                before_area = self.calculate_mask_area(before_masks[i])
                metrics = {
                    'consumption_ratio': 1.0,  # 100% consumed
                    'before_area_px': before_area,
                    'after_area_px': 0,
                    'area_diff_px': before_area,
                    'percent_consumed': 100.0,
                    'item_index': i,
                    'food_name': food_name
                }
                food_analysis.append(metrics)
        
        # Handle items that exist in after but not in before (newly added - negative consumption)
        # This is unusual but we should handle it
        if num_after > num_before:
            for i in range(num_before, num_after):
                after_area = self.calculate_mask_area(after_masks[i])
                metrics = {
                    'consumption_ratio': -1.0,  # Negative consumption (food added)
                    'before_area_px': 0,
                    'after_area_px': after_area,
                    'area_diff_px': -after_area,
                    'percent_consumed': -100.0,
                    'item_index': i,
                    'food_name': food_name
                }
                food_analysis.append(metrics)
        
        return food_analysis
    
    def generate_full_report(self, segmentation_results: Dict) -> Dict:
        """
        Generate complete consumption report for all food items
        
        Args:
            segmentation_results: Output from FoodSAMVLMIntegration.process_image_pair()
            
        Returns:
            Dictionary with consumption analysis per food name
        """
        report = {}
        
        food_names = segmentation_results['before_masks'].keys()
        
        for food_name in food_names:
            food_metrics = self.analyze_category_consumption(
                segmentation_results, food_name
            )
            
            if food_metrics:
                # Calculate food summary
                total_consumption = np.mean([m['consumption_ratio'] 
                                            for m in food_metrics])
                
                report[food_name] = {
                    'items': food_metrics,
                    'num_items': len(food_metrics),
                    'avg_consumption': float(total_consumption),
                    'avg_percent_consumed': float(total_consumption * 100)
                }
            else:
                report[food_name] = {
                    'items': [],
                    'num_items': 0,
                    'avg_consumption': 0.0,
                    'avg_percent_consumed': 0.0
                }
        
        return report


def visualize_masks_overlay(image: np.ndarray, food_masks: Dict[str, List[np.ndarray]], 
                            alpha: float = 0.5) -> np.ndarray:
    """
    Create colored overlay of segmentation masks on image
    
    Args:
        image: Input BGR image
        food_masks: Dictionary mapping food names to lists of masks
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        
    Returns:
        Image with colored mask overlays
    """
    overlay = image.copy()
    
    # Color map for common foods (BGR format for OpenCV)
    predefined_colors = {
        'fruit': (0, 0, 255),        # Red
        'vegetable': (0, 255, 0),     # Green
        'whole_grain': (0, 165, 255), # Orange
        'sugary_bev': (255, 0, 0),    # Blue
        'banana': (0, 255, 255),      # Yellow
        'milk': (255, 255, 255),      # White
        'muffin': (165, 42, 42),      # Brown
        'strawberry': (255, 0, 127),  # Pink
        'chicken': (0, 128, 255),    # Light Blue
        'cereal': (0, 215, 255),      # Gold
        'apple juice': (0, 255, 128), # Light Green
        'other': (128, 128, 128)      # Gray
    }
    
    for food_name, masks in food_masks.items():
        # Get color or generate a random one
        if food_name in predefined_colors:
            color = predefined_colors[food_name]
        else:
            # Generate deterministic color based on food name hash
            hash_val = hash(food_name)
            color = (
                (hash_val & 0xFF),
                ((hash_val >> 8) & 0xFF),
                ((hash_val >> 16) & 0xFF)
            )
        
        for mask in masks:
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask] = color
            
            # Blend with original image
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return overlay
