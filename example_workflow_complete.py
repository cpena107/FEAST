#!/usr/bin/env python3
"""
Complete Example: FoodSAM + GPT-5 Integration Workflow

This script demonstrates the complete workflow from GPT-5 analysis
to consumption reporting with FoodSAM segmentation.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from NutritionProject.gpt5 import analyze_single_image_with_gpt5

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from foodsam_vlm_integration import (
    FoodSAMVLMIntegration,
    ConsumptionCalculator,
    visualize_masks_overlay
)


def workflow(before_image: str, after_image: str, output_folder_name: str = None):
    """
    Complete workflow example
    
    Args:
        before_image: Path to before meal image
        after_image: Path to after meal image
        output_folder_name: Name for the output folder (optional)
    """
    print("=" * 70)
    print("FoodSAM + GPT-5 Integration - Complete Workflow Example")
    print("=" * 70)
    
    # Step 1: Load GPT-5 output
    print("\n[1/5] Generating GPT-5 analysis...")
    print(f"  Image before: {before_image}")
    print(f"  Image after: {after_image}")
    
    try:
        gpt5_response_before = analyze_single_image_with_gpt5(before_image)
        # Strip markdown code fences if present
        gpt5_response_before = gpt5_response_before.strip()
        if gpt5_response_before.startswith('```json'):
            gpt5_response_before = gpt5_response_before[7:]  # Remove ```json
        if gpt5_response_before.startswith('```'):
            gpt5_response_before = gpt5_response_before[3:]  # Remove ```
        if gpt5_response_before.endswith('```'):
            gpt5_response_before = gpt5_response_before[:-3]  # Remove trailing ```
        gpt5_response_before = gpt5_response_before.strip()
        
        # Parse the JSON response
        gpt5_data_before = json.loads(gpt5_response_before)
        print(f"  ✓ Found {len(gpt5_data_before)} food items")
        for item in gpt5_data_before:
            print(f"    - {item['food_name']}: {item['boundary_box']}")
    except json.JSONDecodeError as e:
        print(f"  ✗ Error parsing GPT-5 JSON response for before image: {e}")
        print(f"  Raw response: {gpt5_response_before[:200]}...")
        return None
    except Exception as e:
        print(f"  ✗ Error generating GPT-5 analysis for before image: {e}")
        return None
    
    try:
        gpt5_response_after = analyze_single_image_with_gpt5(after_image)
        # Strip markdown code fences if present
        gpt5_response_after = gpt5_response_after.strip()
        if gpt5_response_after.startswith('```json'):
            gpt5_response_after = gpt5_response_after[7:]  # Remove ```json
        if gpt5_response_after.startswith('```'):
            gpt5_response_after = gpt5_response_after[3:]  # Remove ```
        if gpt5_response_after.endswith('```'):
            gpt5_response_after = gpt5_response_after[:-3]  # Remove trailing ```
        gpt5_response_after = gpt5_response_after.strip()
        
        # Parse the JSON response
        gpt5_data_after = json.loads(gpt5_response_after)
        print(f"  ✓ Found {len(gpt5_data_after)} food items")
        for item in gpt5_data_after:
            print(f"    - {item['food_name']}: {item['boundary_box']}")
    except json.JSONDecodeError as e:
        print(f"  ✗ Error parsing GPT-5 JSON response for after image: {e}")
        print(f"  Raw response: {gpt5_response_after[:200]}...")
        return None
    except Exception as e:
        print(f"  ✗ Error generating GPT-5 analysis for after image: {e}")
        return None
    
    # Step 2: Initialize FoodSAM
    print("\n[2/5] Initializing FoodSAM...")
    sam_checkpoint = "ckpts/sam_vit_h_4b8939.pth"
    
    if not os.path.exists(sam_checkpoint):
        print(f"  ✗ SAM checkpoint not found: {sam_checkpoint}")
        print("  Please download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return None
    
    try:
        foodsam = FoodSAMVLMIntegration(sam_checkpoint, device='cuda')
        print("  ✓ SAM model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading SAM: {e}")
        print("  Try using device='cpu' if CUDA is not available")
        return None
    
    # Step 3: Segment images
    print("\n[3/5] Segmenting food items...")
    print(f"  Before: {before_image}")
    print(f"  After: {after_image}")
    
    try:
        results = foodsam.process_image_pair(
            before_image_path=before_image,
            after_image_path=after_image,
            vlm_bboxes_before=gpt5_data_before,
            vlm_bboxes_after=gpt5_data_after,
            is_gpt5_format=True
        )
        print("  ✓ Segmentation complete")
        
        # Show mask statistics
        for food_name, masks in results['before_masks'].items():
            total_pixels = sum(mask.sum() for mask in masks)
            print(f"    {food_name}: {len(masks)} item(s), {total_pixels} pixels")
    except Exception as e:
        print(f"  ✗ Error during segmentation: {e}")
        return None
    
    # Step 4: Calculate consumption
    print("\n[4/5] Analyzing consumption...")
    
    try:
        calculator = ConsumptionCalculator()
        report = calculator.generate_full_report(results)
        print("  ✓ Consumption analysis complete")
    except Exception as e:
        print(f"  ✗ Error calculating consumption: {e}")
        return None
    
    # Step 5: Save results
    print("\n[5/5] Saving results...")
    
    # Use custom folder name if provided, otherwise use default
    if output_folder_name:
        output_dir = f"results/{output_folder_name}"
    else:
        output_dir = "results/example_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save GPT-5 analysis (raw JSON)
        with open(f"{output_dir}/gpt5_analysis_before.json", 'w') as f:
            json.dump(gpt5_data_before, f, indent=2)
        
        with open(f"{output_dir}/gpt5_analysis_after.json", 'w') as f:
            json.dump(gpt5_data_after, f, indent=2)
        
        # Save original images with bounding boxes
        foodsam.visualize_bboxes(
            results['before_image'],
            results['pixel_bboxes_before'],
            f"{output_dir}/before_bboxes.jpg"
        )
        
        foodsam.visualize_bboxes(
            results['after_image'],
            results['pixel_bboxes_after'],
            f"{output_dir}/after_bboxes.jpg"
        )
        
        # Save individual masks as separate images
        masks_dir = f"{output_dir}/masks"
        os.makedirs(masks_dir, exist_ok=True)
        
        for food_name, masks in results['before_masks'].items():
            for i, mask in enumerate(masks):
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(f"{masks_dir}/before_{food_name}_{i+1}.png", mask_img)
        
        for food_name, masks in results['after_masks'].items():
            for i, mask in enumerate(masks):
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(f"{masks_dir}/after_{food_name}_{i+1}.png", mask_img)
        
        # Save panoptic segmentation view (colored overlay)
        before_panoptic = visualize_masks_overlay(results['before_image'], results['before_masks'], alpha=0.5)
        after_panoptic = visualize_masks_overlay(results['after_image'], results['after_masks'], alpha=0.5)
        
        cv2.imwrite(f"{output_dir}/before_panoptic.jpg", before_panoptic)
        cv2.imwrite(f"{output_dir}/after_panoptic.jpg", after_panoptic)
        
        # Save consumption report
        with open(f"{output_dir}/consumption_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✓ Results saved to {output_dir}/")
        print(f"    - gpt5_analysis_before.json")
        print(f"    - gpt5_analysis_after.json")
        print(f"    - before_bboxes.jpg")
        print(f"    - after_bboxes.jpg")
        print(f"    - before_panoptic.jpg")
        print(f"    - after_panoptic.jpg")
        print(f"    - masks/ (directory with individual masks)")
        print(f"    - consumption_report.json")
    except Exception as e:
        print(f"  ✗ Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Display results
    print("\n" + "=" * 70)
    print("CONSUMPTION REPORT")
    print("=" * 70)
    
    for food_name, data in report.items():
        print(f"\n{food_name.upper()}")
        print(f"  Items detected: {data['num_items']}")
        print(f"  Average consumption: {data['avg_percent_consumed']:.1f}%")
        
        if data['items']:
            for i, item in enumerate(data['items'], 1):
                print(f"\n  Item {i}:")
                print(f"    Before: {item['before_area_px']:,} pixels")
                print(f"    After: {item['after_area_px']:,} pixels")
                print(f"    Consumed: {item['percent_consumed']:.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ Workflow completed successfully!")
    print("=" * 70)
    
    return report


def quick_test():
    """
    Quick test with example files (if available)
    """
    # Example file paths - modify these to match your setup
    before_image = "NutritionProject/Nutrition_Project/Breakfast_Photos/Pair 2 PX3-0054B.jpg"
    after_image = "NutritionProject/Nutrition_Project/Breakfast_Photos/Pair 2 PX3-0054A.jpg"
    
    # Check if files exist
    if not os.path.exists(before_image):
        print(f"Before image not found: {before_image}")
        return False
    
    if not os.path.exists(after_image):
        print(f"After image not found: {after_image}")
        return False
    
    # Run workflow
    report = workflow(before_image, after_image)
    return report is not None


def print_usage():
    """Print usage instructions"""
    print(__doc__)
    print("\nUsage Options:")
    print("\n1. Run with custom files:")
    print("   from example_workflow_complete import workflow")
    print("   workflow('before.jpg', 'after.jpg')")
    
    print("\n2. Quick test with example files:")
    print("   from example_workflow_complete import quick_test")
    print("   quick_test()")
    
    print("\n3. Command line:")
    print("   python example_workflow_complete.py [before] [after]")
    
    print("\nBefore running, ensure you have:")
    print("  ✓ SAM checkpoint at ckpts/sam_vit_h_4b8939.pth")
    print("  ✓ Before and after images")
    print("  ✓ Required packages: cv2, numpy, segment_anything")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Command line usage with path prefix
        # Example: python script.py NutritionProject/Nutrition_Project/Breakfast_Photos/Pair 2 PX3-0361
        # Will look for files ending in B.jpg and A.jpg
        path_prefix = sys.argv[1]
        before = path_prefix + "B.jpg"
        after = path_prefix + "A.jpg"
        
        # Extract folder name from the path prefix for output directory
        # Get the last part of the path (e.g., "Pair 2 PX3-0361" from the full path)
        folder_name = os.path.basename(path_prefix)
        if not folder_name:
            # If basename is empty (path ends with /), get the parent
            folder_name = os.path.basename(os.path.dirname(path_prefix))
        
        if not all(os.path.exists(f) for f in [before, after]):
            print("Error: One or more files not found")
            print(f"  Before: {before} - {'✓' if os.path.exists(before) else '✗'}")
            print(f"  After: {after} - {'✓' if os.path.exists(after) else '✗'}")
            sys.exit(1)
        
        workflow(before, after, output_folder_name=os.path.join(path_prefix.split('/')[-2], folder_name))
    else:
        print_usage()
        print("\nAttempting quick test with example files...")
        success = quick_test()
        
        if not success:
            print("\n" + "=" * 70)
            print("Quick test could not run. Please check file paths.")
            print("=" * 70)
