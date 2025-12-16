#!/usr/bin/env python3
"""
Batch Process Meal Images

This script processes all before/after meal image pairs in the specified directories
using the FoodSAM + GPT-5 integration workflow.
"""

import os
import sys
import glob
import json
from pathlib import Path
from datetime import datetime
from example_workflow_complete import workflow

def find_image_pairs(directory):
    """
    Find all before/after image pairs in a directory
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        List of tuples (before_path, after_path, pair_name)
    """
    pairs = []
    
    # Find all B.jpg files (before images)
    before_pattern = os.path.join(directory, "*B.jpg")
    before_images = sorted(glob.glob(before_pattern))
    
    print(f"Found {len(before_images)} 'before' images in {directory}")
    
    for before_img in before_images:
        # Generate corresponding after image path
        after_img = before_img.replace("B.jpg", "A.jpg")
        
        if os.path.exists(after_img):
            # Extract pair name from filename
            pair_name = os.path.basename(before_img).replace("B.jpg", "").strip()
            pairs.append((before_img, after_img, pair_name))
        else:
            print(f"  Warning: No matching 'after' image for {before_img}")
    
    return pairs


def process_directory(directory, meal_type):
    """
    Process all image pairs in a directory
    
    Args:
        directory: Path to directory containing images
        meal_type: Type of meal (e.g., 'Breakfast', 'Lunch')
        
    Returns:
        Dictionary with processing results
    """
    print("\n" + "=" * 80)
    print(f"Processing {meal_type} Photos")
    print("=" * 80)
    
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return {"success": 0, "failed": 0, "pairs": []}
    
    # Find all image pairs
    pairs = find_image_pairs(directory)
    
    if not pairs:
        print(f"No image pairs found in {directory}")
        return {"success": 0, "failed": 0, "pairs": []}
    
    print(f"\nFound {len(pairs)} image pairs to process\n")
    
    results = {
        "meal_type": meal_type,
        "directory": directory,
        "total_pairs": len(pairs),
        "success": 0,
        "failed": 0,
        "pairs": []
    }
    
    # Process each pair
    for i, (before_img, after_img, pair_name) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing: {pair_name}")
        print("-" * 80)
        
        try:
            # Create output folder name with meal type
            output_folder = f"{meal_type}/{pair_name}"
            
            # Run workflow
            report = workflow(before_img, after_img, output_folder_name=output_folder)
            
            if report is not None:
                results["success"] += 1
                results["pairs"].append({
                    "pair_name": pair_name,
                    "before": before_img,
                    "after": after_img,
                    "status": "success",
                    "output_folder": f"results/{output_folder}"
                })
                print(f"✓ Successfully processed {pair_name}")
            else:
                results["failed"] += 1
                results["pairs"].append({
                    "pair_name": pair_name,
                    "before": before_img,
                    "after": after_img,
                    "status": "failed",
                    "error": "Workflow returned None"
                })
                print(f"✗ Failed to process {pair_name}")
                
        except Exception as e:
            results["failed"] += 1
            results["pairs"].append({
                "pair_name": pair_name,
                "before": before_img,
                "after": after_img,
                "status": "failed",
                "error": str(e)
            })
            print(f"✗ Error processing {pair_name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)
    
    return results


def main():
    """Main batch processing function"""
    print("=" * 80)
    print("FoodSAM Batch Processing - Meal Analysis")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define directories to process
    base_dir = "NutritionProject/Nutrition_Project"
    directories = [
        (os.path.join(base_dir, "Breakfast_Photos"), "Breakfast"),
        (os.path.join(base_dir, "Lunch_Photos"), "Lunch")
    ]
    
    all_results = []
    
    # Process each directory
    for directory, meal_type in directories:
        results = process_directory(directory, meal_type)
        all_results.append(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    
    total_success = 0
    total_failed = 0
    
    for results in all_results:
        print(f"\n{results['meal_type']}:")
        print(f"  Total pairs: {results['total_pairs']}")
        print(f"  Successful: {results['success']}")
        print(f"  Failed: {results['failed']}")
        
        total_success += results['success']
        total_failed += results['failed']
    
    print(f"\nOverall:")
    print(f"  Total processed: {total_success + total_failed}")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Success rate: {(total_success / (total_success + total_failed) * 100):.1f}%")
    
    # Save batch results to JSON
    batch_results_file = f"results/batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(batch_results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "total_pairs": total_success + total_failed,
                "successful": total_success,
                "failed": total_failed
            }
        }, f, indent=2)
    
    print(f"\nBatch results saved to: {batch_results_file}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBatch processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
