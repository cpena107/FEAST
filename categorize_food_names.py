#!/usr/bin/env python3
"""
Categorize Food Names into Nutritional Categories

This script reads the unique food names and categorizes them into:
- Fruit
- Vegetable
- Whole Grain
- Sugary Beverage
- Other
"""

import json
import os


def load_food_names(file_path="results/unique_food_names.txt"):
    """Load unique food names from file"""
    food_names = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    for line in lines:
        line = line.strip()
        if line and line != "Unique Food Names" and not line.startswith("="):
            food_names.append(line)
    
    return food_names


def categorize_food_item(food_name):
    """
    Categorize a food item into one of the nutritional categories
    
    Args:
        food_name: Name of the food item (case-insensitive matching)
        
    Returns:
        Category name: 'Fruit', 'Vegetable', 'Whole Grain', 'Sugary Beverage', or 'Other'
    """
    food_lower = food_name.lower()
    
    # Define keywords for each category
    
    # FRUIT keywords
    fruit_keywords = [
        'apple', 'banana', 'orange', 'grape', 'strawberr', 'blueberr',
        'melon', 'cantaloupe', 'pineapple', 'peach', 'fruit', 'avocado'
    ]
    
    # VEGETABLE keywords
    vegetable_keywords = [
        'vegetable', 'carrot', 'broccoli', 'cauliflower', 'cucumber',
        'lettuce', 'salad', 'sweet potato'
    ]
    
    # WHOLE GRAIN keywords
    whole_grain_keywords = [
        'bread', 'muffin', 'cereal', 'cheerios', 'rice', 'pasta',
        'tortilla', 'bun', 'roll', 'wheat', 'whole grain', 'whole_grain',
        'croissant', 'quesadilla', 'wrap', 'biscuit'
    ]
    
    # SUGARY BEVERAGE keywords
    sugary_beverage_keywords = [
        'juice', 'capri sun', 'powerade', 'sports drink', 'sugary_bev',
        'chocolate milk', 'strawberry milk'
    ]
    
    # Check for fruits
    for keyword in fruit_keywords:
        if keyword in food_lower:
            return 'Fruit'
    
    # Check for vegetables (before checking sweet potato -> other)
    for keyword in vegetable_keywords:
        if keyword in food_lower:
            return 'Vegetable'
    
    # Check for whole grains
    for keyword in whole_grain_keywords:
        if keyword in food_lower:
            return 'Whole Grain'
    
    # Check for sugary beverages (but exclude milk)
    if 'milk' not in food_lower:
        for keyword in sugary_beverage_keywords:
            if keyword in food_lower:
                return 'Sugary Beverage'
    
    # Everything else is "Other"
    return 'Other'


def categorize_all_foods(food_names):
    """
    Categorize all food names
    
    Args:
        food_names: List of food names
        
    Returns:
        Dictionary with categories as keys and lists of food names as values
    """
    categories = {
        'Fruit': [],
        'Vegetable': [],
        'Whole Grain': [],
        'Sugary Beverage': [],
        'Other': []
    }
    
    for food_name in food_names:
        category = categorize_food_item(food_name)
        categories[category].append(food_name)
    
    return categories


def save_categorized_foods(categories, output_dir="results"):
    """
    Save categorized food names to files
    
    Args:
        categories: Dictionary of categorized food names
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_file = os.path.join(output_dir, "food_categories.json")
    with open(json_file, 'w') as f:
        json.dump(categories, f, indent=2)
    print(f"✓ Saved categorized foods (JSON) to: {json_file}")
    
    # Save as formatted text file
    text_file = os.path.join(output_dir, "food_categories.txt")
    with open(text_file, 'w') as f:
        f.write("Food Items by Nutritional Category\n")
        f.write("=" * 70 + "\n\n")
        
        for category in ['Fruit', 'Vegetable', 'Whole Grain', 'Sugary Beverage', 'Other']:
            items = sorted(categories[category])
            f.write(f"{category.upper()}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Count: {len(items)}\n\n")
            
            for item in items:
                f.write(f"  • {item}\n")
            
            f.write("\n")
    
    print(f"✓ Saved categorized foods (TXT) to: {text_file}")
    
    # Save category summary as CSV
    csv_file = os.path.join(output_dir, "food_categories_summary.csv")
    with open(csv_file, 'w') as f:
        f.write("category,food_name\n")
        
        for category in ['Fruit', 'Vegetable', 'Whole Grain', 'Sugary Beverage', 'Other']:
            for item in sorted(categories[category]):
                # Escape commas in food names
                item_escaped = item.replace(',', ';')
                f.write(f"{category},{item_escaped}\n")
    
    print(f"✓ Saved categorized foods (CSV) to: {csv_file}")


def print_summary(categories):
    """
    Print summary of categorization
    
    Args:
        categories: Dictionary of categorized food names
    """
    total = sum(len(items) for items in categories.values())
    
    print("\n" + "=" * 70)
    print("FOOD CATEGORIZATION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal unique food items: {total}\n")
    
    for category in ['Fruit', 'Vegetable', 'Whole Grain', 'Sugary Beverage', 'Other']:
        items = categories[category]
        percentage = (len(items) / total * 100) if total > 0 else 0
        print(f"{category:20s}: {len(items):3d} items ({percentage:5.1f}%)")
    
    print("\n" + "-" * 70)
    print("\nDETAILED BREAKDOWN BY CATEGORY")
    print("-" * 70)
    
    for category in ['Fruit', 'Vegetable', 'Whole Grain', 'Sugary Beverage', 'Other']:
        items = sorted(categories[category])
        print(f"\n{category.upper()} ({len(items)} items):")
        print("-" * 70)
        
        # Show first 10 items
        for item in items[:10]:
            print(f"  • {item}")
        
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
    
    print("\n" + "=" * 70)


def main():
    """Main function"""
    print("=" * 70)
    print("Food Categorization Tool")
    print("=" * 70 + "\n")
    
    # Load food names
    print("Loading food names from unique_food_names.txt...")
    food_names = load_food_names("results/unique_food_names.txt")
    print(f"✓ Loaded {len(food_names)} unique food names\n")
    
    # Categorize foods
    print("Categorizing food items...")
    categories = categorize_all_foods(food_names)
    print("✓ Categorization complete\n")
    
    # Print summary
    print_summary(categories)
    
    # Save results
    print("\nSaving results...")
    save_categorized_foods(categories, output_dir="results")
    
    print("\n✓ Food categorization complete!")


if __name__ == "__main__":
    main()
