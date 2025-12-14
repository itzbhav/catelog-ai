"""
src/config_generator.py - Generate search configuration from data analysis

This script:
1. Reads config/data_analysis.json (created by analyze_data.py)
2. Generates category detection patterns (PRODUCT TYPE AWARE)
3. Creates attribute extraction rules
4. Defines scoring weights
5. Saves to config/search_config.json

Usage:
    python src/config_generator.py

Input:
    config/data_analysis.json (must exist - run analyze_data.py first)

Output:
    config/search_config.json - Configuration for app.py
"""

import json
import re
import os
from datetime import datetime


def generate_category_patterns(categories, product_types):
    """
    Generate search patterns for each category with PRODUCT TYPE AWARENESS.
    Prevents overlapping patterns between Track System, Track Light, Track Accessory.
    """
    patterns = {}
    
    for category in categories.keys():
        cat_lower = category.lower()
        product_type = product_types.get(category, 'product')
        
        # Start with exact category name
        pattern_list = [cat_lower]
        
        # Split category into words
        words = cat_lower.split()
        
        # TYPE-SPECIFIC PATTERN GENERATION (prevents overlap!)
        if product_type == 'system':
            # Systems: Infrastructure/mounting related
            if words:
                base_word = words[0]
                pattern_list.extend([
                    f"{base_word} system",
                    f"{base_word} rail",
                    f"{base_word} mounting",
                    "rail system",
                    "mounting system"
                ])
        
        elif product_type == 'accessory':
            # Accessories: Connectors/adapters/parts
            if words:
                base_word = words[0]
                pattern_list.extend([
                    f"{base_word} connector",
                    f"{base_word} adapter",
                    f"{base_word} accessory",
                    f"{base_word} fitting",
                    "connector",
                    "adapter",
                    "fitting",
                    "joint"
                ])
        
        else:
            # Products (main fixtures): Use category-specific patterns
            pattern_list.extend(words)
            
            # Add specific variations based on category keywords
            if 'bulb' in cat_lower or 'lamp' in cat_lower:
                pattern_list.extend(['bulb', 'bulbs', 'lamp', 'lamps'])
            
            if 'down' in cat_lower and 'light' in cat_lower:
                pattern_list.extend(['downlight', 'down light', 'recessed light'])
            
            if 'track' in cat_lower and 'light' in cat_lower:
                # ONLY for Track Light (product), NOT Track System or Track Accessory
                pattern_list.extend(['track light', 'track fixture', 'track spotlight'])
            
            if 'panel' in cat_lower:
                pattern_list.extend(['panel', 'panel light', 'ceiling panel'])
            
            if 'outdoor' in cat_lower or 'bollard' in cat_lower:
                pattern_list.extend(['outdoor', 'bollard', 'garden', 'pathway'])
            
            if 'surface' in cat_lower and 'light' in cat_lower:
                pattern_list.extend(['surface light', 'surface mounted'])
            
            if 'spot' in cat_lower:
                pattern_list.extend(['spot', 'spotlight'])
        
        # Remove duplicates and sort
        patterns[category] = sorted(list(set(pattern_list)))
    
    return patterns


def generate_attribute_patterns():
    """
    Define regex patterns for attribute extraction
    """
    return {
        'wattage': r'\b(\d+)\s*w\b',           # Matches: "12w", "12 w", "12W"
        'voltage': r'\b(\d+)\s*v\b',           # Matches: "220v", "230 V"
        'ip_rating': r'ip\s*(\d+)',            # Matches: "IP65", "ip 44"
        'lumens': r'(\d+)\s*(?:lm|lumens?)',   # Matches: "800 lumens", "1200lm"
        'color_temp_kelvin': r'(\d{4})k',      # Matches: "3000k", "6500K"
        'color_temp': {
            'warm_white': [
                'warm white', 'warm', 'ww', 'warm tone',
                '2700k', '3000k', '3200k'
            ],
            'cool_white': [
                'cool white', 'cool', 'cw', 'cool tone', 'daylight',
                '6000k', '6500k', 'cool daylight'
            ],
            'neutral_white': [
                'neutral', 'neutral white', 'natural', 'natural white',
                '4000k', '4500k', 'nw'
            ]
        }
    }


def generate_scoring_weights():
    """
    Define scoring weights for hybrid search
    These can be tuned based on testing
    """
    return {
        # Vector similarity weights
        'text_vector_weight': 0.4,        # 40% weight to text embedding similarity
        'image_vector_weight': 0.3,       # 30% weight to image embedding similarity
        
        # Attribute matching bonuses (points to add)
        'keyword_match_per_word': 10,     # +10 points per matching keyword
        'wattage_exact_match': 15,        # +15 if wattage matches exactly
        'voltage_exact_match': 10,        # +10 if voltage matches
        'color_temp_match': 10,           # +10 if color temperature matches
        'brand_exact_match': 20,          # +20 if brand matches exactly
        'category_match': 5,              # +5 if category matches
        'ip_rating_match': 8,             # +8 if IP rating matches
        
        # Synchronization bonus
        'text_and_image_sync': 20,        # +20 if product found in BOTH text and image search
        
        # Filtering thresholds
        'min_score_threshold': 0,         # Minimum score to include result
        'diversity_boost': 5              # +5 for diverse brand representation
    }


def generate_search_config():
    """
    Main function to generate complete search configuration
    """
    
    print("\n" + "="*70)
    print("⚙️  SEARCH CONFIGURATION GENERATOR")
    print("="*70)
    
    try:
        # ============================================================
        # SETUP PATHS
        # ============================================================
        # Get project root (parent of src/)
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
        project_root = os.path.dirname(current_dir)               # BM/
        
        config_dir = os.path.join(project_root, "config")
        analysis_path = os.path.join(config_dir, "data_analysis.json")
        output_path = os.path.join(config_dir, "search_config.json")
        
        print(f"\n📂 Project Structure:")
        print(f"   Project root: {project_root}")
        print(f"   Input: {analysis_path}")
        print(f"   Output: {output_path}")
        
        # ============================================================
        # STEP 1: Load Analysis Data
        # ============================================================
        print("\n📂 Loading data analysis...")
        
        try:
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
            print("   ✅ Loaded data_analysis.json")
        except FileNotFoundError:
            print("   ❌ ERROR: data_analysis.json not found!")
            print(f"   Expected at: {analysis_path}")
            print("   Please run 'python src/analyze_data.py' first")
            return None
        
        # ============================================================
        # STEP 2: Extract Key Information
        # ============================================================
        print("\n🔍 Extracting patterns...")
        
        brands = analysis['brands']['list']
        categories = analysis['categories']['distribution']
        wattages = analysis['attributes']['wattages']['values']
        voltages = analysis['attributes']['voltages']['values'] if 'voltages' in analysis['attributes'] else []
        
        # ✅ NEW: Load product types
        product_types = analysis.get('product_types', {}).get('category_mapping', {})
        
        print(f"   ✅ Found {len(brands)} brands")
        print(f"   ✅ Found {len(categories)} categories")
        print(f"   ✅ Found {len(product_types)} product type mappings")
        print(f"   ✅ Found {len(wattages)} wattage values")
        print(f"   ✅ Found {len(voltages)} voltage values")
        
        # ============================================================
        # STEP 3: Generate Configuration Components
        # ============================================================
        print("\n🔧 Generating configuration...")
        
        # ✅ NEW: Pass product_types to pattern generator
        category_patterns = generate_category_patterns(categories, product_types)
        print(f"   ✅ Generated patterns for {len(category_patterns)} categories")
        
        attribute_patterns = generate_attribute_patterns()
        print(f"   ✅ Generated attribute extraction patterns")
        
        scoring_weights = generate_scoring_weights()
        print(f"   ✅ Generated scoring weights")
        
        # ============================================================
        # STEP 4: Build Final Configuration
        # ============================================================
        config = {
            'config_version': '2.0',  # Updated version
            'generated_at': datetime.now().isoformat(),
            'source_data': {
                'total_products': analysis['summary']['total_text_products'],
                'analysis_date': analysis['analysis_date']
            },
            
            # Discovered data
            'brands': {
                'list': brands,
                'count': len(brands)
            },
            'categories': {
                'patterns': category_patterns,
                'count': len(category_patterns)
            },
            'product_types': product_types,  # ✅ NEW: Include product type mappings
            'known_attributes': {
                'wattages': wattages,
                'voltages': voltages
            },
            
            # Search rules
            'attribute_patterns': attribute_patterns,
            'scoring_weights': scoring_weights,
            
            # Optional: Query expansion synonyms
            'query_synonyms': {
                'bright': ['bright', 'high brightness', 'high lumen', 'powerful'],
                'cheap': ['cheap', 'affordable', 'budget', 'economical', 'low cost'],
                'efficient': ['efficient', 'energy saving', 'eco', 'star rated'],
                'decorative': ['decorative', 'aesthetic', 'ornamental', 'designer']
            }
        }
        
        # ============================================================
        # STEP 5: Save Configuration
        # ============================================================
        print("\n💾 Saving configuration...")
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ Configuration saved to: {output_path}")
        
        # ============================================================
        # STEP 6: Display Summary
        # ============================================================
        print("\n" + "="*70)
        print("📊 CONFIGURATION SUMMARY")
        print("="*70)
        
        print(f"\n🏷️  BRANDS ({len(brands)}):")
        print(f"   {', '.join(brands[:10])}...")
        if len(brands) > 10:
            print(f"   ... and {len(brands) - 10} more")
        
        print(f"\n📂 CATEGORIES ({len(category_patterns)}):")
        for i, (category, patterns) in enumerate(list(category_patterns.items())[:5]):
            ptype = product_types.get(category, 'product')
            print(f"   {category} [{ptype}]:")
            print(f"      Patterns: {', '.join(patterns[:5])}...")
        if len(category_patterns) > 5:
            print(f"   ... and {len(category_patterns) - 5} more")
        
        print(f"\n🏗️  PRODUCT TYPES:")
        type_counts = {}
        for ptype in product_types.values():
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        for ptype, count in type_counts.items():
            print(f"   {ptype.title()}: {count} categories")
        
        print(f"\n⚡ SCORING WEIGHTS:")
        for key, value in scoring_weights.items():
            if 'weight' in key:
                print(f"   {key:30s} : {value:.1f}")
            else:
                print(f"   {key:30s} : +{value} points")
        
        print("\n" + "="*70)
        print("✅ CONFIGURATION GENERATED SUCCESSFULLY!")
        print("="*70)
        print("\nNext step: Update app.py and restart the server")
        print("="*70 + "\n")
        
        return config
        
    except Exception as e:
        print(f"\n❌ ERROR during config generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    generate_search_config()
