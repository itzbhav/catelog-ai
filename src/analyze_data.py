"""
src/analyze_data.py - Analyze ChromaDB data

This script:
1. Connects to your existing ChromaDB databases
2. Reads metadata and documents (NOT embeddings)
3. Discovers brands, categories, wattages, keywords
4. Classifies product types (product/system/accessory)
5. Saves analysis to config/data_analysis.json

Usage:
    python src/analyze_data.py

Output:
    config/data_analysis.json - Contains discovered patterns
"""

import chromadb
import re
import json
import os
import sys
from collections import Counter
from datetime import datetime


def classify_product_type(category):
    """
    Automatically classify product type based on category name patterns.
    Works for ANY product catalog, not just lighting.
    """
    if not category:
        return 'product'
    
    category_lower = category.lower()
    
    # Pattern-based classification (generic rules)
    SYSTEM_KEYWORDS = ['system', 'rail', 'track system', 'mounting', 'infrastructure']
    ACCESSORY_KEYWORDS = ['accessory', 'connector', 'adapter', 'joint', 'fitting', 'component', 'part', 'profile']
    
    # Check if it's a system/infrastructure
    if any(keyword in category_lower for keyword in SYSTEM_KEYWORDS):
        return 'system'
    
    # Check if it's an accessory/part
    if any(keyword in category_lower for keyword in ACCESSORY_KEYWORDS):
        return 'accessory'
    
    # Default: it's a main product/fixture
    return 'product'


def analyze_chromadb():
    """
    Analyze ChromaDB to discover data patterns
    """
    
    print("\n" + "="*70)
    print("🔍 CHROMADB DATA ANALYSIS")
    print("="*70)
    
    try:
        # ============================================================
        # SETUP PATHS
        # ============================================================
        # Get project root (parent of src/)
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
        project_root = os.path.dirname(current_dir)               # BM/
        
        # Define paths
        text_db_path = os.path.join(project_root, "chroma_db_bge")
        image_db_path = os.path.join(project_root, "chroma_db")
        config_dir = os.path.join(project_root, "config")
        output_path = os.path.join(config_dir, "data_analysis.json")
        
        print(f"\n📂 Project Structure:")
        print(f"   Project root: {project_root}")
        print(f"   Text DB: {text_db_path}")
        print(f"   Image DB: {image_db_path}")
        print(f"   Output: {output_path}")
        
        # ============================================================
        # STEP 1: Connect to ChromaDB
        # ============================================================
        print("\n📂 Connecting to ChromaDB...")
        
        text_client = chromadb.PersistentClient(path=text_db_path)
        image_client = chromadb.PersistentClient(path=image_db_path)
        
        text_collection = text_client.get_collection("product_text_embeddings")
        image_collection = image_client.get_collection("product_image_embeddings")
        
        print("   ✅ Connected successfully")
        
        # ============================================================
        # STEP 2: Get ALL data (metadata + documents only)
        # ============================================================
        print("\n📊 Fetching product data...")
        
        text_data = text_collection.get(
            include=['metadatas', 'documents']  # NOT including embeddings!
        )
        
        image_data = image_collection.get(
            include=['metadatas']
        )
        
        total_text = len(text_data['metadatas'])
        total_image = len(image_data['metadatas'])
        
        print(f"   ✅ Text embeddings: {total_text} products")
        print(f"   ✅ Image embeddings: {total_image} products")
        
        # ============================================================
        # STEP 3: Analyze Metadata & Documents
        # ============================================================
        print("\n🔬 Analyzing patterns...")
        
        # Initialize counters
        brands = Counter()
        categories = Counter()
        wattages = Counter()
        voltages = Counter()
        colors = Counter()
        keywords = Counter()
        row_ids = set()
        
        # Analyze text collection
        for i, metadata in enumerate(text_data['metadatas']):
            # Extract row_id for sync checking
            row_id = metadata.get('row_id', f'text_{i}')
            row_ids.add(row_id)
            
            # Extract brand
            brand = metadata.get('brand', '').strip()
            if brand and brand.lower() not in ['unknown', '', 'n/a']:
                brands[brand.lower()] += 1
            
            # Extract category
            category = metadata.get('category', '').strip()
            if category and category.lower() not in ['unknown', '', 'n/a']:
                categories[category] += 1
            
            # Extract document text
            document = text_data['documents'][i] if i < len(text_data['documents']) else ''
            doc_lower = document.lower()
            
            # Extract wattages (12w, 15w, etc.)
            wattage_matches = re.findall(r'\b(\d+)\s*w\b', doc_lower)
            for w in wattage_matches:
                if 1 <= int(w) <= 200:  # Reasonable wattage range
                    wattages[w] += 1
            
            # Extract voltages (220v, 230v, etc.)
            voltage_matches = re.findall(r'\b(\d+)\s*v\b', doc_lower)
            for v in voltage_matches:
                if 100 <= int(v) <= 300:  # Reasonable voltage range
                    voltages[v] += 1
            
            # Extract color temperatures
            if 'warm white' in doc_lower or 'warm' in doc_lower:
                colors['warm_white'] += 1
            if 'cool white' in doc_lower or 'cool' in doc_lower or 'daylight' in doc_lower:
                colors['cool_white'] += 1
            if 'neutral' in doc_lower or 'natural' in doc_lower:
                colors['neutral_white'] += 1
            
            # Extract common keywords (4+ letters, alphabetic)
            words = re.findall(r'\b[a-z]{4,}\b', doc_lower)
            keywords.update(words)
        
        # ============================================================
        # STEP 3.5: Analyze Product Types (NEW!)
        # ============================================================
        print("\n🏗️  Analyzing product types...")
        
        product_types = Counter()
        category_to_type = {}
        
        for category, count in categories.items():
            ptype = classify_product_type(category)
            product_types[ptype] += count
            category_to_type[category] = ptype
        
        print(f"   ✅ Product types classified: {len(category_to_type)} categories")
        
        # Check synchronization (products in both text and image)
        image_row_ids = set()
        for metadata in image_data['metadatas']:
            row_id = metadata.get('row_id', '')
            if row_id:
                image_row_ids.add(row_id)
        
        synchronized_ids = row_ids & image_row_ids
        sync_percentage = (len(synchronized_ids) / len(row_ids) * 100) if row_ids else 0
        
        # ============================================================
        # STEP 4: Generate Analysis Report
        # ============================================================
        print("\n" + "="*70)
        print("📊 ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\n📦 TOTAL PRODUCTS:")
        print(f"   Text embeddings: {total_text}")
        print(f"   Image embeddings: {total_image}")
        print(f"   Synchronized: {len(synchronized_ids)} ({sync_percentage:.1f}%)")
        
        print(f"\n🏷️  BRANDS DISCOVERED ({len(brands)}):")
        for brand, count in brands.most_common(15):
            percentage = (count / total_text * 100)
            print(f"   {brand.title():20s} : {count:4d} products ({percentage:5.1f}%)")
        
        print(f"\n📂 CATEGORIES DISCOVERED ({len(categories)}):")
        for category, count in categories.most_common():
            percentage = (count / total_text * 100)
            ptype = category_to_type.get(category, 'product')
            print(f"   {category:30s} : {count:4d} products ({percentage:5.1f}%) [{ptype}]")
        
        print(f"\n🏗️  PRODUCT TYPES:")
        for ptype, count in product_types.most_common():
            percentage = (count / total_text * 100)
            print(f"   {ptype.title():20s} : {count:4d} products ({percentage:5.1f}%)")
        
        print(f"\n⚡ WATTAGES DISCOVERED ({len(wattages)}):")
        wattage_list = sorted([(int(w), count) for w, count in wattages.items()])
        for i in range(0, min(len(wattage_list), 20), 5):  # Print 5 per line
            line_items = wattage_list[i:i+5]
            line = "   " + "  ".join([f"{w}W({c})" for w, c in line_items])
            print(line)
        
        print(f"\n🔌 VOLTAGES DISCOVERED ({len(voltages)}):")
        for voltage, count in voltages.most_common(10):
            print(f"   {voltage}V: {count} products")
        
        print(f"\n🎨 COLOR TEMPERATURES:")
        for color, count in colors.most_common():
            percentage = (count / total_text * 100)
            color_name = color.replace('_', ' ').title()
            print(f"   {color_name:20s} : {count:4d} products ({percentage:5.1f}%)")
        
        print(f"\n🔤 TOP KEYWORDS (appearing 20+ times):")
        # Filter out common words
        stopwords = {'with', 'bulb', 'light', 'lighting', 'lamp', 'product', 
                    'this', 'that', 'from', 'have', 'more', 'your', 'about'}
        top_keywords = [(w, c) for w, c in keywords.most_common(50) 
                       if c >= 20 and w not in stopwords]
        
        for i in range(0, min(30, len(top_keywords)), 3):  # Print 3 per line
            line_items = top_keywords[i:i+3]
            line = "   " + "  ".join([f"{w}({c})" for w, c in line_items])
            print(line)
        
        # ============================================================
        # STEP 5: Build and Save JSON Report
        # ============================================================
        print("\n💾 Saving analysis report...")
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'summary': {
                'total_text_products': total_text,
                'total_image_products': total_image,
                'synchronized_products': len(synchronized_ids),
                'sync_percentage': round(sync_percentage, 2)
            },
            'brands': {
                'count': len(brands),
                'list': [b for b in brands.keys()],
                'top_10': [{'name': b, 'count': c} for b, c in brands.most_common(10)]
            },
            'categories': {
                'count': len(categories),
                'distribution': dict(categories)
            },
            'product_types': {
                'distribution': dict(product_types),
                'category_mapping': category_to_type
            },
            'attributes': {
                'wattages': {
                    'count': len(wattages),
                    'values': sorted([int(w) for w in wattages.keys()]),
                    'distribution': {w: c for w, c in wattages.items()}
                },
                'voltages': {
                    'count': len(voltages),
                    'values': sorted([int(v) for v in voltages.keys()]),
                    'distribution': {v: c for v, c in voltages.items()}
                },
                'colors': dict(colors)
            },
            'keywords': {
                'top_50': [{'word': w, 'count': c} for w, c in keywords.most_common(50) 
                          if w not in stopwords]
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Report saved to: {output_path}")
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("\nNext step: Run 'python src/config_generator.py'")
        print("="*70 + "\n")
        
        return report
        
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__": 
    analyze_chromadb()
