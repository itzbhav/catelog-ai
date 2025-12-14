"""
Image Embedding Script with OpenCLIP (Marqo-Ecommerce-L Model)
Processes multiple Excel files with multimodal embeddings
"""

import pandas as pd
import numpy as np
import chromadb
import open_clip
import torch
from PIL import Image
import requests
from io import BytesIO
import os

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILES = [
    '../data/lighting_led_lamps.xlsx',
    '../data/lighting_downlight_v2.xlsx',
    '../data/lighting_track_systems.xlsx',
    '../data/lighting_outdoor_expanded.xlsx',
    '../data/lighting_downlight_v1.xlsx',
    '../data/lighting_bollard_expanded.xlsx'
]

CHROMA_PATH = "../chroma_db"
COLLECTION_NAME = "product_image_embeddings"
BATCH_SIZE = 50

# ============================================================
# SETUP
# ============================================================
print("\n" + "="*70)
print("IMAGE EMBEDDING WITH MARQO - MULTIPLE FILES")
print(f"📁 Working from: {os.getcwd()}")
print("="*70)

# ============================================================
# LOAD AND COMBINE ALL EXCEL FILES
# ============================================================
print("\n📊 Loading Excel files...")
all_dfs = []
total_products = 0

for file_path in DATA_FILES:
    if os.path.exists(file_path):
        print(f"   📂 {file_path}...")
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # ✅ FIX: Check if Row 0 has description text and skip it
            if 'Product Name' in df.columns:
                first_val = str(df.iloc[0]['Product Name'])
                if len(first_val) > 100 or 'official name' in first_val.lower():
                    print(f"   ⚠️ Detected description row, skipping Row 0")
                    df = df[1:].reset_index(drop=True)  # Skip description row
            
            # ✅ FIX: For files with generic columns, use Row 0 as headers
            elif 'product' in df.columns and 'attribute_1' in df.columns:
                print(f"   ⚠️ Detected attribute format, using Row 0 as headers")
                new_columns = df.iloc[0].tolist()
                df = df[1:].reset_index(drop=True)
                df.columns = new_columns
            
            # Add source file for tracking
            df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)
            total_products += len(df)
            print(f"   ✅ {len(df)} products loaded")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ⚠️ File not found: {file_path}")

if not all_dfs:
    print("\n❌ No files loaded! Exiting...")
    exit(1)

# Combine all dataframes
df = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ Total products: {len(df)} from {len(all_dfs)} files")
print(f"📋 Sample columns: {list(df.columns)[:15]}")

# ============================================================
# DETECT IMAGE COLUMNS
# ============================================================
print("\n🔍 Detecting image URL columns...")

# Find primary image column
image_col = None
for col in ['img1', 'Image_URL', 'image_url', 'Image']:
    if col in df.columns:
        image_col = col
        print(f"✅ Primary image column: '{col}'")
        break

if image_col is None:
    # Try flexible matching
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['img', 'image', 'url', 'photo']):
            if not any(x in col_lower for x in ['2', '3', '4', 'secondary', 'alt']):
                image_col = col
                print(f"✅ Primary image column: '{col}'")
                break

if image_col is None:
    print("❌ No image column found!")
    print(f"   Available columns: {list(df.columns)[:20]}")
    exit(1)

# Find secondary image column
image_col_2 = None
for col in ['img2', 'img3']:
    if col in df.columns:
        image_col_2 = col
        print(f"✅ Secondary image column: '{col}'")
        break

# ============================================================
# LOAD MARQO MODEL
# ============================================================
print("\n🤖 Loading Marqo-Ecommerce-L model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
)
model.eval()
print("✅ Marqo model loaded")

# ============================================================
# INITIALIZE CHROMADB
# ============================================================
print("\n💾 Initializing ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Delete existing collection if it exists
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"   🗑️ Deleted existing collection")
except:
    pass

# Create new collection
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print("✅ ChromaDB collection created")

# ============================================================
# PROCESS IMAGES WITH UNIQUE row_id
# ============================================================
print("\n📸 Processing images...\n")

texts = []
embeddings = []
metadatas = []
ids = []
processed_count = 0
failed_count = 0
skipped_count = 0
global_idx = 0  # ✅ NEW: Global counter

for df_item in all_dfs:
    source_file = df_item['source_file'].iloc[0]
    file_prefix = source_file.replace('.xlsx', '').replace('lighting_', '')
    
    for local_idx, row in df_item.iterrows():
        # Get product info with EXACT same keys as text embeddings
        product_name = str(row.get('Product Name', f'Product {global_idx}')).strip()
        brand = str(row.get('Brand', 'Unknown')).strip()
        category = str(row.get('Category', '')).strip()
        
        # ✅ FIX: Skip description rows that might have slipped through
        if len(product_name) > 100 or 'official name' in product_name.lower():
            skipped_count += 1
            global_idx += 1
            continue
        
        # Get image URLs
        image_url_1 = ''
        image_url_2 = ''
        
        # Primary image
        if pd.notna(row.get(image_col)):
            url = str(row.get(image_col)).strip()
            # ✅ FIX: Skip placeholder values
            if url and url.startswith('http') and 'has the image link' not in url.lower():
                image_url_1 = url
        
        # Secondary image
        if image_col_2 and pd.notna(row.get(image_col_2)):
            url = str(row.get(image_col_2)).strip()
            if url and url.startswith('http'):
                image_url_2 = url
        
        # Skip if no valid image URL
        if not image_url_1:
            skipped_count += 1
            global_idx += 1
            continue
        
        # Skip invalid product names
        if product_name in ['nan', 'None', 'Unknown', ''] or len(product_name) < 2:
            skipped_count += 1
            global_idx += 1
            continue
        
        # Progress indicator
        if (processed_count + failed_count + 1) % 50 == 0:
            print(f"Processing {processed_count + failed_count + 1}/{len(df)}: {product_name[:30]}")
        
        try:
            # Download and process image
            response = requests.get(image_url_1, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Preprocess and generate embedding
            image_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                image_feat = model.encode_image(image_tensor)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
            
            embedding = image_feat[0].tolist()
            
            # ✅ FIXED: Globally unique row_id (matches text embeddings)
            row_id = f"{file_prefix}_row{global_idx}"
            
            # Create metadata with EXACT SAME KEYS as text embeddings
            metadata = {
                'row_id': row_id,  # ✅ NOW MATCHES TEXT EMBEDDINGS
                'product_name': product_name,
                'brand': brand,
                'category': category,
                'image_url': image_url_1,
                'image_url_2': image_url_2,
                'source_file': source_file
            }
            
            # Store for batch insertion
            texts.append(f"Product: {product_name} by {brand}")
            embeddings.append(embedding)
            metadatas.append(metadata)
            ids.append(f"img_{global_idx}")
            
            processed_count += 1
            
        except requests.exceptions.RequestException as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"   ⚠️ Network error for {product_name}: {str(e)[:50]}")
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"   ❌ Failed to process {product_name}: {str(e)[:50]}")
        
        global_idx += 1

print(f"\n✅ Processed: {processed_count}, Failed: {failed_count}, Skipped: {skipped_count}")

# ============================================================
# ADD TO CHROMADB IN BATCHES
# ============================================================
print(f"\n💾 Adding to ChromaDB...")
total_batches = (len(embeddings) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(embeddings), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_embeddings = embeddings[i:i+BATCH_SIZE]
    batch_metadatas = metadatas[i:i+BATCH_SIZE]
    batch_ids = ids[i:i+BATCH_SIZE]
    
    collection.add(
        documents=batch_texts,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas,
        ids=batch_ids
    )
    
    batch_num = (i // BATCH_SIZE) + 1
    if batch_num % 10 == 0 or batch_num == total_batches:
        print(f"   ✅ Batch {batch_num}/{total_batches}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("✅ SUCCESS!")
print(f"📦 Products with images: {processed_count}/{len(df)}")
print(f"❌ Failed: {failed_count}/{len(df)}")
print(f"⏭️ Skipped (no image): {skipped_count}/{len(df)}")
print(f"💾 Total embeddings: {len(embeddings)}")
print(f"📁 From {len(all_dfs)} files:")
for i, df_item in enumerate(all_dfs, 1):
    filename = df_item['source_file'].iloc[0]
    print(f"   {i}. {filename}: {len(df_item)} products")
print(f"📂 Location: {CHROMA_PATH}/")
print("="*70 + "\n")

# ============================================================
# VERIFY
# ============================================================
print("🔍 Verifying embeddings...")
try:
    if len(embeddings) > 0:
        test_results = collection.query(
            query_embeddings=[embeddings[0]],
            n_results=min(3, len(embeddings)),
            include=['metadatas']
        )
        print(f"✅ Verification successful!")
        print(f"   Sample metadata keys: {list(test_results['metadatas'][0][0].keys())}")
        print(f"   Sample product: {test_results['metadatas'][0][0].get('product_name', 'N/A')}")
        print(f"   Sample row_id: {test_results['metadatas'][0][0].get('row_id', 'N/A')}")
        print(f"   Sample image URL: {test_results['metadatas'][0][0].get('image_url', 'N/A')[:60]}...")
    else:
        print(f"⚠️ No embeddings to verify")
except Exception as e:
    print(f"⚠️ Verification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Image embedding complete! Run analyze_data.py next.")
