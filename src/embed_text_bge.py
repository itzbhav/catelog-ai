"""Embed product text using BGE model - MULTIPLE FILES"""

import pandas as pd
from langchain_chroma import Chroma
from FlagEmbedding import FlagModel
import chromadb
import os

# Force to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

print("\n" + "="*70)
print("TEXT EMBEDDING WITH BGE - MULTIPLE FILES")
print(f"📁 Working from: {os.getcwd()}")
print("="*70)

# ============================================================
# LIST ALL YOUR EXCEL FILES HERE
# ============================================================
data_files = [
    'data/lighting_led_lamps.xlsx',
    'data/lighting_downlight_v2.xlsx',
    'data/lighting_track_systems.xlsx',
    'data/lighting_outdoor_expanded.xlsx',
    'data/lighting_downlight_v1.xlsx',
    'data/lighting_bollard_expanded.xlsx',
]

# ============================================================
# LOAD AND COMBINE ALL EXCEL FILES
# ============================================================
print("\n📊 Loading Excel files...")
all_dfs = []
total_products = 0

for file_path in data_files:
    if os.path.exists(file_path):
        print(f"   📂 {file_path}...")
        try:
            df = pd.read_excel(file_path)
            # Add source file as metadata for tracking
            df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)
            total_products += len(df)
            print(f"   ✅ {len(df)} products loaded")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    else:
        print(f"   ⚠️ File not found: {file_path}")

if not all_dfs:
    print("\n❌ No files loaded! Exiting...")
    exit(1)

# Combine all dataframes into one
df = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ Total products loaded: {len(df)} from {len(all_dfs)} files")
print(f"📋 Columns: {list(df.columns)}")

# ============================================================
# LOAD BGE MODEL
# ============================================================
print("\n🤖 Loading BGE model...")
bge_model = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)
print("✅ BGE model loaded")

# ============================================================
# INITIALIZE CHROMADB
# ============================================================
print("\n💾 Initializing ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db_bge")

# Delete existing collection
try:
    client.delete_collection("product_text_embeddings")
    print("⚠️ Deleted existing collection")
except:
    pass

# Create embedding function
class BGEEmbeddings:
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Create collection
text_store = Chroma(
    persist_directory="./chroma_db_bge",
    embedding_function=BGEEmbeddings(bge_model),
    collection_name="product_text_embeddings"
)
print("✅ ChromaDB collection created")

# ============================================================
# PROCESS PRODUCTS WITH UNIQUE row_id
# ============================================================
print("\n📝 Processing products...\n")

texts = []
metadatas = []
global_idx = 0  # ✅ NEW: Global counter across all files

for df_item in all_dfs:
    source_file = df_item['source_file'].iloc[0]
    file_prefix = source_file.replace('.xlsx', '').replace('lighting_', '')
    
    for local_idx, row in df_item.iterrows():
        product_name = row.get('Product Name', f'Product {global_idx}')
        brand = row.get('Brand', 'Unknown')
        category = row.get('Category', '')
        description = row.get('Description', '')
        
        # Get BOTH image URLs (try multiple column name variations)
        image_url_1 = ''
        image_url_2 = ''
        
        # Try different column names for primary image (including img1)
        for col in ['img1', 'Image_URL', 'Image', 'image_url', 'image', 'Image URL', 'URL']:
            if col in df_item.columns and pd.notna(row.get(col)):
                image_url_1 = str(row.get(col)).strip()
                break
        
        # Try different column names for secondary image (including img2, img3)
        for col in ['img2', 'img3', 'Image_URL_2', 'Image 2', 'image_url_2', 'Image2', 'Secondary Image']:
            if col in df_item.columns and pd.notna(row.get(col)):
                image_url_2 = str(row.get(col)).strip()
                break
        
        # Create text for embedding
        text = f"""Product Name: {product_name}
Brand: {brand}
Category: {category}
Description: {description}"""
        
        # ✅ FIXED: Globally unique row_id
        row_id = f"{file_prefix}_row{global_idx}"
        
        # Create metadata with BOTH image URLs and source file
        metadata = {
            'row_id': row_id,  # ✅ NOW UNIQUE ACROSS ALL FILES
            'product_name': product_name,
            'brand': brand,
            'category': category,
            'image_url': image_url_1,      # Primary image
            'image_url_2': image_url_2,    # Secondary image
            'source_file': source_file     # Track which file it came from
        }
        
        texts.append(text)
        metadatas.append(metadata)
        global_idx += 1
        
        # Progress display
        if global_idx % 100 == 0 or global_idx == len(df):
            print(f"   Processed {global_idx}/{len(df)} products...")

print(f"\n✅ Total products processed: {len(texts)}")

# ============================================================
# ADD TO CHROMADB IN BATCHES
# ============================================================
print("\n💾 Adding to ChromaDB in batches...")
batch_size = 100  # Safe batch size (under 166 limit)
total_batches = (len(texts) - 1) // batch_size + 1

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_metadatas = metadatas[i:i+batch_size]
    
    text_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)
    
    batch_num = i // batch_size + 1
    print(f"   ✅ Batch {batch_num}/{total_batches} added ({len(batch_texts)} products)")

print("\n" + "="*70)
print(f"✅ SUCCESS!")
print(f"📦 Total products stored: {len(texts)}")
print(f"📁 From {len(all_dfs)} files:")
for i, df_item in enumerate(all_dfs, 1):
    source = df_item['source_file'].iloc[0]
    count = len(df_item)
    print(f"   {i}. {source}: {count} products")
print(f"📂 Location: ./chroma_db_bge/")
print(f"🗃️ Collection: product_text_embeddings")
print("="*70 + "\n")
