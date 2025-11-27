"""Embed images using Marqo-Ecommerce-L - STORE IN CHROMADB"""
import pandas as pd
import numpy as np
import open_clip
import torch
from PIL import Image
import requests
from io import BytesIO
import chromadb

print("\n" + "="*70)
print("IMAGE EMBEDDING WITH MARQO - TO CHROMADB")
print("="*70)

# Load Excel
print("\n📊 Loading Excel file...")
df = pd.read_excel('data/data4.xlsx')
print(f"✅ Loaded {len(df)} products")
print(f"📋 Columns: {list(df.columns)}")

# Detect image columns
print("\n🔍 Detecting image URL columns...")
image_col_1 = None
image_col_2 = None

for col in ['Image_URL', 'Image', 'image_url', 'image', 'Image URL', 'URL']:
    if col in df.columns:
        image_col_1 = col
        print(f"✅ Primary image column: '{col}'")
        break

for col in ['Image_URL_2', 'Image 2', 'image_url_2', 'Image2']:
    if col in df.columns:
        image_col_2 = col
        print(f"✅ Secondary image column: '{col}'")
        break

if not image_col_1:
    print("❌ No image column found!")
    exit(1)

# Load Marqo model
print("\n🤖 Loading Marqo-Ecommerce-L model...")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
)
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-ecommerce-embeddings-L')
model.eval()
print("✅ Marqo model loaded")

# Initialize ChromaDB
print("\n💾 Initializing ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("product_image_embeddings")
    print("⚠️  Deleted existing collection")
except:
    pass

image_collection = client.create_collection(
    name="product_image_embeddings",
    metadata={"description": "Marqo-Ecommerce-L image embeddings"}
)
print("✅ ChromaDB collection created")

# Helper function
def embed_image(url):
    """Embed a single image from URL"""
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_tensor = preprocess_val(img).unsqueeze(0)
        
        with torch.no_grad():
            features = model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    except Exception as e:
        return None

# Process products
print("\n📸 Processing images...\n")
ids_list = []
embeddings_list = []
documents_list = []
metadatas_list = []
success = 0
failed = 0

for idx, row in df.iterrows():
    product_name = row.get('Product Name', f'Product {idx}')
    brand = row.get('Brand', 'Unknown')
    category = row.get('Category', '')
    
    # Get image URLs
    url_1 = row.get(image_col_1, '') if image_col_1 else ''
    url_2 = row.get(image_col_2, '') if image_col_2 else ''
    
    print(f"{idx + 1}/{len(df)}: {product_name}")
    
    # Collect valid URLs
    urls = []
    if pd.notna(url_1) and str(url_1).strip():
        urls.append(('primary', str(url_1)))
    if pd.notna(url_2) and str(url_2).strip():
        urls.append(('secondary', str(url_2)))
    
    if not urls:
        print(f"   ⚠️  No valid URLs")
        # Text fallback
        text = f"{product_name} {brand}"
        text_input = tokenizer([text])
        with torch.no_grad():
            features = model.encode_text(text_input)
            features /= features.norm(dim=-1, keepdim=True)
        embedding = features.cpu().numpy().flatten()
        
        ids_list.append(f"img_{idx}_text")
        embeddings_list.append(embedding.tolist())
        documents_list.append(f"{product_name} {brand} {category}")
        metadatas_list.append({
            'row_id': str(idx),
            'product_name': product_name,
            'brand': brand,
            'category': category,
            'image_url': '',
            'image_type': 'text_fallback'
        })
        failed += 1
        continue
    
    # Process each URL
    embedded = False
    for img_type, url in urls:
        embedding = embed_image(url)
        
        if embedding is not None:
            ids_list.append(f"img_{idx}_{img_type}")
            embeddings_list.append(embedding.tolist())
            documents_list.append(f"{product_name} {brand} {category}")
            metadatas_list.append({
                'row_id': str(idx),
                'product_name': product_name,
                'brand': brand,
                'category': category,
                'image_url': url,
                'image_type': img_type
            })
            print(f"   ✅ {img_type.capitalize()} image embedded")
            embedded = True
        else:
            print(f"   ❌ {img_type.capitalize()} image failed")
    
    if embedded:
        success += 1
    else:
        failed += 1

# Add to ChromaDB
print("\n💾 Adding to ChromaDB...")
batch_size = 10
for i in range(0, len(ids_list), batch_size):
    image_collection.add(
        ids=ids_list[i:i+batch_size],
        embeddings=embeddings_list[i:i+batch_size],
        documents=documents_list[i:i+batch_size],
        metadatas=metadatas_list[i:i+batch_size]
    )
    print(f"   ✅ Batch {i//batch_size + 1}/{(len(ids_list)-1)//batch_size + 1}")

count = image_collection.count()
print("\n" + "="*70)
print(f"✅ SUCCESS!")
print(f"📦 Products with images: {success}/{len(df)}")
print(f"❌ Failed: {failed}/{len(df)}")
print(f"💾 Total embeddings: {count}")
print(f"📂 Location: ./chroma_db/")
print("="*70 + "\n")
