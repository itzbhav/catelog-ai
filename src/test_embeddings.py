"""Embed images using Marqo-Ecommerce model - DETAILED DEBUG VERSION"""
import pandas as pd
import open_clip
import torch
from PIL import Image
import requests
from io import BytesIO
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_marqo_ecommerce_model():
    print("Loading Marqo-Ecommerce-L model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-ecommerce-embeddings-L')
    return model, preprocess, tokenizer

def download_image(url):
    """Download image with detailed error reporting"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, timeout=15, headers=headers, verify=False)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image, None
    except Exception as e:
        return None, str(e)

def embed_images_from_excel(excel_path):
    print("Loading Excel file...")
    df = pd.read_excel(excel_path)
    
    print(f"\n{'='*70}")
    print(f"📊 Excel Info:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Total rows: {len(df)}")
    print(f"{'='*70}\n")
    
    model, preprocess, tokenizer = load_marqo_ecommerce_model()
    model.eval()
    
    image_embeddings = {}
    text_embeddings = {}
    metadata = {}
    
    success_count = 0
    fail_count = 0
    no_url_count = 0
    
    for idx, row in df.iterrows():
        # Try multiple column name variations
        product_name = (str(row.get('product_name', '')) or 
                       str(row.get('Product Name', '')) or 
                       str(row.get('name', '')) or 
                       str(row.get('Name', '')) or 
                       'Unknown Product')
        
        image_url = (str(row.get('image_url', '')) or 
                    str(row.get('Image_URL', '')) or 
                    str(row.get('image', '')) or 
                    str(row.get('Image', '')) or 
                    str(row.get('img_url', '')) or '')
        
        print(f"\n{'─'*70}")
        print(f"Product {idx+1}/{len(df)}: {product_name[:50]}")
        print(f"{'─'*70}")
        
        # Process IMAGE
        if image_url and image_url != 'nan' and image_url != '' and image_url != 'None':
            print(f"📸 Image URL: {image_url[:80]}")
            print(f"   Downloading...")
            
            image, error = download_image(image_url)
            
            if image:
                try:
                    print(f"   ✅ Downloaded successfully (Size: {image.size})")
                    
                    # Preprocess and encode
                    image_input = preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        img_features = model.encode_image(image_input)
                        img_features /= img_features.norm(dim=-1, keepdim=True)
                    
                    image_embeddings[str(idx)] = img_features.numpy()
                    print(f"   ✅ Embedded successfully (Shape: {img_features.shape})")
                    success_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Encoding error: {e}")
                    fail_count += 1
            else:
                print(f"   ❌ Download failed: {error[:100]}")
                fail_count += 1
        else:
            print(f"   ⚠️  No valid image URL found")
            print(f"   URL value: '{image_url}'")
            no_url_count += 1
            fail_count += 1
        
        # Process TEXT (always works)
        description = str(row.get('description', '') or row.get('Description', ''))
        brand = str(row.get('brand', '') or row.get('Brand', ''))
        
        text = f"{product_name} {brand} {description}".strip()
        text_input = tokenizer([text])
        
        with torch.no_grad():
            txt_features = model.encode_text(text_input)
            txt_features /= txt_features.norm(dim=-1, keepdim=True)
        
        text_embeddings[str(idx)] = txt_features.numpy()
        print(f"   ✅ Text embedded")
        
        # Save metadata
        metadata[str(idx)] = {
            'product_name': product_name,
            'brand': brand,
            'category': str(row.get('category', '') or row.get('Category', '')),
            'image_url': image_url,
            'description': description[:100]
        }
    
    # Save everything
    print(f"\n{'='*70}")
    print("💾 Saving embeddings...")
    
    with open('marqo_ecommerce_embeddings.pkl', 'wb') as f:
        pickle.dump({
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'metadata': metadata
        }, f)
    
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY:")
    print(f"{'='*70}")
    print(f"✅ Images successfully embedded: {success_count}/{len(df)}")
    print(f"❌ Images failed to embed: {fail_count}/{len(df)}")
    print(f"⚠️  Products without URLs: {no_url_count}/{len(df)}")
    print(f"✅ Text embeddings created: {len(text_embeddings)}/{len(df)}")
    print(f"💾 Saved to: marqo_ecommerce_embeddings.pkl")
    print(f"{'='*70}\n")
    
    if success_count == 0:
        print("⚠️  WARNING: No images were embedded!")
        print("   Please check:")
        print("   1. Are image URLs valid?")
        print("   2. Can URLs be accessed from your network?")
        print("   3. Do URLs point to actual images?")

if __name__ == "__main__":
    embed_images_from_excel('./data/data4.xlsx')
