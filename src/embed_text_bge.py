"""Embed product text using BGE model - WITH MULTIPLE IMAGE URLS"""
import pandas as pd
from langchain_chroma import Chroma
from FlagEmbedding import FlagModel
import chromadb

print("\n" + "="*70)
print("TEXT EMBEDDING WITH BGE")
print("="*70)

# Load Excel
print("\n📊 Loading Excel file...")
df = pd.read_excel('C:/Users/bhava/Desktop/E-constru/files/BM/data/data7.xlsx')
print(f"✅ Loaded {len(df)} products")
print(f"📋 Columns: {list(df.columns)}")

# Load BGE model
print("\n🤖 Loading BGE model...")
bge_model = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)
print("✅ BGE model loaded")

# Initialize ChromaDB
print("\n💾 Initializing ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db_bge")

# Delete existing collection
try:
    client.delete_collection("product_text_embeddings")
    print("⚠️  Deleted existing collection")
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

# Process products
print("\n📝 Processing products...\n")
texts = []
metadatas = []

for idx, row in df.iterrows():
    product_name = row.get('Product Name', f'Product {idx}')
    brand = row.get('Brand', 'Unknown')
    category = row.get('Category', '')
    description = row.get('Description', '')
    
    # Get BOTH image URLs (try multiple column name variations)
    image_url_1 = ''
    image_url_2 = ''
    
    # Try different column names for primary image
    for col in ['Image_URL', 'Image', 'image_url', 'image', 'Image URL', 'URL']:
        if col in df.columns and pd.notna(row.get(col)):
            image_url_1 = str(row.get(col))
            break
    
    # Try different column names for secondary image
    for col in ['Image_URL_2', 'Image 2', 'image_url_2', 'Image2', 'Secondary Image']:
        if col in df.columns and pd.notna(row.get(col)):
            image_url_2 = str(row.get(col))
            break
    
    # Create text for embedding
    text = f"""
Product Name: {product_name}
Brand: {brand}
Category: {category}
Description: {description}
"""
    
    # Create metadata with BOTH image URLs
    metadata = {
        'row_id': str(idx),
        'product_name': product_name,
        'brand': brand,
        'category': category,
        'image_url': image_url_1,      # Primary image
        'image_url_2': image_url_2,    # Secondary image
    }
    
    texts.append(text)
    metadatas.append(metadata)
    
    print(f"{idx + 1}/{len(df)}: {product_name}")
    if image_url_1:
        print(f"   📸 Primary: {image_url_1[:50]}...")
    if image_url_2:
        print(f"   📸 Secondary: {image_url_2[:50]}...")

# Add to ChromaDB
print("\n💾 Adding to ChromaDB...")
text_store.add_texts(texts=texts, metadatas=metadatas)

print("\n" + "="*70)
print(f"✅ SUCCESS!")
print(f"📦 Stored: {len(texts)} products")
print(f"📂 Location: ./chroma_db_bge/")
print(f"🗃️ Collection: product_text_embeddings")
print("="*70 + "\n")
