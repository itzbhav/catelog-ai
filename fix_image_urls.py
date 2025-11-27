"""Fix missing image URLs in ChromaDB"""
import pandas as pd
import chromadb
from langchain_chroma import Chroma
from FlagEmbedding import FlagModel

print("Loading Excel...")
df = pd.read_excel('data/data5.xlsx')  # Use data5.xlsx

print(f"Loaded {len(df)} products")
print(f"Columns: {list(df.columns)}")

# Check image columns
print("\nImage columns found:")
for col in df.columns:
    if 'img' in col.lower() or 'image' in col.lower():
        print(f"  - {col}")

# Load BGE model
print("\nLoading BGE model...")
bge_model = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db_bge")

# Delete old collection
try:
    client.delete_collection("product_text_embeddings")
    print("Deleted old collection")
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

print("\nProcessing products...")
texts = []
metadatas = []

for idx, row in df.iterrows():
    product_name = row.get('Product Name', f'Product {idx}')
    brand = row.get('Brand', 'Mascon')
    category = row.get('Category', '')
    description = row.get('Description', '')
    
    # Get image URLs - CHECK YOUR EXCEL COLUMN NAMES!
    image_url_1 = ''
    image_url_2 = ''
    
    # Try to find image columns (adjust these based on your Excel)
    if 'img1' in df.columns:
        image_url_1 = str(row.get('img1', '')) if pd.notna(row.get('img1')) else ''
    if 'img2' in df.columns:
        image_url_2 = str(row.get('img2', '')) if pd.notna(row.get('img2')) else ''
    
    # Create text
    text = f"Product: {product_name}\nBrand: {brand}\nCategory: {category}\n{description}"
    
    # Create metadata
    metadata = {
        'row_id': str(idx),
        'product_name': product_name,
        'brand': brand,
        'category': category,
        'image_url': image_url_1,
        'image_url_2': image_url_2,
    }
    
    texts.append(text)
    metadatas.append(metadata)
    
    print(f"{idx + 1}/{len(df)}: {product_name}")
    if image_url_1:
        print(f"  img1: {image_url_1[:60]}...")
    if image_url_2:
        print(f"  img2: {image_url_2[:60]}...")

# Add to ChromaDB
print("\nAdding to ChromaDB...")
text_store.add_texts(texts=texts, metadatas=metadatas)

print(f"\n✅ Done! Updated {len(texts)} products")
