"""Migrate Marqo image embeddings from pickle to ChromaDB"""
import pickle
import chromadb
from chromadb.config import Settings
import numpy as np

print("="*60)
print("MIGRATING MARQO EMBEDDINGS TO CHROMADB")
print("="*60)

# Step 1: Load existing Marqo embeddings from pickle
print("\n1️⃣ Loading Marqo embeddings from pickle file...")
with open('marqo_ecommerce_embeddings.pkl', 'rb') as f:
    marqo_data = pickle.load(f)

image_embeddings = marqo_data['image_embeddings']
text_embeddings = marqo_data['text_embeddings']
metadata = marqo_data['metadata']

print(f"   ✅ Loaded {len(image_embeddings)} image embeddings")
print(f"   ✅ Loaded {len(text_embeddings)} text embeddings")
print(f"   ✅ Loaded {len(metadata)} metadata entries")

# Step 2: Initialize ChromaDB
print("\n2️⃣ Initializing ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

# Step 3: Create collection for IMAGE embeddings
print("\n3️⃣ Creating ChromaDB collection for image embeddings...")

# Delete if exists (for clean migration)
try:
    client.delete_collection("product_image_embeddings")
    print("   ⚠️  Deleted existing collection")
except:
    pass

# Create new collection
image_collection = client.create_collection(
    name="product_image_embeddings",
    metadata={"description": "Marqo-Ecommerce-L image embeddings (1024-dim)"}
)

print("   ✅ Created collection: product_image_embeddings")

# Step 4: Prepare data for ChromaDB
print("\n4️⃣ Preparing data for ChromaDB...")

ids = []
embeddings = []
metadatas = []
documents = []

for prod_id in image_embeddings.keys():
    # ID
    ids.append(f"img_{prod_id}")
    
    # Embedding (convert from numpy to list)
    embedding = image_embeddings[prod_id].flatten().tolist()
    embeddings.append(embedding)
    
    # Metadata
    meta = metadata[prod_id]
    metadatas.append({
        'row_id': str(prod_id),
        'product_name': meta['product_name'],
        'brand': meta['brand'],
        'category': meta.get('category', ''),
        'image_url': meta['image_url']
    })
    
    # Document (text representation for ChromaDB)
    doc_text = f"{meta['product_name']} {meta['brand']} {meta.get('category', '')}"
    documents.append(doc_text)

print(f"   ✅ Prepared {len(ids)} entries for ChromaDB")

# Step 5: Add to ChromaDB
print("\n5️⃣ Adding embeddings to ChromaDB...")
print("   (This may take a minute...)")

# Add in batches for safety
batch_size = 10
for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    batch_metadatas = metadatas[i:i+batch_size]
    batch_documents = documents[i:i+batch_size]
    
    image_collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas,
        documents=batch_documents
    )
    
    print(f"   ✅ Added batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")

# Step 6: Verify migration
print("\n6️⃣ Verifying migration...")
count = image_collection.count()
print(f"   ✅ ChromaDB collection has {count} embeddings")

# Test query
print("\n7️⃣ Testing query...")
test_query = [0.1] * 1024  # Dummy query vector
results = image_collection.query(
    query_embeddings=[test_query],
    n_results=3
)
print(f"   ✅ Query successful! Found {len(results['ids'][0])} results")
print(f"   Sample result: {results['metadatas'][0][0]['product_name']}")

# Summary
print("\n" + "="*60)
print("✅ MIGRATION COMPLETE!")
print("="*60)
print(f"📊 Statistics:")
print(f"   - Image embeddings migrated: {count}")
print(f"   - Stored in: ./chroma_db/")
print(f"   - Collection name: product_image_embeddings")
print(f"   - Original pickle file: PRESERVED (backup)")
print("="*60)
