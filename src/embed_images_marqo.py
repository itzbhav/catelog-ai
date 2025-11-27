"""Embed images from Excel using Marqo with marqo-ecommerce model from HuggingFace"""
import pandas as pd
import marqo

def embed_images_from_excel(excel_path):
    print("Loading Excel file...")
    df = pd.read_excel(excel_path)
    
    # Connect to Marqo server
    print("Connecting to Marqo server...")
    mq = marqo.Client(url='http://localhost:8882')
    
    index_name = "product_images"
    
    # Delete index if it exists
    try:
        mq.delete_index(index_name)
        print(f"Deleted existing index: {index_name}")
    except:
        pass
    
    print(f"Creating index with Marqo E-commerce model from HuggingFace...")
    
    # IMPORTANT: Use hf/Marqo prefix to load from HuggingFace
    mq.create_index(
        index_name,
        type="unstructured",
        model="hf/Marqo/marqo-ecommerce-embeddings-B",  # ← Note the hf/ prefix!
        treat_urls_and_pointers_as_images=True,
        normalize_embeddings=True
    )
    
    # Prepare documents
    print(f"Processing {len(df)} products...")
    documents = []
    for idx, row in df.iterrows():
        doc = {
            "_id": str(idx),
            "product_name": str(row.get('product_name', '')),
            "brand": str(row.get('brand', '')),
            "category": str(row.get('category', '')),
            "image_url": str(row.get('image_url', '')),
            "description": str(row.get('description', ''))
        }
        documents.append(doc)
    
    # Add documents to Marqo
    print("Generating embeddings (this will download model on first run)...")
    batch_size = 5
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        try:
            mq.index(index_name).add_documents(
                batch,
                tensor_fields=["image_url", "product_name"]
            )
            print(f"✅ Processed {min(i+batch_size, len(documents))}/{len(documents)} products")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue
    
    print(f"\n✅ Successfully embedded {len(documents)} products with Marqo E-commerce model!")
    return mq

if __name__ == "__main__":
    embed_images_from_excel('./data/data4.xlsx')
