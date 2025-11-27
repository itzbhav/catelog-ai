"""Test synchronization between text and image search results"""
import pickle
import numpy as np
from langchain_chroma import Chroma
from FlagEmbedding import FlagModel
import open_clip
import torch

print("Loading models and data...\n")

# Load BGE
bge_model = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)

class BGEEmbeddings:
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

text_store = Chroma(
    persist_directory="./chroma_db_bge",
    embedding_function=BGEEmbeddings(bge_model),
    collection_name="product_text_embeddings"
)

# Load image data
with open('marqo_ecommerce_embeddings.pkl', 'rb') as f:
    image_data = pickle.load(f)

# Load Marqo model
marqo_model, _, _ = open_clip.create_model_and_transforms(
    'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
)
marqo_tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-ecommerce-embeddings-L')
marqo_model.eval()

def test_query(query_text):
    print("="*80)
    print(f"🔍 QUERY: '{query_text}'")
    print("="*80)
    
    # TEXT SEARCH (BGE)
    print("\n📝 TEXT SEARCH RESULTS (BGE Model):")
    print("-"*80)
    text_results = text_store.similarity_search(query_text, k=5)
    text_products = []
    
    for i, doc in enumerate(text_results, 1):
        prod_name = doc.metadata.get('product_name', 'Unknown')
        brand = doc.metadata.get('brand', 'Unknown')
        row_id = doc.metadata.get('row_id', 'N/A')
        text_products.append((prod_name, brand, row_id))
        print(f"   {i}. {prod_name}")
        print(f"      Brand: {brand}")
        print(f"      Row ID: {row_id}")
        print()
    
    # IMAGE SEARCH (Marqo-Ecommerce)
    print("📸 IMAGE SEARCH RESULTS (Marqo-Ecommerce Model):")
    print("-"*80)
    
    text_input = marqo_tokenizer([query_text])
    with torch.no_grad():
        query_feat = marqo_model.encode_text(text_input)
        query_feat /= query_feat.norm(dim=-1, keepdim=True)
    query_embedding = query_feat.numpy()
    
    similarities = {}
    for prod_id, img_emb in image_data['image_embeddings'].items():
        sim = np.dot(query_embedding.flatten(), img_emb.flatten())
        similarities[prod_id] = sim
    
    top_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    image_products = []
    
    for i, (prod_id, score) in enumerate(top_images, 1):
        meta = image_data['metadata'][prod_id]
        prod_name = meta['product_name']
        brand = meta['brand']
        image_products.append((prod_name, brand, prod_id))
        print(f"   {i}. {prod_name}")
        print(f"      Brand: {brand}")
        print(f"      Similarity Score: {score:.4f}")
        print(f"      Row ID: {prod_id}")
        print()
    
    # SYNCHRONIZATION CHECK
    print("🎯 SYNCHRONIZATION ANALYSIS:")
    print("-"*80)
    
    text_names = set([name for name, _, _ in text_products])
    image_names = set([name for name, _, _ in image_products])
    
    synchronized = text_names & image_names
    text_only = text_names - image_names
    image_only = image_names - text_names
    
    if synchronized:
        print(f"✅ SYNCHRONIZED (appear in both):")
        for name in synchronized:
            print(f"   • {name}")
    else:
        print("⚠️  No products appear in both top-5 results")
    
    if text_only:
        print(f"\n📝 TEXT ONLY (not in image top-5):")
        for name in text_only:
            print(f"   • {name}")
    
    if image_only:
        print(f"\n📸 IMAGE ONLY (not in text top-5):")
        for name in image_only:
            print(f"   • {name}")
    
    sync_percentage = (len(synchronized) / 5) * 100 if synchronized else 0
    print(f"\n📊 Synchronization Rate: {sync_percentage:.0f}% ({len(synchronized)}/5 matches)")
    print()

# Test with different queries
test_queries = [
    "emergency light with battery backup",
    "decorative candle bulb",
    "high wattage LED bulb for commercial use",
    "filament vintage style lamp"
]

for query in test_queries:
    test_query(query)
    input("\nPress Enter to test next query...\n")
