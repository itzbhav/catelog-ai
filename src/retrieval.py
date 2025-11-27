"""Multimodal retrieval combining BGE text and Marqo-Ecommerce image embeddings"""
import pickle
import numpy as np
from langchain_chroma import Chroma
from FlagEmbedding import FlagModel
import open_clip
import torch

class MultimodalRetriever:
    def __init__(self):
        print("Initializing Multimodal Retriever...")
        
        # Load BGE for text
        self.bge_model = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)
        
        class BGEEmbeddings:
            def __init__(self, model):
                self.model = model
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
        
        # Load text vector store
        self.text_store = Chroma(
            persist_directory="./chroma_db_bge",
            embedding_function=BGEEmbeddings(self.bge_model),
            collection_name="product_text_embeddings"
        )
        
        # Load image embeddings
        with open('marqo_ecommerce_embeddings.pkl', 'rb') as f:
            self.image_data = pickle.load(f)
        
        # Load Marqo-Ecommerce model for query
        self.marqo_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
        )
        self.marqo_tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-ecommerce-embeddings-L')
        self.marqo_model.eval()
        
        print("✅ Retriever ready!")
    
    def retrieve(self, query, top_k=5):
        """Retrieve products using both text and image search"""
        
        # 1. Text search (BGE)
        text_results = self.text_store.similarity_search(query, k=top_k)
        
        # 2. Image search (Marqo-Ecommerce)
        text_input = self.marqo_tokenizer([query])
        with torch.no_grad():
            query_feat = self.marqo_model.encode_text(text_input)
            query_feat /= query_feat.norm(dim=-1, keepdim=True)
        query_embedding = query_feat.numpy()
        
        # Calculate similarities
        similarities = {}
        for prod_id, img_emb in self.image_data['image_embeddings'].items():
            sim = np.dot(query_embedding.flatten(), img_emb.flatten())
            similarities[prod_id] = float(sim)
        
        # Top-k images
        top_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 3. Combine results
        combined_results = []
        for doc in text_results:
            row_id = doc.metadata.get('row_id')
            combined_results.append({
                'product_name': doc.metadata.get('product_name'),
                'brand': doc.metadata.get('brand'),
                'description': doc.page_content,
                'image_url': doc.metadata.get('image_url'),
                'source': 'text',
                'row_id': row_id
            })
        
        for prod_id, score in top_images:
            meta = self.image_data['metadata'][prod_id]
            combined_results.append({
                'product_name': meta['product_name'],
                'brand': meta['brand'],
                'description': meta.get('description', ''),
                'image_url': meta['image_url'],
                'source': 'image',
                'score': score
            })
        
        # Remove duplicates (keep highest scoring)
        seen = set()
        unique_results = []
        for item in combined_results:
            key = item['product_name']
            if key not in seen:
                seen.add(key)
                unique_results.append(item)
        
        return unique_results[:top_k]

if __name__ == "__main__":
    retriever = MultimodalRetriever()
    
    test_query = "bedroom ceiling light"
    print(f"\n🔍 Query: {test_query}")
    results = retriever.retrieve(test_query, top_k=3)
    
    print(f"\n📊 Top 3 Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['product_name']} - {result['brand']}")
        print(f"   Image: {result['image_url'][:60]}...")
