"""Unified retrieval using ChromaDB for both text and images"""
import chromadb
import open_clip
import torch
from FlagEmbedding import FlagModel
import re

class UnifiedRetriever:
    def __init__(self):
        print("🚀 Initializing Unified Retriever...")
        
        # Initialize TWO ChromaDB clients (different paths)
        print("   Connecting to ChromaDB...")
        
        # Text collection (BGE) - in chroma_db_bge folder
        self.text_client = chromadb.PersistentClient(path="./chroma_db_bge")
        print("   Loading text collection (BGE) from ./chroma_db_bge...")
        self.text_collection = self.text_client.get_collection("product_text_embeddings")
        
        # Image collection (Marqo) - in chroma_db folder
        self.image_client = chromadb.PersistentClient(path="./chroma_db")
        print("   Loading image collection (Marqo) from ./chroma_db...")
        self.image_collection = self.image_client.get_collection("product_image_embeddings")
        
        # Load models
        print("   Loading BGE model...")
        self.bge_model = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)
        
        print("   Loading Marqo model...")
        self.marqo_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
        )
        self.marqo_tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-ecommerce-embeddings-L')
        self.marqo_model.eval()
        
        print("✅ Unified Retriever ready!")
        print(f"   📝 Text embeddings: {self.text_collection.count()} products")
        print(f"   📸 Image embeddings: {self.image_collection.count()} products\n")
    
    def search(self, query, top_k=8):
        """Search using both text and image embeddings"""
        print(f"\n🔍 Query: '{query}'")
        
        # TEXT SEARCH (BGE)
        print("   📝 Searching text embeddings...")
        text_embedding = self.bge_model.encode([query])[0].tolist()
        
        text_results = self.text_collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k,
            include=['metadatas', 'distances', 'documents']
        )
        
        # IMAGE SEARCH (Marqo)
        print("   📸 Searching image embeddings...")
        text_input = self.marqo_tokenizer([query])
        with torch.no_grad():
            query_feat = self.marqo_model.encode_text(text_input)
            query_feat /= query_feat.norm(dim=-1, keepdim=True)
        image_embedding = query_feat[0].tolist()
        
        image_results = self.image_collection.query(
            query_embeddings=[image_embedding],
            n_results=top_k,
            include=['metadatas', 'distances', 'documents']
        )
        
        # MERGE RESULTS
        print("   🔄 Merging results...")
        merged_products = self._merge_and_rank(query, text_results, image_results, top_k)
        
        print(f"   ✅ Found {len(merged_products)} products\n")
        return merged_products
    
    def _merge_and_rank(self, query, text_results, image_results, top_k):
        """Merge text and image results with smart ranking"""
        
        products = {}
        
        # Process text results
        for i, metadata in enumerate(text_results['metadatas'][0]):
            row_id = metadata['row_id']
            distance = text_results['distances'][0][i]
            
            if row_id not in products:
                products[row_id] = {
                    'name': metadata['product_name'],
                    'brand': metadata['brand'],
                    'category': metadata.get('category', ''),
                    'image_url': metadata.get('image_url', ''),
                    'description': text_results['documents'][0][i][:200],
                    'row_id': row_id,
                    'in_text': True,
                    'in_image': False,
                    'text_distance': distance,
                    'text_rank': i + 1
                }
            else:
                products[row_id]['in_text'] = True
                products[row_id]['text_distance'] = distance
                products[row_id]['text_rank'] = i + 1
        
        # Process image results
        for i, metadata in enumerate(image_results['metadatas'][0]):
            row_id = metadata['row_id']
            distance = image_results['distances'][0][i]
            
            if row_id not in products:
                products[row_id] = {
                    'name': metadata['product_name'],
                    'brand': metadata['brand'],
                    'category': metadata.get('category', ''),
                    'image_url': metadata['image_url'],
                    'description': '',
                    'row_id': row_id,
                    'in_text': False,
                    'in_image': True,
                    'image_distance': distance,
                    'image_rank': i + 1
                }
            else:
                products[row_id]['in_image'] = True
                products[row_id]['image_distance'] = distance
                products[row_id]['image_rank'] = i + 1
        
        # CALCULATE FINAL SCORE
        for prod_id, product in products.items():
            score = 0.0
            
            # Name matching bonus
            name_match = self._calculate_name_similarity(query, product['name'])
            score += name_match * 100
            
            # Synchronization bonus (both text and image)
            if product['in_text'] and product['in_image']:
                score += 50
                product['synchronized'] = True
            else:
                product['synchronized'] = False
            
            # Text ranking (lower distance = better)
            if product['in_text']:
                score += (1 / (1 + product['text_distance'])) * 20
            
            # Image ranking (lower distance = better)
            if product['in_image']:
                score += (1 / (1 + product['image_distance'])) * 30
            
            product['final_score'] = score
        
        # Sort by final score
        sorted_products = sorted(products.values(), key=lambda x: x['final_score'], reverse=True)
        
        return sorted_products[:top_k]
    
    def _calculate_name_similarity(self, query, product_name):
        """Calculate name match score"""
        query_lower = query.lower()
        product_lower = product_name.lower()
        
        if query_lower == product_lower:
            return 1.0
        
        query_words = set(re.findall(r'\w+', query_lower))
        product_words = set(re.findall(r'\w+', product_lower))
        
        if query_words and query_words.issubset(product_words):
            return 0.8
        
        overlap = query_words & product_words
        if overlap and query_words:
            return len(overlap) / len(query_words) * 0.6
        
        return 0.0

# Test the retriever
if __name__ == "__main__":
    retriever = UnifiedRetriever()
    
    # Test queries
    test_queries = [
        "emergency light with battery",
        "bedroom decorative light",
        "high wattage commercial bulb"
    ]
    
    for query in test_queries:
        results = retriever.search(query, top_k=5)
        
        print(f"📊 Results for: '{query}'")
        print("-" * 60)
        for i, product in enumerate(results, 1):
            sync = "✓ SYNCED" if product['synchronized'] else ""
            in_text = "📝" if product['in_text'] else ""
            in_image = "📸" if product['in_image'] else ""
            print(f"{i}. {product['name']} - {product['brand']} {sync}")
            print(f"   Score: {product['final_score']:.2f} | {in_text} {in_image}")
        print()
