"""
Multimodal RAG Chatbot - Enhanced with Dynamic Configuration
- Auto-loads config from config/search_config.json
- Hierarchical category filtering
- Dynamic attribute extraction
- Hybrid scoring (vector + keyword + attributes)
- Comprehensive logging
"""


from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
import re
import traceback
from datetime import datetime
import uuid
import markdown
import logging
import time
import json


# ============================================================
# CONFIGURATION & LOGGING
# ============================================================


load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', '7f862a8a79175f1baef6c83a00a109e717321cb9529b5e3182ea327fddc56a23')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('search_logs.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# LOAD SEARCH CONFIGURATION
# ============================================================


SEARCH_CONFIG = None


def load_search_config():
    """Load search configuration from config folder"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'search_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Search configuration loaded from: {config_path}")
        logger.info(f"   Brands: {len(config['brands']['list'])}")
        logger.info(f"   Categories: {len(config['categories']['patterns'])}")
        return config
    except FileNotFoundError:
        logger.error("❌ search_config.json not found!")
        logger.error("   Run: python src/analyze_data.py")
        logger.error("   Then: python src/config_generator.py")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return None


# ============================================================
# LOAD AI MODELS & GEMINI
# ============================================================


def load_gemini():
    """Load Gemini model"""
    try:
        import google.generativeai as genai
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        if not GOOGLE_API_KEY:
            logger.error("❌ GOOGLE_API_KEY not found!")
            return None
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        test_response = gemini_model.generate_content("Say 'Ready!'")
        logger.info(f"✅ Gemini: {test_response.text}")
        return gemini_model
    except Exception as e:
        logger.error(f"❌ Gemini failed: {e}")
        traceback.print_exc()
        return None



def load_ai_models():
    """Load AI models with detailed error reporting"""
    models = {}
    
    # Load BGE Model
    try:
        from FlagEmbedding import FlagModel
        logger.info("🚀 Loading BGE model...")
        models["bge_model"] = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)
        logger.info("✅ BGE model loaded")
    except Exception as e:
        logger.error(f"❌ BGE model failed: {e}")
        traceback.print_exc()
        models["bge_model"] = None
    
    # Load ChromaDB Collections
    try:
        import chromadb
        logger.info("🚀 Loading ChromaDB clients...")
        text_client = chromadb.PersistentClient(path="./chroma_db_bge")
        image_client = chromadb.PersistentClient(path="./chroma_db")
        models["text_collection"] = text_client.get_collection("product_text_embeddings")
        models["image_collection"] = image_client.get_collection("product_image_embeddings")
        
        text_count = models["text_collection"].count()
        image_count = models["image_collection"].count()
        logger.info(f"✅ ChromaDB loaded - Text: {text_count}, Image: {image_count} embeddings")
    except Exception as e:
        logger.error(f"❌ ChromaDB failed: {e}")
        traceback.print_exc()
        models["text_collection"] = None
        models["image_collection"] = None
    
    # Load Marqo/OpenCLIP
    try:
        import open_clip
        import torch
        logger.info("🚀 Loading Marqo model...")
        marqo_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
        )
        marqo_tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-ecommerce-embeddings-L')
        marqo_model.eval()
        models["marqo_model"] = marqo_model
        models["marqo_tokenizer"] = marqo_tokenizer
        models["torch"] = torch
        logger.info("✅ Marqo model loaded")
    except Exception as e:
        logger.error(f"❌ Marqo model failed: {e}")
        traceback.print_exc()
        models["marqo_model"] = None
        models["marqo_tokenizer"] = None
        models["torch"] = None
    
    # Summary
    loaded = sum(1 for v in models.values() if v is not None)
    logger.info(f"✅ Loaded {loaded}/{len(models)} model components\n")
    
    return models if loaded > 0 else None



# Load resources at module level
logger.info("\n🔧 Initializing application...")
SEARCH_CONFIG = load_search_config()
gemini_model = load_gemini()
ai_models = load_ai_models()


# ============================================================
# PRODUCT TYPE CLASSIFICATION (NEW!)
# ============================================================


def classify_product_type(category):
    """
    Classify product type from category name.
    Matches logic from analyze_data.py
    """
    if not category:
        return 'product'
    
    category_lower = category.lower()
    
    SYSTEM_KEYWORDS = ['system', 'rail', 'track system', 'mounting', 'infrastructure']
    ACCESSORY_KEYWORDS = ['accessory', 'connector', 'adapter', 'joint', 'fitting', 'component', 'part', 'profile']
    
    if any(kw in category_lower for kw in SYSTEM_KEYWORDS):
        return 'system'
    if any(kw in category_lower for kw in ACCESSORY_KEYWORDS):
        return 'accessory'
    return 'product'



def detect_user_intent(query):
    """
    Detect what TYPE of product user wants based on query language.
    Generic - works for any domain.
    """
    query_lower = query.lower()
    
    # Intent: User wants installation/mounting components
    INSTALLATION_KEYWORDS = ['install', 'mount', 'mounting', 'rail', 'system', 'setup', 'infrastructure']
    if any(keyword in query_lower for keyword in INSTALLATION_KEYWORDS):
        return 'system'
    
    # Intent: User wants accessories/parts
    ACCESSORY_KEYWORDS = ['connector', 'accessory', 'adapter', 'fitting', 'part', 'component', 'join']
    if any(keyword in query_lower for keyword in ACCESSORY_KEYWORDS):
        return 'accessory'
    
    # Default: User wants the main product
    return 'product'



def filter_by_product_type(products, user_intent):
    """
    Filter products based on user intent vs product type.
    Boosts matching types, penalizes mismatches.
    """
    for product in products:
        category = product.get('category', '')
        product_type = classify_product_type(category)
        product['product_type'] = product_type
        
        # Safety check: Initialize final_score if it doesn't exist
        if 'final_score' not in product:
            product['final_score'] = product.get('text_score', 0) + product.get('image_score', 0)
        
        # If user intent matches product type, boost score
        if product_type == user_intent:
            product['final_score'] *= 1.5  # 50% boost
            product['type_match'] = True
        # If mismatch, heavily penalize
        elif user_intent in ['system', 'accessory'] and product_type == 'product':
            product['final_score'] *= 0.1  # 90% penalty
            product['type_match'] = False
        elif user_intent == 'product' and product_type in ['system', 'accessory']:
            product['final_score'] *= 0.2  # 80% penalty
            product['type_match'] = False
        else:
            product['type_match'] = True  # neutral
    
    return products


def detect_category_dynamic(query):
    """
    Detect category using config patterns with SPECIFICITY RANKING.
    Prefers more specific matches (e.g., "bollard light" over "light").
    """
    if not SEARCH_CONFIG:
        return None
    
    query_lower = query.lower()
    matches = []
    
    # Find all matching categories with their pattern specificity
    for category, patterns in SEARCH_CONFIG['categories']['patterns'].items():
        for pattern in patterns:
            if pattern in query_lower:
                # Score by pattern length (longer = more specific)
                specificity = len(pattern.split())
                matches.append((category, specificity, pattern))
    
    if not matches:
        logger.info(f"   📂 No specific category detected")
        return None
    
    # Sort by specificity (descending) - prefer longer, more specific patterns
    matches.sort(key=lambda x: x[1], reverse=True)
    
    best_category = matches[0][0]
    best_pattern = matches[0][2]
    logger.info(f"   📂 Category detected: {best_category} (matched: '{best_pattern}')")
    
    return best_category



def extract_attributes_dynamic(query):
    """Extract attributes using config patterns"""
    if not SEARCH_CONFIG:
        return {}
    
    query_lower = query.lower()
    attributes = {}
    
    patterns = SEARCH_CONFIG['attribute_patterns']
    
    # Extract wattage
    wattage_match = re.search(patterns['wattage'], query_lower)
    if wattage_match:
        attributes['wattage'] = wattage_match.group(1)
        logger.info(f"   ⚡ Wattage detected: {attributes['wattage']}W")
    
    # Extract voltage
    voltage_match = re.search(patterns['voltage'], query_lower)
    if voltage_match:
        attributes['voltage'] = voltage_match.group(1)
        logger.info(f"   🔌 Voltage detected: {attributes['voltage']}V")
    
    # Extract color temperature
    for color_type, keywords in patterns['color_temp'].items():
        if any(kw in query_lower for kw in keywords):
            attributes['color'] = color_type
            logger.info(f"   🎨 Color detected: {color_type.replace('_', ' ')}")
            break
    
    # Extract brand
    for brand in SEARCH_CONFIG['brands']['list']:
        if brand in query_lower:
            attributes['brand'] = brand
            logger.info(f"   🏷️  Brand detected: {brand.title()}")
            break
    
    return attributes



def calculate_dynamic_score(product, query, attributes, category):
    """Calculate score using config weights"""
    if not SEARCH_CONFIG:
        return 0
    
    weights = SEARCH_CONFIG['scoring_weights']
    score = 0
    
    content_lower = product.get('content', '').lower()
    
    # Base vector score
    if product.get('in_text'):
        text_score = (1 / (1 + product['text_distance'])) * 100
        score += text_score * weights['text_vector_weight']
    
    if product.get('in_image'):
        image_score = (1 / (1 + product['image_distance'])) * 100
        score += image_score * weights['image_vector_weight']
    
    # Synchronization bonus
    if product.get('in_text') and product.get('in_image'):
        score += weights['text_and_image_sync']
        product['synchronized'] = True
    else:
        product['synchronized'] = False
    
    # Keyword matching
    query_words = query.lower().split()
    keyword_matches = sum(1 for word in query_words if word in content_lower)
    keyword_score = keyword_matches * weights['keyword_match_per_word']
    score += keyword_score
    product['keyword_matches'] = keyword_matches
    product['keyword_score'] = keyword_score
    
    # Attribute matching
    attribute_score = 0
    
    if attributes.get('wattage') and f"{attributes['wattage']}w" in content_lower:
        attribute_score += weights['wattage_exact_match']
    
    if attributes.get('voltage') and f"{attributes['voltage']}v" in content_lower:
        attribute_score += weights['voltage_exact_match']
    
    if attributes.get('color'):
        color_keywords = SEARCH_CONFIG['attribute_patterns']['color_temp'][attributes['color']]
        if any(kw in content_lower for kw in color_keywords):
            attribute_score += weights['color_temp_match']
    
    if attributes.get('brand') and attributes['brand'] in product.get('brand', '').lower():
        attribute_score += weights['brand_exact_match']
    
    # Category match bonus
    if category and category in product.get('category', ''):
        attribute_score += weights['category_match']
    
    score += attribute_score
    product['attribute_score'] = attribute_score
    
    return score



# ============================================================
# CONVERSATION MEMORY
# ============================================================


def initialize_conversation():
    """Initialize conversation state"""
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
        session['last_products_shown'] = []
        session['last_query'] = ""
        session['awaiting_clarification'] = False
        session['original_query'] = None
        session['clarification_attempts'] = 0
        session.modified = True
        logger.info(f"🆕 New conversation: {session['conversation_id'][:8]}")
    return session['conversation_id']



def add_to_conversation(user_query, bot_response, products):
    """Add to conversation memory"""
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    products_with_position = []
    for i, prod in enumerate(products, 1):
        products_with_position.append({
            'position': i,
            'name': prod['name'],
            'brand': prod['brand'],
            'row_id': prod.get('row_id', f"prod_{i}")
        })
    
    conversation_entry = {
        'timestamp': datetime.now().isoformat(),
        'user': user_query,
        'assistant': bot_response[:500],
        'products': products_with_position
    }
    
    session['conversation_history'].append(conversation_entry)
    session['last_products_shown'] = products_with_position
    session['last_query'] = user_query
    
    if len(session['conversation_history']) > 10:
        session['conversation_history'] = session['conversation_history'][-10:]
    
    session.modified = True



def get_conversation_context():
    """Get conversation context"""
    if 'conversation_history' not in session or not session['conversation_history']:
        return ""
    
    context = "Previous Conversation:\n"
    for entry in session['conversation_history'][-3:]:
        context += f"\nUser: {entry['user']}\n"
        context += f"Assistant: {entry['assistant'][:200]}...\n"
        if 'products' in entry and entry['products']:
            product_names = [p['name'] for p in entry['products'][:3]]
            context += f"Products shown: {', '.join(product_names)}\n"
    
    return context



def resolve_contextual_query(query):
    """Resolve contextual queries"""
    query_lower = query.lower()
    
    contextual_keywords = ['cheaper', 'expensive', 'similar', 'brighter', 'dimmer',
                          'compare', 'difference', 'alternative', 'option']
    
    is_contextual = any(keyword in query_lower for keyword in contextual_keywords)
    
    if is_contextual and 'last_query' in session and session['last_query']:
        previous_query = session['last_query']
        expanded_query = f"{previous_query} {query}"
        logger.info(f"   🔗 Expanded: '{expanded_query}'")
        return expanded_query, True
    
    return query, False



# ============================================================
# ENHANCED RETRIEVAL WITH DYNAMIC CONFIG
# ============================================================


def retrieve_products(query, top_k=10):
    """Enhanced retrieval with dynamic configuration and hierarchical filtering"""
    global ai_models
    
    if ai_models is None:
        logger.error("❌ AI models not available")
        return []
    
    if not SEARCH_CONFIG:
        logger.error("❌ Search config not loaded")
        return []
    
    start_time = time.time()
    
    logger.info("\n" + "="*70)
    logger.info(f"🔍 QUERY: '{query}'")
    logger.info("="*70)
    
    # ============================================================
    # STEP 1: ANALYZE QUERY (Dynamic!)
    # ============================================================
    logger.info("\n🧠 STEP 1: ANALYZING QUERY")
    
    category = detect_category_dynamic(query)
    attributes = extract_attributes_dynamic(query)
    
    logger.info(f"   Detected attributes: {attributes}")
    
    # ============================================================
    # STEP 2: HIERARCHICAL TEXT SEARCH
    # ============================================================
    logger.info("\n📝 STEP 2: TEXT EMBEDDING SEARCH")
    
    bge_model = ai_models["bge_model"]
    text_collection = ai_models["text_collection"]
    
    text_embedding = bge_model.encode([query])[0].tolist()
    
    # Build ChromaDB filter for hierarchical search
    where_filter = None
    if category:
        where_filter = {"category": {"$eq": category}}
        logger.info(f"   🔍 Filtering by category: {category}")
    
    # Search with optional category filter
    try:
        text_results = text_collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k * 3,
            where=where_filter,
            include=['metadatas', 'distances', 'documents']
        )
    except Exception as e:
        logger.warning(f"   ⚠️ Filtered search failed, trying without filter: {e}")
        text_results = text_collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k * 3,
            include=['metadatas', 'distances', 'documents']
        )
    
    if text_results and text_results['metadatas']:
        logger.info(f"   ✅ Found {len(text_results['metadatas'][0])} text matches")
        logger.info(f"   📊 Top 3 distances: {[round(d, 3) for d in text_results['distances'][0][:3]]}")
    
    # ============================================================
    # STEP 3: HIERARCHICAL IMAGE SEARCH
    # ============================================================
    logger.info("\n🖼️  STEP 3: IMAGE EMBEDDING SEARCH")
    
    image_collection = ai_models["image_collection"]
    marqo_model = ai_models["marqo_model"]
    marqo_tokenizer = ai_models["marqo_tokenizer"]
    torch = ai_models["torch"]
    
    text_input = marqo_tokenizer([query])
    with torch.no_grad():
        query_feat = marqo_model.encode_text(text_input)
        query_feat /= query_feat.norm(dim=-1, keepdim=True)
    image_embedding = query_feat[0].tolist()
    
    try:
        image_results = image_collection.query(
            query_embeddings=[image_embedding],
            n_results=top_k * 3,
            where=where_filter,
            include=['metadatas', 'distances', 'documents']
        )
    except Exception as e:
        logger.warning(f"   ⚠️ Filtered search failed, trying without filter: {e}")
        image_results = image_collection.query(
            query_embeddings=[image_embedding],
            n_results=top_k * 3,
            include=['metadatas', 'distances', 'documents']
        )
    
    if image_results and image_results['metadatas']:
        logger.info(f"   ✅ Found {len(image_results['metadatas'][0])} image matches")
        logger.info(f"   📊 Top 3 distances: {[round(d, 3) for d in image_results['distances'][0][:3]]}")
    
    # ============================================================
    # STEP 4: MERGE RESULTS
    # ============================================================
    logger.info("\n🔄 STEP 4: MERGING TEXT + IMAGE RESULTS")
    
    products = {}
    
    # Process text results
    for i, metadata in enumerate(text_results['metadatas'][0]):
        row_id = metadata.get('row_id', f'text_{i}')
        products[row_id] = {
            'name': metadata.get('product_name', 'Unknown'),
            'brand': metadata.get('brand', 'Unknown'),
            'category': metadata.get('category', ''),
            'image_url': metadata.get('image_url', ''),
            'image_url_2': metadata.get('image_url_2', ''),
            'description': text_results['documents'][0][i][:200] if i < len(text_results['documents'][0]) else '',
            'content': text_results['documents'][0][i] if i < len(text_results['documents'][0]) else '',
            'row_id': row_id,
            'in_text': True,
            'in_image': False,
            'text_distance': text_results['distances'][0][i]
        }
    
    # Process image results
    for i, metadata in enumerate(image_results['metadatas'][0]):
        row_id = metadata.get('row_id', f'image_{i}')
        if row_id in products:
            products[row_id]['in_image'] = True
            products[row_id]['image_distance'] = image_results['distances'][0][i]
        else:
            products[row_id] = {
                'name': metadata.get('product_name', 'Unknown'),
                'brand': metadata.get('brand', 'Unknown'),
                'category': metadata.get('category', ''),
                'image_url': metadata.get('image_url', ''),
                'image_url_2': '',
                'description': '',
                'content': '',
                'row_id': row_id,
                'in_text': False,
                'in_image': True,
                'image_distance': image_results['distances'][0][i]
            }
    
    synchronized_count = sum(1 for p in products.values() if p['in_text'] and p['in_image'])
    logger.info(f"   📊 Total unique products: {len(products)}")
    logger.info(f"   ✅ Synchronized (text + image): {synchronized_count}")
    
    # ============================================================
    # STEP 4.5: DEDUPLICATE
    # ============================================================
    logger.info("\n🔄 STEP 4.5: DEDUPLICATING RESULTS")
    
    seen_ids = set()
    unique_products = {}
    
    for row_id, product in products.items():
        if row_id not in seen_ids:
            seen_ids.add(row_id)
            unique_products[row_id] = product
    
    products = unique_products
    logger.info(f"   ✅ Deduplicated to {len(products)} unique products")
    
    all_products = list(products.values())
    
    # ============================================================
    # STEP 5: DYNAMIC SCORING
    # ============================================================
    logger.info("\n⚡ STEP 5: HYBRID SCORING")
    
    for product in all_products:
        score = calculate_dynamic_score(product, query, attributes, category)
        product['final_score'] = score  # Ensure it's assigned
    
    # ============================================================
    # STEP 5.5: PRODUCT TYPE FILTERING (FIXED - OUTSIDE LOOP!)
    # ============================================================
    logger.info("\n🎯 STEP 5.5: PRODUCT TYPE FILTERING")
    
    user_intent = detect_user_intent(query)
    logger.info(f"   User intent detected: {user_intent}")
    
    all_products = filter_by_product_type(all_products, user_intent)
    
    # Sort by score AFTER type filtering
    all_products.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    # ============================================================
    # STEP 6: LOG TOP RESULTS
    # ============================================================
    logger.info(f"\n🏆 TOP 3 RESULTS:")
    for i, p in enumerate(all_products[:3], 1):
        logger.info(f"\n   #{i} {p['name']} by {p['brand']}")
        logger.info(f"      Category: {p.get('category', 'Unknown')}")
        logger.info(f"      Product Type: {p.get('product_type', 'Unknown')}")
        logger.info(f"      Type Match: {'✅' if p.get('type_match', False) else '❌'}")
        logger.info(f"      Final Score: {p.get('final_score', 0):.2f}")
        logger.info(f"      Text Match: {'✅' if p.get('in_text', False) else '❌'}")
        logger.info(f"      Image Match: {'✅' if p.get('in_image', False) else '❌'}")
        logger.info(f"      Keywords: {p.get('keyword_matches', 0)}/{len(query.split())} (+{p.get('keyword_score', 0)})")
        logger.info(f"      Attributes: +{p.get('attribute_score', 0)}")
        logger.info(f"      Synchronized: {'✅' if p.get('synchronized', False) else '❌'}")
    
    search_time = (time.time() - start_time) * 1000
    logger.info(f"\n⏱️  SEARCH COMPLETED in {search_time:.0f}ms")
    logger.info(f"📦 Returning {min(len(all_products), top_k)} products")
    logger.info("="*70 + "\n")
    
    return all_products[:top_k]



def generate_response_with_gemini(query, products, conversation_context, was_clarification=False):
    """Generate response with clarification awareness"""
    global gemini_model
    
    logger.info(f"📤 Generating Gemini response...")
    
    if not gemini_model:
        return "⚠️ AI unavailable.", products[:3]
    
    # Build product list
    product_context = "AVAILABLE PRODUCTS (mention ONLY these by EXACT NAME):\n\n"
    for i, prod in enumerate(products, 1):
        sync = " (HIGH CONFIDENCE)" if prod.get('synchronized', False) else ""
        product_context += f"{i}. **{prod['name']}** by {prod['brand']}{sync}\n"
    
    # Add clarification acknowledgment if needed
    clarification_note = ""
    if was_clarification:
        words = query.lower().split()
        context_word = next((w for w in words if w not in ['i', 'need', 'want', 'show', 'me', 'lights', 'bulbs', 'for', 'the', 'a']), words[0] if words else 'your')
        clarification_note = f"\n\n✨ NOTE: The user just provided clarification. Start your response by acknowledging this naturally (e.g., 'Perfect! For {context_word} lighting, here are...' or 'Great choice! For {context_word} spaces, I recommend...').\n"
    
    prompt = f"""You are a professional lighting assistant.

{conversation_context}
{clarification_note}

Question: "{query}"

{product_context}

FORMAT (Keep response under 150 words total):

Perfect! [1 sentence acknowledgment].

**Recommended Products:**

1. **[Product Name]** by [Brand]
   - [Feature 1]
   - [Feature 2]
   - Why it's perfect: [1 sentence benefit]

2. **[Product Name]** by [Brand]
   - [Feature 1]
   - [Feature 2]
   - Why it's perfect: [1 sentence benefit]

[Optional 3rd product if highly relevant]

**Key Benefits:**
- [Benefit 1]
- [Benefit 2]
- [Benefit 3]

[End with 1 question]

RULES:
- Recommend 2-3 products maximum (use EXACT names from list above)
- Use bullet points (-) not (•)
- Keep each point under 8 words
- Total response: 100-150 words
- Be conversational, not robotic
- {f"Acknowledge their clarification naturally" if was_clarification else "Start with 'Perfect!' or 'Great choice!'"}

Response:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text
        logger.info(f"✅ Gemini response generated ({len(response_text)} chars)")
        
        # Extract mentioned products
        mentioned_products = extract_strictly_mentioned_products(response_text, products)
        
        if not mentioned_products:
            logger.warning("⚠️ No products clearly mentioned, using top 3 synced")
            mentioned_products = [p for p in products if p.get('synchronized', False)][:3]
            if not mentioned_products:
                mentioned_products = products[:3]
        
        logger.info(f"✅ Displaying {len(mentioned_products)} mentioned products")
        
        return response_text, mentioned_products
        
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
        return "I found great products for you!", products[:3]



def extract_strictly_mentioned_products(response_text, all_products):
    """Extract products explicitly mentioned by name"""
    mentioned = []
    response_lower = response_text.lower()
    
    for product in all_products:
        product_name_lower = product['name'].lower()
        
        # Exact full name match
        if product_name_lower in response_lower:
            mentioned.append(product)
            continue
        
        # Word-by-word matching
        product_words = set(re.findall(r'\w+', product_name_lower))
        product_words -= {'philips', 'mascon', 'led', 'bulb', 'lamp', 'light', 'by', 'and', 'the'}
        
        if len(product_words) >= 2:
            words_found = sum(1 for word in product_words if f' {word} ' in f' {response_lower} ')
            match_ratio = words_found / len(product_words)
            
            threshold = 1.0 if len(product_words) == 2 else 0.8
            
            if match_ratio >= threshold:
                mentioned.append(product)
    
    return mentioned[:4]



# ============================================================
# FLASK ROUTES
# ============================================================


@app.route('/')
def home():
    initialize_conversation()
    return render_template('chatbot.html')



@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with enhanced search"""
    try:
        if ai_models is None:
            return jsonify({
                'error': 'AI models not loaded',
                'response': '❌ Sorry, the AI models are still loading or failed to load. Please try again in a moment.',
                'products': [],
                'stats': {'total': 0, 'synchronized': 0, 'sync_percentage': 0}
            }), 503
        
        if not SEARCH_CONFIG:
            return jsonify({
                'error': 'Configuration not loaded',
                'response': '❌ Sorry, the search configuration is not loaded. Please restart the application.',
                'products': [],
                'stats': {'total': 0, 'synchronized': 0, 'sync_percentage': 0}
            }), 503
        
        query = request.json.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        conversation_id = initialize_conversation()
        
        logger.info(f"\n{'🔵'*35}")
        logger.info(f"💬 USER MESSAGE: '{query}'")
        logger.info(f"   Conversation ID: {conversation_id[:8]}")
        logger.info(f"{'🔵'*35}")
        
        was_clarification = False
        
        # Check if responding to clarification
        if session.get('awaiting_clarification', False) and session.get('original_query'):
            original = session.get('original_query', '')
            
            if query.lower() != original.lower() and len(original) > 0:
                logger.info(f"✅ Received clarification response")
                combined_query = f"{query} {original}"
                logger.info(f"   Combined: '{combined_query}'")
                
                session['awaiting_clarification'] = False
                session['original_query'] = None
                was_clarification = True
                session.modified = True
                
                query = combined_query
            else:
                session['awaiting_clarification'] = False
                session['original_query'] = None
                session['clarification_attempts'] = 0
                session.modified = True
        
        # Check for ambiguous query
        if not was_clarification:
            ambiguous_patterns = ['i need', 'show me', 'want', 'looking for', 'lights', 'bulbs', 'give me', 'find me']
            is_ambiguous = any(pattern in query.lower() for pattern in ambiguous_patterns) and len(query.split()) < 5
            
            max_attempts = 2
            
            if is_ambiguous and session.get('clarification_attempts', 0) < max_attempts:
                logger.info(f"❓ Ambiguous query - asking clarification")
                
                session['clarification_attempts'] = session.get('clarification_attempts', 0) + 1
                session['awaiting_clarification'] = True
                session['original_query'] = query
                session.modified = True
                
                clarification_response = """I'd be happy to help you find the perfect lighting! To give you the best recommendations, could you tell me:

🏠 **Room Type?** (bedroom, living room, kitchen, bathroom, outdoor)
💡 **Purpose?** (general lighting, task lighting, decorative, accent lighting)
✨ **Style Preference?** (modern, traditional, minimalist)
💰 **Budget Range?** (if any)

Just describe what you're looking for and I'll find the best options!"""
                
                html_clarification = markdown.markdown(clarification_response, extensions=['nl2br', 'sane_lists'])
                
                return jsonify({
                    'query': query,
                    'response': html_clarification,
                    'products': [],
                    'is_clarification': True,
                    'stats': {'total': 0, 'synchronized': 0, 'sync_percentage': 0}
                })
            
            elif is_ambiguous and session.get('clarification_attempts', 0) >= max_attempts:
                logger.info(f"⚠️ Max clarification attempts reached. Proceeding...")
                session['clarification_attempts'] = 0
                session['awaiting_clarification'] = False
                session.modified = True
        
        # Continue with enhanced search
        expanded_query, is_contextual = resolve_contextual_query(query)
        search_query = expanded_query if is_contextual else query
        
        conversation_context = get_conversation_context()
        
        all_products = retrieve_products(search_query, top_k=10)
        
        if not all_products:
            return jsonify({
                'query': query,
                'response': "I couldn't find any products matching your query. Please try rephrasing or ask about different lighting products.",
                'products': [],
                'is_clarification': False,
                'stats': {'total': 0, 'synchronized': 0, 'sync_percentage': 0}
            })
        
        ai_response, display_products = generate_response_with_gemini(
            query, 
            all_products, 
            conversation_context,
            was_clarification=was_clarification
        )
        
        add_to_conversation(query, ai_response, display_products)
        
        html_response = markdown.markdown(ai_response, extensions=['nl2br', 'sane_lists'])
        
        if was_clarification:
            session['clarification_attempts'] = 0
            session.modified = True
        
        sync_count = sum(1 for p in display_products if p.get('synchronized', False))
        sync_pct = (sync_count / len(display_products) * 100) if display_products else 0
        
        logger.info(f"✅ Response sent to user ({len(display_products)} products)")
        
        return jsonify({
            'query': query,
            'response': html_response,
            'products': display_products,
            'is_clarification': False,
            'stats': {
                'total': len(display_products),
                'synchronized': sync_count,
                'sync_percentage': round(sync_pct, 0)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ ERROR in /chat: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'response': '❌ Sorry, something went wrong. Please try again.',
            'products': [],
            'stats': {'total': 0, 'synchronized': 0, 'sync_percentage': 0}
        }), 500



@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation and reset session"""
    session.clear()
    return jsonify({'message': 'Conversation cleared'})



@app.route('/reset_debug', methods=['GET'])
def reset_debug():
    """Debug: Force reset session"""
    session.clear()
    return jsonify({
        'message': 'Session cleared!',
        'note': 'Refresh the chat page and try again'
    })



if __name__ == '__main__':
    if not SEARCH_CONFIG:
        print("\n" + "="*70)
        print("❌ ERROR: search_config.json not found!")
        print("="*70)
        print("Please run these commands first:")
        print("  1. python src/analyze_data.py")
        print("  2. python src/config_generator.py")
        print("="*70 + "\n")
        exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("🤖 ENHANCED MULTIMODAL RAG CHATBOT")
    logger.info("="*70)
    logger.info("✨ Features:")
    logger.info("   • Dynamic configuration (no hardcoding!)")
    logger.info("   • Hierarchical category filtering")
    logger.info("   • Product type classification (system/accessory/product)")
    logger.info("   • Auto-discovered attributes")
    logger.info("   • Hybrid scoring (vector + keyword + attributes)")
    logger.info("   • Comprehensive logging")
    logger.info(f"   • {SEARCH_CONFIG['source_data']['total_products']} products indexed")
    logger.info(f"   • {len(SEARCH_CONFIG['brands']['list'])} brands")
    logger.info(f"   • {len(SEARCH_CONFIG['categories']['patterns'])} categories")
    logger.info("🌐 URL: http://localhost:8080")
    logger.info("📝 Logs: search_logs.txt")
    logger.info("="*70 + "\n")
    
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
