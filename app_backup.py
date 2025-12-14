"""Multimodal RAG Chatbot - Complete with Filters & Smart Questions"""
from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
import re
import traceback
from datetime import datetime
import uuid
import markdown

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure session
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', '7f862a8a79175f1baef6c83a00a109e717321cb9529b5e3182ea327fddc56a23')

# ============================================================
# SAFE GEMINI LOADING
# ============================================================
def load_gemini():
    try:
        import google.generativeai as genai
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        if not GOOGLE_API_KEY:
            print("❌ ERROR: GOOGLE_API_KEY not found!")
            return None
        print(f"✅ API Key found: {GOOGLE_API_KEY[:10]}...")
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        test_response = gemini_model.generate_content("Say 'Ready!'")
        print(f"✅ Gemini: {test_response.text}")
        return gemini_model
    except Exception as e:
        print(f"❌ Gemini failed: {e}")
        traceback.print_exc()
        return None

# ============================================================
# SAFE AI MODELS + VECTOR DB LOADING
# ============================================================
def load_ai_models():
    """Load AI models with detailed error reporting"""
    models = {}
    
    # Load BGE Model
    try:
        from FlagEmbedding import FlagModel
        print("🚀 Loading BGE model...")
        models["bge_model"] = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)
        print("✅ BGE model loaded")
    except Exception as e:
        print(f"❌ BGE model failed: {e}")
        traceback.print_exc()
        models["bge_model"] = None
    
    # Load ChromaDB Collections
    try:
        import chromadb
        print("🚀 Loading ChromaDB clients...")
        text_client = chromadb.PersistentClient(path="./chroma_db_bge")
        image_client = chromadb.PersistentClient(path="./chroma_db")
        models["text_collection"] = text_client.get_collection("product_text_embeddings")
        models["image_collection"] = image_client.get_collection("product_image_embeddings")
        print("✅ ChromaDB loaded")
    except Exception as e:
        print(f"❌ ChromaDB failed: {e}")
        traceback.print_exc()
        models["text_collection"] = None
        models["image_collection"] = None
    
    # Load Marqo/OpenCLIP
    try:
        import open_clip
        import torch
        print("🚀 Loading Marqo model...")
        marqo_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
        )
        marqo_tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-ecommerce-embeddings-L')
        marqo_model.eval()
        models["marqo_model"] = marqo_model
        models["marqo_tokenizer"] = marqo_tokenizer
        models["torch"] = torch
        print("✅ Marqo model loaded")
    except Exception as e:
        print(f"❌ Marqo model failed: {e}")
        traceback.print_exc()
        models["marqo_model"] = None
        models["marqo_tokenizer"] = None
        models["torch"] = None
    
    # Summary
    loaded = sum(1 for v in models.values() if v is not None)
    print(f"\n✅ Loaded {loaded}/{len(models)} model components")
    
    return models if loaded > 0 else None

# Load resources at module level
print("\n🔧 Configuring Gemini...")
gemini_model = load_gemini()
ai_models = load_ai_models()

# ============================================================
# FILTER EXTRACTION
# ============================================================

def extract_filters(query):
    """Extract filters from natural language query"""
    filters = {
        'price_max': None,
        'price_min': None,
        'wattage_min': None,
        'wattage_max': None,
        'features': [],
        'color': None,
        'brand': None
    }
    
    query_lower = query.lower()
    
    # Price filters
    price_patterns = [
        (r'under\s+₹?(\d+)', 'max'),
        (r'below\s+₹?(\d+)', 'max'),
        (r'less than\s+₹?(\d+)', 'max'),
        (r'above\s+₹?(\d+)', 'min'),
        (r'over\s+₹?(\d+)', 'min'),
        (r'more than\s+₹?(\d+)', 'min'),
        (r'between\s+₹?(\d+)\s+and\s+₹?(\d+)', 'range')
    ]
    
    for pattern, filter_type in price_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if filter_type == 'max':
                filters['price_max'] = int(match.group(1))
            elif filter_type == 'min':
                filters['price_min'] = int(match.group(1))
            elif filter_type == 'range':
                filters['price_min'] = int(match.group(1))
                filters['price_max'] = int(match.group(2))
            print(f"   💰 Price filter: {filter_type} = {match.groups()}")
    
    # Wattage filters
    wattage_patterns = [
        (r'(\d+)w\s+to\s+(\d+)w', 'range'),
        (r'above\s+(\d+)w', 'min'),
        (r'over\s+(\d+)w', 'min'),
        (r'(\d+)\+w', 'min'),
        (r'under\s+(\d+)w', 'max'),
        (r'below\s+(\d+)w', 'max'),
        (r'low\s+wattage', 'low'),
        (r'high\s+wattage', 'high')
    ]
    
    for pattern, filter_type in wattage_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if filter_type == 'range':
                filters['wattage_min'] = int(match.group(1))
                filters['wattage_max'] = int(match.group(2))
            elif filter_type == 'min':
                filters['wattage_min'] = int(match.group(1))
            elif filter_type == 'max':
                filters['wattage_max'] = int(match.group(1))
            elif filter_type == 'low':
                filters['wattage_max'] = 10
            elif filter_type == 'high':
                filters['wattage_min'] = 20
            print(f"   ⚡ Wattage filter: {filter_type}")
    
    # Feature keywords
    feature_keywords = {
        'battery': ['battery', 'backup', 'emergency', 'inverter'],
        'dimmable': ['dimmable', 'dimmer', 'adjustable brightness'],
        'energy_efficient': ['energy efficient', 'energy saving', 'saver', 'star rated'],
        'decorative': ['decorative', 'aesthetic', 'candle', 'filament', 'vintage'],
        'bright': ['bright', 'high brightness', 'high lumen']
    }
    
    for feature, keywords in feature_keywords.items():
        if any(kw in query_lower for kw in keywords):
            filters['features'].append(feature)
            print(f"   ✨ Feature filter: {feature}")
    
    # Color filters
    color_keywords = ['warm white', 'cool white', 'daylight', 'cdl', 'ww']
    for color in color_keywords:
        if color in query_lower:
            filters['color'] = color
            print(f"   🎨 Color filter: {color}")
    
    # Brand filter
    if 'philips' in query_lower:
        filters['brand'] = 'philips'
        print(f"   🏷️ Brand filter: Philips")
    
    return filters

def apply_filters(products, filters):
    """Apply extracted filters to product list"""
    filtered = products
    
    if filters['features']:
        temp_filtered = []
        for product in filtered:
            product_text = f"{product['name']} {product.get('description', '')}".lower()
            
            matches_features = False
            for feature in filters['features']:
                if feature == 'battery' and any(kw in product_text for kw in ['emergency', 'inverter', 'battery']):
                    matches_features = True
                elif feature == 'energy_efficient' and any(kw in product_text for kw in ['saver', 'star', 'efficient']):
                    matches_features = True
                elif feature == 'decorative' and any(kw in product_text for kw in ['deco', 'candle', 'filament']):
                    matches_features = True
                elif feature == 'bright' and any(kw in product_text for kw in ['high wattage', 'bright', 'stellarbright']):
                    matches_features = True
            
            if matches_features:
                temp_filtered.append(product)
        
        if temp_filtered:
            filtered = temp_filtered
            print(f"   ✅ Filtered by features: {len(filtered)} products remain")
    
    return filtered

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
        print(f"🆕 New conversation: {session['conversation_id'][:8]}")
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
        print(f"   🔗 Expanded: '{expanded_query}'")
        return expanded_query, True
    
    return query, False

# ============================================================
# RETRIEVAL - FIXED VERSION
# ============================================================

def calculate_name_similarity(query, product_name):
    """Calculate name similarity"""
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

def deduplicate_products(products):
    """Remove duplicates"""
    seen = set()
    unique = []
    
    for product in products:
        key = f"{product['name'].lower()}_{product['brand'].lower()}"
        if key not in seen:
            seen.add(key)
            unique.append(product)
    
    return unique

def filter_irrelevant_products(products, query):
    """Filter irrelevant products"""
    query_lower = query.lower()
    filtered = []
    
    incompatible = {
        'bedroom': ['emergency', 't-bulb', 't-beamer'],
        'decorative': ['emergency', 'stellarbright super'],
        'ambient': ['emergency', 'super high'],
    }
    
    for product in products:
        product_lower = product['name'].lower()
        is_incompatible = False
        
        for query_term, incompatible_terms in incompatible.items():
            if query_term in query_lower:
                if any(term in product_lower for term in incompatible_terms):
                    is_incompatible = True
                    break
        
        if not is_incompatible:
            filtered.append(product)
    
    return filtered if filtered else products

def retrieve_products(query, top_k=10):
    """Retrieve products from both text and image stores"""
    global ai_models
    
    if ai_models is None:
        print("❌ AI models not available")
        return []
    
    print(f"\n🔍 Query: '{query}'")
    
    bge_model = ai_models["bge_model"]
    text_collection = ai_models["text_collection"]
    image_collection = ai_models["image_collection"]
    marqo_model = ai_models["marqo_model"]
    marqo_tokenizer = ai_models["marqo_tokenizer"]
    torch = ai_models["torch"]
    
    # TEXT SEARCH
    print("   📝 Searching text...")
    text_embedding = bge_model.encode([query])[0].tolist()
    text_results = text_collection.query(
        query_embeddings=[text_embedding],
        n_results=top_k * 2,
        include=['metadatas', 'distances', 'documents']
    )
    
    # IMAGE SEARCH
    print("   📸 Searching images...")
    text_input = marqo_tokenizer([query])
    with torch.no_grad():
        query_feat = marqo_model.encode_text(text_input)
        query_feat /= query_feat.norm(dim=-1, keepdim=True)
    image_embedding = query_feat[0].tolist()
    
    image_results = image_collection.query(
        query_embeddings=[image_embedding],
        n_results=top_k * 2,
        include=['metadatas', 'distances', 'documents']
    )
    
    # MERGE
    products = {}
    
    # Process text results
    for i, metadata in enumerate(text_results['metadatas'][0]):
        row_id = metadata.get('row_id', f'text_{i}')
        products[row_id] = {
            'name': metadata.get('product_name', 'Unknown Product'),
            'brand': metadata.get('brand', 'Unknown'),
            'category': metadata.get('category', ''),
            'image_url': metadata.get('image_url', ''),
            'image_url_2': metadata.get('image_url_2', ''),
            'description': text_results['documents'][0][i][:200] if i < len(text_results['documents'][0]) else '',
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
                'name': metadata.get('product_name', 'Unknown Product'),
                'brand': metadata.get('brand', 'Unknown'),
                'category': metadata.get('category', ''),
                'image_url': metadata.get('image_url', ''),
                'image_url_2': '',
                'description': '',
                'row_id': row_id,
                'in_text': False,
                'in_image': True,
                'image_distance': image_results['distances'][0][i]
            }
    
    # SCORE
    for prod_id, product in products.items():
        score = 0.0
        
        name_match = calculate_name_similarity(query, product['name'])
        score += name_match * 100
        
        if product['in_text'] and product['in_image']:
            score += 50
            product['synchronized'] = True
        else:
            product['synchronized'] = False
        
        if product['in_text']:
            score += (1 / (1 + product['text_distance'])) * 20
        
        if product['in_image']:
            score += (1 / (1 + product['image_distance'])) * 30
        
        product['final_score'] = score
    
    sorted_products = sorted(products.values(), key=lambda x: x['final_score'], reverse=True)
    filtered_products = filter_irrelevant_products(sorted_products, query)
    unique_products = deduplicate_products(filtered_products)
    
    final_products = unique_products[:top_k]
    print(f"   ✅ Found {len(final_products)} products\n")
    
    return final_products

def generate_response_with_gemini(query, products, conversation_context, was_clarification=False):
    """Generate response with clarification awareness"""
    global gemini_model
    
    print(f"\n📤 Generating response...")
    
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
    
    prompt = f"""You are a professional Philips lighting assistant.

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
        print(f"✅ Generated ({len(response_text)} chars)")
        
        # Extract mentioned products
        mentioned_products = extract_strictly_mentioned_products(response_text, products)
        
        if not mentioned_products:
            print("⚠️ No products clearly mentioned, using top 3 synced")
            mentioned_products = [p for p in products if p.get('synchronized', False)][:3]
            if not mentioned_products:
                mentioned_products = products[:3]
        
        print(f"✅ Displaying {len(mentioned_products)} mentioned products")
        for p in mentioned_products:
            print(f"   • {p['name']}")
        
        return response_text, mentioned_products
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return "I found great products for you!", products[:3]

def extract_strictly_mentioned_products(response_text, all_products):
    """Extract products explicitly mentioned by name"""
    mentioned = []
    response_lower = response_text.lower()
    
    for product in all_products:
        product_name_lower = product['name'].lower()
        
        # Exact full name match (highest priority)
        if product_name_lower in response_lower:
            mentioned.append(product)
            print(f"   ✓ Mentioned: {product['name']}")
            continue
        
        # Word-by-word matching (stricter)
        product_words = set(re.findall(r'\w+', product_name_lower))
        product_words -= {'philips', 'led', 'bulb', 'lamp', 'light', 'by', 'and', 'the'}
        
        # Need at least 2 unique words for matching
        if len(product_words) >= 2:
            # Check if ALL significant words are present (not just 75%)
            words_found = sum(1 for word in product_words if f' {word} ' in f' {response_lower} ')
            match_ratio = words_found / len(product_words)
            
            # Require 100% match for products with 2 words, 80% for 3+ words
            threshold = 1.0 if len(product_words) == 2 else 0.8
            
            if match_ratio >= threshold:
                mentioned.append(product)
                print(f"   ✓ Partial match: {product['name']} ({match_ratio:.0%})")
    
    # Limit to maximum 4 products
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
    """Main chat endpoint with Smart Question Asking"""
    try:
        # Check if models are available
        if ai_models is None:
            return jsonify({
                'error': 'AI models not loaded',
                'response': '❌ Sorry, the AI models are still loading or failed to load. Please try again in a moment.',
                'products': [],
                'stats': {'total': 0, 'synchronized': 0, 'sync_percentage': 0}
            }), 503
        
        query = request.json.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Initialize conversation
        conversation_id = initialize_conversation()
        
        print(f"\n{'='*60}")
        print(f"💬 Conversation: {conversation_id[:8]}")
        print(f"{'='*60}")
        
        # DEBUG: Show session state
        print(f"📊 Session State:")
        print(f"   awaiting_clarification: {session.get('awaiting_clarification', False)}")
        print(f"   original_query: '{session.get('original_query', 'None')}'")
        print(f"   clarification_attempts: {session.get('clarification_attempts', 0)}")
        
        was_clarification = False
        
        # ============================================================
        # CHECK IF RESPONDING TO CLARIFICATION
        # ============================================================
        if session.get('awaiting_clarification', False) and session.get('original_query'):
            original = session.get('original_query', '')
            
            print(f"🔍 Checking clarification: current='{query}' vs original='{original}'")
            
            if query.lower() != original.lower() and len(original) > 0:
                print(f"✅ Received clarification response")
                
                combined_query = f"{query} {original}"
                
                print(f"   Original: '{original}'")
                print(f"   Clarification: '{query}'")
                print(f"   Combined: '{combined_query}'")
                
                session['awaiting_clarification'] = False
                session['original_query'] = None
                was_clarification = True
                session.modified = True
                
                query = combined_query
            else:
                print(f"⚠️ Same query or invalid, treating as new query")
                session['awaiting_clarification'] = False
                session['original_query'] = None
                session['clarification_attempts'] = 0
                session.modified = True
        
        # ============================================================
        # CHECK FOR AMBIGUOUS QUERY (NEW VAGUE QUERY)
        # ============================================================
        if not was_clarification:
            ambiguous_patterns = ['i need', 'show me', 'want', 'looking for', 'lights', 'bulbs', 'give me', 'find me']
            is_ambiguous = any(pattern in query.lower() for pattern in ambiguous_patterns) and len(query.split()) < 5
            
            max_attempts = 2
            
            if is_ambiguous and session.get('clarification_attempts', 0) < max_attempts:
                print(f"❓ Ambiguous query detected - asking clarifying questions (Attempt {session.get('clarification_attempts', 0) + 1}/{max_attempts})")
                
                session['clarification_attempts'] = session.get('clarification_attempts', 0) + 1
                session['awaiting_clarification'] = True
                session['original_query'] = query
                session.modified = True
                
                clarification_response = """I'd be happy to help you find the perfect lighting! To give you the best recommendations, could you tell me:

🏠 **Room Type?** (bedroom, living room, kitchen, bathroom, outdoor)
💡 **Purpose?** (general lighting, task lighting, decorative, emergency backup)
✨ **Style Preference?** (modern, traditional, energy-efficient)
💰 **Budget Range?** (if any)

Just describe what you're looking for and I'll find the best options!"""
                
                # Convert markdown for clarification response too
                html_clarification = markdown.markdown(clarification_response, extensions=['nl2br', 'sane_lists'])
                
                return jsonify({
                    'query': query,
                    'response': html_clarification,
                    'products': [],
                    'is_clarification': True,
                    'stats': {'total': 0, 'synchronized': 0, 'sync_percentage': 0}
                })
            
            elif is_ambiguous and session.get('clarification_attempts', 0) >= max_attempts:
                print(f"⚠️ Max clarification attempts reached ({max_attempts}). Proceeding with best-effort search...")
                session['clarification_attempts'] = 0
                session['awaiting_clarification'] = False
                session.modified = True
        
        # ============================================================
        # CONTINUE WITH NORMAL SEARCH
        # ============================================================
        print(f"🔍 Searching for: '{query}'")
        
        filters = extract_filters(query)
        if any([filters['price_max'], filters['wattage_min'], filters['features']]):
            print(f"📊 Filters: {filters}")
        
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
        
        if any([filters['price_max'], filters['wattage_min'], filters['features']]):
            all_products = apply_filters(all_products, filters)
            print(f"   ✅ After filtering: {len(all_products)} products")
        
        ai_response, display_products = generate_response_with_gemini(
            query, 
            all_products, 
            conversation_context,
            was_clarification=was_clarification
        )
        
        add_to_conversation(query, ai_response, display_products)
        
        # Convert Markdown to HTML for proper rendering
        html_response = markdown.markdown(ai_response, extensions=['nl2br', 'sane_lists'])
        
        if was_clarification:
            session['clarification_attempts'] = 0
            session.modified = True
        
        sync_count = sum(1 for p in display_products if p.get('synchronized', False))
        sync_pct = (sync_count / len(display_products) * 100) if display_products else 0
        
        return jsonify({
            'query': query,
            'response': html_response,
            'products': display_products,
            'is_clarification': False,
            'filters_applied': filters if any([filters['price_max'], filters['wattage_min'], filters['features']]) else None,
            'stats': {
                'total': len(display_products),
                'synchronized': sync_count,
                'sync_percentage': round(sync_pct, 0)
            }
        })
        
    except Exception as e:
        print(f"❌ ERROR in /chat: {str(e)}")
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
    print("\n" + "="*60)
    print("🤖 COMPLETE MULTIMODAL RAG CHATBOT")
    print("="*60)
    print("✨ Features:")
    print("   • Conversation memory with context")
    print("   • Smart filters (price, wattage, features)")
    print("   • Clarifying questions for vague queries")
    print("   • Clarification attempt limits (2 max)")
    print("   • Strict text-image synchronization")
    print("   • Deduplication & relevance filtering")
    print("🌐 URL: http://localhost:8080")
    print("="*60 + "\n")
    
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
