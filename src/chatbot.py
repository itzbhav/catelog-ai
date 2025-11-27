from src.vector_store import VectorStore
from src.retrieval import multimodal_retrieval
from bge_client import BGEClient  # hypothetical BGE client

text_store = VectorStore("text_vectors")
image_store = VectorStore("image_vectors")
text_store.load()
image_store.load()
bge = BGEClient()

def answer_query(user_query):
    query_vector = bge.embed(user_query)
    results = multimodal_retrieval(query_vector, text_store, image_store)

    response = ""
    images = []
    for item in results:
        response += f"Product: {item['text']['product_description']}\n"
        images.append(item['image'])

    return response, images
