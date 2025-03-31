import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def search_products(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("products-index")
    namespace = ""
    
    try:
        # Try using the search method with text input for integrated embedding
        results = index.search(
            namespace=namespace,
            query={
                "inputs": {"text": query},
                "top_k": top_k
            },
            fields=["name", "price", "image_url", "product_link", "search_query", "text"]
        )
        print(f"Search completed using integrated embedding")
    except Exception as e:
        print(f"Error using integrated search: {e}")
        print("Falling back to standard search with embedding API")
        
        # Fallback to manual embedding and vector search
        embedding_model = "llama-text-embed-v2"
        embedding_api_url = "https://api.pinecone.io/embedding/v1/embed"
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Api-Key": os.getenv("PINECONE_API_KEY")
        }
        
        payload = {
            "model": embedding_model,
            "texts": [query]
        }
        
        import requests
        response = requests.post(embedding_api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Error getting embedding: {response.text}")
        
        result = response.json()
        query_embedding = result["embeddings"][0]
        
        # Search using the embedding vector
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
    
    products = []
    
    # Handle different result formats based on which method was used
    if hasattr(results, 'matches'):
        # Standard query result format
        for match in results.matches:
            product = {
                "id": match.id,
                "score": match.score,
                "name": match.metadata.get("name", ""),
                "price": match.metadata.get("price", 0.0),
                "image_url": match.metadata.get("image_url", ""),
                "product_link": match.metadata.get("product_link", ""),
                "description": match.metadata.get("text", ""), # Text might be stored here
                "search_query": match.metadata.get("search_query", "")
            }
            products.append(product)
    elif hasattr(results, 'records'):
        # New search API result format
        for record in results.records:
            product = {
                "id": record.get("_id", ""),
                "score": record.get("_score", 0.0),
                "name": record.get("name", ""),
                "price": record.get("price", 0.0),
                "image_url": record.get("image_url", ""),
                "product_link": record.get("product_link", ""),
                "description": record.get("text", ""),
                "search_query": record.get("search_query", "")
            }
            products.append(product)
    
    return products
