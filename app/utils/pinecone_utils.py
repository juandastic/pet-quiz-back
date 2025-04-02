import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def search_products(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    logger.info(f"Searching for products with query: '{query}', top_k={top_k}")

    if not query or query.strip() == "":
        logger.warning("Empty query provided to search_products")
        return []

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if not pc or not os.getenv("PINECONE_API_KEY"):
            logger.error("Pinecone API key not found or invalid")
            return []

        index = pc.Index("products-index")
        namespace = ""
        logger.info(f"Connected to Pinecone index 'products-index'")

        try:
            logger.info(f"Searching with integrated embedding")
            results = index.search(
                namespace=namespace,
                query={
                    "inputs": {"text": query},
                    "top_k": top_k
                },
                fields=["name", "price", "image_url", "product_link", "search_query", "text"]
            )
            logger.info(f"Search completed successfully")
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return []
    except Exception as e:
        logger.error(f"Unexpected error in search_products: {str(e)}")
        return []

    products = []

    # Guard against None results
    if results is None:
        logger.warning("Search returned None results")
        return products

    try:
        if not hasattr(results, 'result') or not hasattr(results.result, 'hits'):
            logger.warning(f"Unexpected response format from Pinecone: {results}")
            return products

        hits = results.result.hits
        if not hits:
            logger.info("No hits found in search results")
            return products

        logger.info(f"Processing {len(hits)} hits from search")
        for hit in hits:
            try:
                fields = hit.get('fields', {})
                product = {
                    "id": hit.get('_id', ''),
                    "score": hit.get('_score', 0.0),
                    "name": fields.get('name', ''),
                    "price": fields.get('price', 0.0),
                    "image_url": fields.get('image_url', ''),
                    "product_link": fields.get('product_link', ''),
                    "description": fields.get('text', ''),
                    "search_query": fields.get('search_query', '')
                }
                products.append(product)
            except Exception as e:
                logger.error(f"Error processing hit: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing search results: {str(e)}")

    logger.info(f"Returning {len(products)} products from search")
    return products
