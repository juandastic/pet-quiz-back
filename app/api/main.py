from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import logging
import traceback
from dotenv import load_dotenv

from app.api.recommendation_agent import create_pet_recommendation_graph
from app.utils.pinecone_utils import search_products

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Pet Quiz API")

# Add exception handler for better error logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled error: {str(exc)}"
    logger.error(f"{error_msg}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg, "traceback": traceback.format_exc()}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizResponse(BaseModel):
    formatted_quiz: str

class RecommendationResponse(BaseModel):
    summary_es: str
    summary_en: str
    products: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {"message": "Pet Quiz API is running"}

@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(quiz_data: QuizResponse):
    try:
        logger.info("Received recommendation request")

        if not quiz_data.formatted_quiz:
            logger.warning("Empty quiz data received")
            raise HTTPException(status_code=400, detail="Formatted quiz data is required")

        logger.info(f"Processing quiz data: {quiz_data.formatted_quiz[:100]}...")

        # Create and run the recommendation graph
        pet_recommendation_graph = create_pet_recommendation_graph()
        result = pet_recommendation_graph.invoke({"quiz_data": quiz_data.formatted_quiz})

        logger.info(f"Recommendation graph result keys: {result.keys() if result else 'None'}")

        if not result or "products" not in result:
            logger.warning("No products found in recommendation result")
            return RecommendationResponse(
                summary_es=result.get("summary_es", "No se pudo generar un resumen"),
                summary_en=result.get("summary_en", "Could not generate a summary"),
                products=[]
            )

        logger.info(f"Returning {len(result['products'])} products")
        return RecommendationResponse(
            summary_es=result["summary_es"],
            summary_en=result["summary_en"],
            products=result["products"]
        )

    except Exception as e:
        error_msg = f"Error processing recommendation: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
