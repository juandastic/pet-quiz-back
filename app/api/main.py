from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

from app.api.recommendation_agent import create_pet_recommendation_graph
from app.utils.pinecone_utils import search_products

load_dotenv()

app = FastAPI(title="Pet Quiz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizResponse(BaseModel):
    responses: Dict[str, str]

class RecommendationResponse(BaseModel):
    summary_es: str
    summary_en: str
    products: List[Dict[str, Any]]
    explanation: str

@app.get("/")
async def root():
    return {"message": "Pet Quiz API is running"}

@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(quiz_data: QuizResponse):
    try:
        if not quiz_data.responses:
            raise HTTPException(status_code=400, detail="Quiz responses are required")
        
        # Format the quiz responses
        formatted_quiz = "\n".join([f"Pregunta: {q}\nRespuesta: {a}" for q, a in quiz_data.responses.items()])
        
        # Create and run the recommendation graph
        pet_recommendation_graph = create_pet_recommendation_graph()
        result = pet_recommendation_graph.invoke({"quiz_data": formatted_quiz})
        
        if not result or "products" not in result:
            return RecommendationResponse(
                summary_es=result.get("summary_es", "No se pudo generar un resumen"),
                summary_en=result.get("summary_en", "Could not generate a summary"),
                products=[],
                explanation="No se encontraron productos que coincidan con tus necesidades."
            )
        
        return RecommendationResponse(
            summary_es=result["summary_es"],
            summary_en=result["summary_en"],
            products=result["products"],
            explanation=result["explanation"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing recommendation: {str(e)}")
