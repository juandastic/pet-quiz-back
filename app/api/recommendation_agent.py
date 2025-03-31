import os
import json
import logging
from typing import Dict, List, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from app.utils.pinecone_utils import search_products

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    quiz_data: str
    summary_es: str
    summary_en: str
    products: List[Dict[str, Any]]

def create_summarize_node():
    prompt = ChatPromptTemplate.from_template("""
        Basándote en las siguientes respuestas del quiz de un usuario sobre su mascota:

        {quiz_data}

        1. Resume la necesidad principal del usuario en una frase o párrafo corto en español.
        2. Traduce ese resumen al inglés.

        Output en formato JSON:
        {{
        "summary_es": "...",
        "summary_en": "..."
        }}
    """)

    model = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    chain = prompt | model

    def summarize(state: AgentState) -> AgentState:
        try:
            logger.info(f"Processing quiz data: {state['quiz_data'][:100]}...")
            result = chain.invoke({"quiz_data": state["quiz_data"]})
            logger.info(f"LLM response: {result.content}")

            # Clean the response to ensure it's valid JSON
            content = result.content.strip()
            # Remove any markdown formatting if present
            if content.startswith('```json'):
                content = content.split('```json', 1)[1]
            if content.endswith('```'):
                content = content.rsplit('```', 1)[0]
            content = content.strip()

            parsed_result = json.loads(content)
            logger.info(f"Parsed result: {parsed_result}")

            state["summary_es"] = parsed_result["summary_es"]
            state["summary_en"] = parsed_result["summary_en"]
            return state
        except Exception as e:
            logger.error(f"Error in summarize node: {str(e)}")
            # Provide default values instead of failing
            state["summary_es"] = "No se pudo generar un resumen debido a un error."
            state["summary_en"] = "Could not generate a summary due to an error."
            return state

    return summarize

def create_search_products_node():
    def search_for_products(state: AgentState) -> AgentState:
        try:
            query = state["summary_en"]
            logger.info(f"Searching products with query: {query}")
            search_results = search_products(query)
            logger.info(f"Found {len(search_results)} products")
            state["products"] = search_results
            return state
        except Exception as e:
            logger.error(f"Error in search_products node: {str(e)}")
            state["products"] = []
            return state

    return search_for_products

def create_explanation_node():
    prompt = ChatPromptTemplate.from_template("""
        Actúa como un experto en mascotas y asistente de compras. Basándote en las necesidades del usuario, crea una explicación personalizada de por qué este producto específico es adecuado.

        Necesidades del usuario (español):
        {summary_es}

        Necesidades del usuario (inglés):
        {summary_en}

        Producto:
        Nombre: {product_name}
        Descripción: {product_description}

        Por favor escribe una explicación concisa y detallada en español de por qué este producto es bueno para las necesidades específicas de la mascota del usuario. Menciona características específicas del producto que se alinean con las necesidades identificadas.
    """)

    model = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    chain = prompt | model

    def create_explanation(state: AgentState) -> AgentState:
        try:
            if not state["products"]:
                logger.info("No products found, skipping explanations")
                return state

            logger.info(f"Generating explanations for {len(state['products'])} products")
            products_with_explanations = []

            for i, product in enumerate(state["products"]):
                logger.info(f"Processing product {i+1}/{len(state['products'])}: {product['name']}")
                result = chain.invoke({
                    "summary_es": state["summary_es"],
                    "summary_en": state["summary_en"],
                    "product_name": product["name"],
                    "product_description": product["description"][:200]
                })

                # Create a new product dict with the explanation
                product_with_explanation = product.copy()
                product_with_explanation["explanation"] = result.content
                products_with_explanations.append(product_with_explanation)

            state["products"] = products_with_explanations
            return state
        except Exception as e:
            logger.error(f"Error in create_explanation node: {str(e)}")
            # Return products without explanations rather than failing
            return state

    return create_explanation

def create_pet_recommendation_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", create_summarize_node())
    workflow.add_node("search_products", create_search_products_node())
    workflow.add_node("create_explanation", create_explanation_node())

    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "search_products")
    workflow.add_edge("search_products", "create_explanation")
    workflow.add_edge("create_explanation", END)

    return workflow.compile()
