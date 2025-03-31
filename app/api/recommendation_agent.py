import os
import json
import logging
import re
from typing import Dict, List, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
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
    model = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

    def create_explanation(state: AgentState) -> AgentState:
        try:
            if not state["products"]:
                logger.info("No products found, skipping explanations")
                return state

            logger.info(f"Generating explanations for {len(state['products'])} products in a single call")

            # Prepare simplified product data for the prompt
            products_for_prompt = []
            for product in state["products"]:
                products_for_prompt.append({
                    "id": product["id"],
                    "name": product["name"],
                    "description": product["description"][:200]
                })

            # Create the prompt directly as a message
            system_message = """Actúa como un experto en mascotas. Genera explicaciones breves de por qué cada producto satisface las necesidades específicas del usuario.

                Instrucciones:
                1. Para cada producto en la lista, crea una explicación corta y concisa (máximo 2 frases).
                2. Enfócate en por qué el producto es adecuado para las necesidades específicas de la mascota.
                3. Menciona solo las características más relevantes que se alinean con las necesidades.
                4. Responde con un JSON que contenga el ID del producto y su explicación.
            """

            user_message = f"""Necesidades del usuario (español):
                {state['summary_es']}

                Necesidades del usuario (inglés):
                {state['summary_en']}

                Productos:
                {json.dumps(products_for_prompt, ensure_ascii=False)}

                Genera una explicación concisa para cada producto y devuelve un JSON con este formato:
                [{{"id": "id_del_producto", "explanation": "explicación_concisa"}}]
            """

            # Make a single call to the model with all products
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            result = model.invoke(messages)

            # Parse the response using LangChain's JSON parser
            try:
                # Extract JSON from potential markdown code blocks
                parser = JsonOutputParser()
                parsed_content = parser.parse(result.content)

                # Create a dictionary for quick lookup
                explanation_dict = {item["id"]: item["explanation"] for item in parsed_content}

                # Add explanations to products and remove descriptions
                products_with_explanations = []
                for product in state["products"]:
                    product_copy = {k: v for k, v in product.items() if k != "description"}
                    product_copy["explanation"] = explanation_dict.get(
                        product["id"], "No se pudo generar una explicación para este producto."
                    )
                    products_with_explanations.append(product_copy)

                state["products"] = products_with_explanations
            except Exception as e:
                logger.error(f"Failed to parse model response: {str(e)}\nResponse content: {result.content}")
                # Fallback: keep products without descriptions removed
                products_with_explanations = []
                for product in state["products"]:
                    product_copy = {k: v for k, v in product.items() if k != "description"}
                    products_with_explanations.append(product_copy)
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
