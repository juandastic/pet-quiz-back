import os
from typing import Dict, List, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from app.utils.pinecone_utils import search_products

class AgentState(TypedDict):
    quiz_data: str
    summary_es: str
    summary_en: str
    products: List[Dict[str, Any]]
    explanation: str

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
    
    model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    chain = prompt | model
    
    def summarize(state: AgentState) -> AgentState:
        result = chain.invoke({"quiz_data": state["quiz_data"]})
        parsed_result = eval(result.content)
        state["summary_es"] = parsed_result["summary_es"]
        state["summary_en"] = parsed_result["summary_en"]
        return state
    
    return summarize

def create_search_products_node():
    def search_for_products(state: AgentState) -> AgentState:
        query = state["summary_en"]
        search_results = search_products(query)
        state["products"] = search_results
        return state
    
    return search_for_products

def create_explanation_node():
    prompt = ChatPromptTemplate.from_template("""
Actúa como un experto en mascotas y asistente de compras. Basándote en las necesidades del usuario y los productos encontrados, crea una explicación personalizada de por qué estos productos son adecuados.

Necesidades del usuario (español):
{summary_es}

Necesidades del usuario (inglés):
{summary_en}

Productos encontrados:
{products}

Por favor escribe una explicación empática y detallada en español de por qué estos productos son buenos para las necesidades específicas de la mascota del usuario. Menciona características específicas de los productos que se alinean con las necesidades identificadas.
""")
    
    model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    chain = prompt | model
    
    def create_explanation(state: AgentState) -> AgentState:
        if not state["products"]:
            state["explanation"] = "No se encontraron productos que coincidan con tus necesidades."
            return state
        
        result = chain.invoke({
            "summary_es": state["summary_es"],
            "summary_en": state["summary_en"],
            "products": "\n".join([f"- {p['name']}: {p['description'][:200]}..." for p in state["products"]])
        })
        state["explanation"] = result.content
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
