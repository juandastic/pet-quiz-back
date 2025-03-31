# Pet Quiz Backend

A FastAPI backend application for indexing pet product data in Pinecone and providing personalized product recommendations based on user quiz responses.

## Features

- Data indexing script for Amazon pet toys dataset using Pinecone's llama-text-embed-v2 embedding model
- FastAPI web server with LangGraph-powered recommendation agent
- Simple, clean architecture with clear separation of concerns

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - MacOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `example.env` to `.env` and add your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Data Indexing

To index the product data in Pinecone:

```bash
python run.py index
```

### Starting the Server

To start the FastAPI server:

```bash
python run.py serve
```

The server will be available at http://localhost:8000.

## API Endpoints

- `GET /`: Health check endpoint
- `POST /api/recommend`: Submit quiz responses and get product recommendations

### Example Request to /api/recommend

```json
{
  "responses": {
    "¿Qué tipo de mascota tienes?": "Perro",
    "¿Qué raza es tu mascota?": "Labrador",
    "¿Cuál es la edad de tu mascota?": "2 años",
    "¿Tu mascota es muy activa?": "Sí, muy activa"
  }
}
```
