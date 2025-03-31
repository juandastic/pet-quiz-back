import argparse
import uvicorn
from app.indexing.pinecone_indexer import main as run_indexing

def main():
    parser = argparse.ArgumentParser(description="Pet Quiz Backend")
    parser.add_argument("action", choices=["index", "serve"], help="Action to perform")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server")

    args = parser.parse_args()

    if args.action == "index":
        run_indexing()
    elif args.action == "serve":
        uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
