import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

class PineconeIndexer:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "products-index"
        self.embedding_model = "llama-text-embed-v2"
        self.namespace = ""

    def create_index_if_not_exists(self):
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index: {self.index_name}")
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region="us-west-2",
                embed={
                    "model": self.embedding_model,
                    "field_map": {
                        "text": "text"  # Map the direct text field to be embedded
                    }
                }
            )
        self.index = self.pc.Index(self.index_name)
        print(f"Index {self.index_name} created or already exists", self.index)

    def index_products(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} products from CSV")

        batch_size = 50
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            self._process_batch(batch)
            print(f"Indexed batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")

    def _process_batch(self, batch):
        records = []

        for _, row in batch.iterrows():
            product_id = row["id"]
            description = row["description_keywords"]

            if pd.isna(description) or not description.strip():
                print(f"Skipping product {product_id} with empty description")
                continue

            record = {
                "_id": product_id,
                "text": description,  # Field that will be embedded
                "name": row["name"],
                "price": row["price"] if not pd.isna(row["price"]) else 0.0,
                "image_url": row["image_url"],
                "product_link": row["product_link"],
                "search_query": row["search_query"] if not pd.isna(row["search_query"]) else ""
            }
            records.append(record)

        if records:
            # Use upsert_records for indexes with integrated embedding
            try:
                self.index.upsert_records(namespace=self.namespace, records=records)
                print(f"Successfully upserted {len(records)} records")
            except Exception as e:
                print(f"Error upserting records: {e}")
                # Fallback to standard upsert if upsert_records is not available
                print("Falling back to standard upsert method")

                # We need to generate embeddings for the fallback method
                import requests

                # Process in smaller batches to avoid hitting limits
                max_batch = 20
                for i in range(0, len(records), max_batch):
                    batch_records = records[i:i + max_batch]
                    texts = [r["text"] for r in batch_records]

                    # Get embeddings from Pinecone
                    embedding_api_url = "https://api.pinecone.io/embedding/v1/embed"
                    headers = {
                        "accept": "application/json",
                        "content-type": "application/json",
                        "Api-Key": os.getenv("PINECONE_API_KEY")
                    }
                    payload = {
                        "model": self.embedding_model,
                        "texts": texts
                    }

                    response = requests.post(embedding_api_url, headers=headers, json=payload)
                    if response.status_code != 200:
                        raise Exception(f"Error getting embeddings: {response.text}")

                    embeddings = response.json()["embeddings"]

                    # Create vectors with embeddings
                    vectors = []
                    for idx, record in enumerate(batch_records):
                        vector_id = record.pop("_id")
                        text = record.pop("text")

                        vector = {
                            "id": vector_id,
                            "values": embeddings[idx],
                            "metadata": record
                        }
                        # Add text to metadata
                        vector["metadata"]["text"] = text
                        vectors.append(vector)

                    # Upsert this batch
                    self.index.upsert(vectors=vectors, namespace=self.namespace)
                    print(f"Upserted {len(vectors)} vectors using fallback method")

def main():
    indexer = PineconeIndexer()
    indexer.create_index_if_not_exists()
    indexer.index_products("data/amazon_pet_toys_db.csv")
    print("Indexing complete!")

if __name__ == "__main__":
    main()
