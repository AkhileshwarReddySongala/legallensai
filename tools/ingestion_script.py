import os
from pinecone import Pinecone
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Connections
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("legal-lens-index")

# 2. Define the States to start with (The "Hubs")
states = ["alabama", "alaska", "arizona", "arkansas", "colorado", "connecticut", 
    "delaware", "florida", "georgia", "hawaii", "idaho", "indiana", "iowa", 
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", 
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new-hampshire", 
    "new-jersey", "new-mexico", "north-carolina", "north-dakota", "ohio", 
    "oklahoma", "oregon", "pennsylvania", "rhode-island", "south-carolina", "south-dakota", 
    "tennessee", "utah", "vermont", "virginia", "washington", "west-virginia", 
    "wisconsin", "wyoming"]
base_url = "https://www.nolo.com/legal-encyclopedia/overview-landlord-tenant-laws-{}.html"
urls = [base_url.format(s) for s in states]
splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)

def run_ingestion(url_list, batch_size=5):
    loader = SimpleWebPageReader(html_to_text=True)
    
    for i in range(0, len(url_list), batch_size):
        batch_urls = url_list[i:i + batch_size]
        print(f"--- Processing Batch {i//batch_size + 1} ({len(batch_urls)} states) ---")
        
        try:
            # Scrape the batch
            documents = loader.load_data(urls=batch_urls)
            records_to_upsert = []
            
            for doc, url in zip(documents, batch_urls):
                # Format Metadata
                state_slug = url.split("-")[-1].replace(".html", "")
                state_name = state_slug.replace("-", " ").title()
                
                # Split the large legal text into smaller chunks
                nodes = splitter.get_nodes_from_documents([doc])
                print(f"  > {state_name}: Split into {len(nodes)} chunks")
                
                for j, node in enumerate(nodes):
                    records_to_upsert.append({
                        "id": f"state-{state_slug}-chunk-{j}",
                        "text": node.text, # Matches your Pinecone Field Map
                        "state": state_name,
                        "source": "Nolo Legal Overview",
                        "jurisdiction": "State"
                    })
            
            # 4. Upsert to Pinecone
            if records_to_upsert:
                # We use upsert_records for Integrated Inference (NVIDIA)
                pinecone_index.upsert_records(
                    namespace="state-laws", 
                    records=records_to_upsert
                )
                print(f"  Successfully indexed Batch {i//batch_size + 1}\n")
            
        except Exception as e:
            print(f"  Error processing batch starting with {batch_urls[0]}: {e}")

if __name__ == "__main__":
    print("Starting LegalLens AI Ingestion...")
    run_ingestion(urls)
    print("Full 50-state ingestion complete!")