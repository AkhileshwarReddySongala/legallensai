import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-lens-index")

def legal_search(query_text, state_filter=None, namespace="state-laws", top_k=2):
    search_kwargs = {
        "namespace": namespace,
        "query": {"inputs": {"text": query_text}, "top_k": top_k},
        "fields": ["text", "state", "source", "title"],
    }
    if state_filter:
        search_kwargs["query"]["filter"] = {"state": {"$eq": state_filter.title()}}

    results = index.search(**search_kwargs)
    
    fact_sheet = ""
    for match in results.get('matches', []):
        metadata = match.get('metadata', {})
        text = metadata.get('text', "No text found.")
        source = metadata.get('source', "Official Records")
        fact_sheet += f"\n[SOURCE: {source}]\nLAW: {text}\n"
    
    return fact_sheet if fact_sheet else "No matching laws found in the library."