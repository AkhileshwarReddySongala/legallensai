import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-lens-index")



def search_law_library(query, state_name, namespace="state_laws"):
    """
    Uses Pinecone's Integrated NVIDIA Inference to find the exact law.
    Locks the search to a specific state using metadata filters.
    """
    print(f"🔍 Searching {namespace} for '{state_name}' laws...")

    # Using Pinecone's Integrated Inference (NVIDIA llama-text-embed-v2)
    # This automatically handles the embedding of your query string
    results = index.query(
        namespace=namespace,
        top_k=3,
        vector=pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[query],
            parameters={"input_type": "query"}
        )[0].values,
        filter={"state": {"$eq": state_name}}, # METADATA FILTER: No hallucinations from other states!
        include_metadata=True
    )

    # Combine the top 3 matches into one "Fact Sheet"
    context_text = ""
    for match in results['matches']:
        context_text += f"\n---\nSource: {match['metadata'].get('source', 'Unknown')}\n"
        context_text += f"Text: {match['metadata']['text']}\n"

    return context_text

# --- Test the RAG Fact-Finder ---
if __name__ == "__main__":
    query = "The landlord may enter the premises at any time without notice for inspections."
    state = "California"

    facts = search_law_library(query, state)

    print("\n--- ✅ BACKEND READY FOR AUDITOR ---")
    print(f"Facts: {facts[:200]}...")
    print("Ready to send to the auditor.")