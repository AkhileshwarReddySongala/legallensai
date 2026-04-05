import os
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from dotenv import load_dotenv

load_dotenv()

Settings.embed_model = local
# 1. Initialize Connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("legal-lens-index")


# 2. Setup the Vector Store
# LlamaIndex will detect the 'llama-text-embed-v2' config on your index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

def legal_search(query_text, state_filter=None, namespace="state-laws"):
    """
    Performs a filtered RAG search.
    :param query_text: The user's natural language question.
    :param state_filter: (Optional) The specific state to lock the search to.
    :param namespace: The bucket to search ('state-laws' or 'local-docs').
    """
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

    # Build filters if a state is provided
    filters = None
    if state_filter:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="state", value=state_filter.title())]
        )

    # Configure the query engine
    # similarity_top_k=3 ensures we get enough context for a complex answer
    query_engine = index.as_query_engine(
        namespace=namespace,
        filters=filters,
        similarity_top_k=3
    )

    print(f"--- Executing LegalLens Search [{namespace}] ---")
    response = query_engine.query(query_text)
    
    return response

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Example 1: Searching the State Hub
    ca_answer = legal_search(
        "What is the notice period for a month-to-month lease termination?", 
        state_filter="California"
    )
    print(f"California Law says:\n{ca_answer}\n")

    # Example 2: Searching your uploaded PDFs
    pdf_answer = legal_search(
        "Does my contract mention any pet deposits?", 
        namespace="local-docs"
    )
    print(f"Your PDF says:\n{pdf_answer}")