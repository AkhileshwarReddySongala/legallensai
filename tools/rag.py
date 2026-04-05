import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-lens-index")

def legal_search(query_text, state_filter=None, namespace="state-laws", top_k=3):
    """
    Search a Pinecone integrated-embedding index using raw text.
    Pinecone embeds the query server-side.
    """
    search_kwargs = {
        "namespace": namespace,
        "query": {
            "inputs": {"text": query_text},
            "top_k": top_k,
        },
        "fields": ["text", "state", "source", "title"],
    }

    if state_filter:
        search_kwargs["query"]["filter"] = {
            "state": {"$eq": state_filter.title()}
        }

    results = index.search(**search_kwargs)
    return results


if __name__ == "__main__":
    ca_results = legal_search(
        "What is the notice period for a month-to-month lease termination?",
        state_filter="California",
        namespace="state-laws",
    )
    print(ca_results)

    pdf_results = legal_search(
        "Does my contract mention any pet deposits?",
        namespace="local-docs",
    )
    print(pdf_results)