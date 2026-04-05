import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-lens-index")

def debug_index():
    print("--- 🔍 Pinecone Index Debugger ---")
    
    # 1. Check Stats (Namespaces and Counts)
    stats = index.describe_index_stats()
    print("\n📊 Index Stats:")
    print(stats)
    
    # 2. Peek at 1 Vector in the state_laws namespace
    # This shows us exactly what your metadata keys are called
    print("\n👀 Peeking at metadata in 'state_laws'...")
    try:
        peek = index.query(
            namespace="state_laws",
            vector=[0.0] * 1024, # Dummy vector
            top_k=1,
            include_metadata=True
        )
        if peek['matches']:
            print("✅ Found a match! Here is the metadata structure:")
            print(peek['matches'][0]['metadata'])
        else:
            print("❌ No vectors found in 'state_laws' namespace.")
    except Exception as e:
        print(f"❌ Error peeking: {e}")

if __name__ == "__main__":
    debug_index()
