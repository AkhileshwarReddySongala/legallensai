import os
from pinecone import Pinecone
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import TokenTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("legal-lens-index")

# 2. Setup Tools
reader = PyMuPDFReader()
splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)

def ingest_local_pdfs(folder_path):
    # Get all PDF files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    for file_name in files:
        print(f"--- Processing {file_name} ---")
        file_path = os.path.join(folder_path, file_name)
        
        
        # Load the PDF
        documents = reader.load(file_path=file_path)
        records_to_upsert = []
        
        for i, doc in enumerate(documents):
            # Split each page into safe chunks
            nodes = splitter.get_nodes_from_documents([doc])
            
            for j, node in enumerate(nodes):
                records_to_upsert.append({
                    "id": f"pdf-{file_name}-pg{i}-chunk-{j}",
                    "text": node.text,
                    "filename": file_name,
                    "jurisdiction": "Local/Private",
                    "source": "User Upload"
                })
        
        # Upsert to a separate namespace to keep things clean
        if records_to_upsert:
            pinecone_index.upsert_records(
                namespace="local-docs", 
                records=records_to_upsert
            )
            print(f"Successfully indexed {file_name}\n")

if __name__ == "__main__":
    # Create the directory if it doesn't exist and drop your PDFs there
    pdf_folder = "./law_docs"
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"Created {pdf_folder} folder. Put your PDFs there and run again!")
    else:
        ingest_local_pdfs(pdf_folder)