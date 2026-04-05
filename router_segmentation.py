import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize Featherless Client
client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.getenv("FEATHERLESS_API_KEY")
)

def encode_file_to_base64_image(file_path):
    if file_path.lower().endswith('.pdf'):
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        page = doc.load_page(0)  # load the first page
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("jpeg")
        return base64.b64encode(img_bytes).decode('utf-8')
    else:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def route_and_segment(file_path):
    print("Scanning and segmenting document with Qwen-VL...")
    base64_image = encode_file_to_base64_image(file_path)

    # The prompt forces Qwen-VL to be a JSON extraction engine
    system_prompt = """
    You are an expert document analysis AI. Analyze the provided image.
    1. Categorize it as either "Legal" or "Medical".
    2. Read the text and segment it into logical chunks based on layout (e.g., separate clauses, paragraphs, or bullet points).
    3. Return ONLY a valid JSON object. Do not include markdown blocks like ```json.
    
    Format:
    {
        "category": "Legal",
        "segments": [
            {"title": "Header or Subject", "content": "Exact text of the paragraph or clause..."}
        ]
    }
    """

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-30B-A3B-Instruct", # Verify this exact string in Featherless catalog
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0.0 # Strict adherence to the JSON format
    )
    
    # Parse the JSON response
    raw_output = response.choices[0].message.content.strip()
    
    # Clean up markdown if the model accidentally adds it
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:-3]
        
    try:
        document_data = json.loads(raw_output)
        return document_data
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        print("Raw output:", raw_output)
        return None

# Test it!
if __name__ == "__main__":
    result = route_and_segment(r"C:\akhil\legallensai\BILL OF SALE OF MOTOR VEHICLE.pdf")
    
    if result:
        print("\n✅ MULTIMODAL ROUTING SUCCESSFUL\n")
        print(f"Document Category: {result['category'].upper()}\n")
        print("--- Extracted Segments ---")
        
        for i, segment in enumerate(result['segments'], 1):
            print(f"\nSegment {i}: {segment['title']}")
            print(f"Text: {segment['content'][:100]}...") # Printing just the first 100 chars to keep the terminal clean