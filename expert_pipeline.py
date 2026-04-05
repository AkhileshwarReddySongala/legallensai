import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.getenv("FEATHERLESS_API_KEY")
)

# 1. Replace these with your actual fine-tuned model names on Featherless
LEGAL_EXPERT_MODEL = "Akhil-reddy/Meta-Llma-legal-lens-500" 
MEDICAL_EXPERT_MODEL = "your-username/lingolink-medical-expert-v1"

def process_segments_with_expert(document_data):
    category = document_data["category"].upper()
    segments = document_data["segments"]
    
    # 2. The Router Logic: Pick the right brain based on Qwen-VL's output
    if category == "LEGAL":
        active_model = LEGAL_EXPERT_MODEL
        instruction = "Review the document clause and extract the risks in simple English."
    elif category == "MEDICAL":
        active_model = MEDICAL_EXPERT_MODEL
        instruction = "Explain this medical text in simple terms and flag any risks."
    else:
        print("Unknown category.")
        return

    print(f"\n🧠 Routing to {category} Expert Model: {active_model}...\n")
    
    expert_results = []

    # 3. The Expert Loop: Process each segment one by one
    for i, segment in enumerate(segments, 1):
        # We skip tiny segments (like "Make:" or "Model:") to save API calls
        if len(segment["content"]) < 15:
            continue
            
        print(f"Processing Segment {i}: {segment['title']}...")
        
        # Format this exactly like your training data (Instruction + Input)
        prompt = f"Instruction: {instruction}\n\nInput: {segment['content']}"
        
        try:
            response = client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": "You are LingoLink's specialized translation agent. Output only the requested Explanation, Risk Level, and Reason format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2 # Low temperature so it sticks to your fine-tuned format
            )
            
            # This should output the * Explanation: ... Risk Level: ... format from your dataset!
            expert_output = response.choices[0].message.content.strip()
            
            print(f"\n{expert_output}\n")
            print("-" * 50)
            
            # Save it for the next phase (The Auditor)
            expert_results.append({
                "original_text": segment["content"],
                "expert_simplification": expert_output
            })
            
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            
    return expert_results

# --- How to connect it to your previous script ---
# If you run this, pass the 'result' JSON from the previous Qwen-VL step into this function:
#
# Ifmock_data_from_previous_step = {
# If  "category": "Legal",
# If  "segments": [
# If       {"title": "Segment 11", "content": "OTHER THAN THE SELLER'S WARRANTY OF OWNERSHIP STATED ABOVE, THE BUYER TAKES THE VEHICLE AS-IS WITH ALL FAULTS..."}
# If  ]
# If }
# final_results = process_segments_with_expert(mock_data_from_previous_step)