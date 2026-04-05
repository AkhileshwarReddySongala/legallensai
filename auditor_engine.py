import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url="https://api.featherless.ai/v1", api_key=os.getenv("FEATHERLESS_API_KEY"))

def run_audit(original_text, expert_output, rag_facts, rule_flags):
    system_prompt = """
    You are the LegalLens Auditor. Goal: 98% accuracy.
    Compare the Expert's simplification against the Laws and Rule Flags.
    Return ONLY a JSON object: {risk_level, final_explanation, confidence_score}.
    """
    payload = f"TEXT: {original_text}\nEXPERT: {expert_output}\nFACTS: {rag_facts}\nFLAGS: {rule_flags}"

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct", # Or google/gemma-4
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": payload}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(response.choices[0].message.content)