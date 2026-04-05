import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict
import tempfile
import uvicorn

# Import the existing pipeline tools
from router_segmentation import route_and_segment
from expert_pipeline import process_segments_with_expert
from rules_engine import check_hard_rules
from auditor_engine import run_audit
from rag_search import search_law_library

app = FastAPI(title="LegalLens AI API")

# Allow CORS for the dashboard UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve the Stitch UI globally
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_dashboard():
    # Attempt to serve index.html from static folder
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "LegalLens API is running. Export your Stitch HTML into the /static folder as index.html to view the dashboard here."}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "LegalLens AI API is running"}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # 1. Save uploaded file to temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        # 2. Phase 2: Route & Segment using Qwen-VL
        document_data = route_and_segment(temp_file_path)
        if not document_data:
            raise HTTPException(status_code=500, detail="Failed to segment document or categorize it.")

        category = document_data.get("category", "UNKNOWN").upper()
        
        # 3. Phase 3 & 4: Specialised Experts & Parallel Checks
        # Pass segments to the specific LLaMa fine-tuned model
        expert_results = process_segments_with_expert(document_data)
        
        final_report = []
        total_flags = 0
        total_score = 0
        
        # Merge Expert Insights with Hard Rules Engine
        for result in expert_results:
            original = result["original_text"]
            expert_simplification = result["expert_simplification"]
            
            # Phase 4 Parallel Check: Hard Rules Engine
            rule_check = check_hard_rules(original, category)
            rule_flags = rule_check["flags"] if not rule_check["passed"] else []
            
            # Phase 4 Parallel Check: RAG Fact Finder
            rag_facts = "N/A"
            if category == "LEGAL":
                try:
                    # Searching Pinecone for exact law matches
                    rag_facts = search_law_library(original, "California", namespace="state-laws")
                except Exception as e:
                    rag_facts = f"Failed to retrieve facts: {e}"
            
            # Phase 5: Auditing
            try:
                # Provide all parallel info to Gemma Reasoning App
                audit_result = run_audit(
                    original_text=original, 
                    expert_output=expert_simplification, 
                    rag_facts=rag_facts, 
                    rule_flags=rule_flags
                )
                
                status = audit_result.get("risk_level", "Unknown")
                final_explanation = audit_result.get("final_explanation", expert_simplification)
                conf_score_raw = audit_result.get("confidence_score", 0.0)
                
                # Convert confidence score string to float safely
                if isinstance(conf_score_raw, str):
                    conf_score_str = ''.join(c for c in conf_score_raw if c.isdigit() or c == '.')
                    conf_score = float(conf_score_str) if conf_score_str else 0.0
                else:
                    conf_score = float(conf_score_raw)
                    
            except Exception as e:
                print(f"Audit generation failed: {e}")
                status = "Predatory/High-Risk" if rule_flags else "Caution"
                final_explanation = expert_simplification
                conf_score = 0.5
            
            # Convert status to Color for dashboard
            color = "Green"
            status_lower = status.lower()
            if "high" in status_lower or "predatory" in status_lower or "red" in status_lower:
                color = "Red"
                total_flags += 1
            elif "caution" in status_lower or "medium" in status_lower or "yellow" in status_lower:
                color = "Yellow"

            total_score += conf_score

            final_report.append({
                "original_text": original,
                "simplification": final_explanation,
                "status": status,
                "color": color,
                "confidence_score": conf_score,
                "hard_rule_flags": rule_flags,
                "rag_facts": rag_facts
            })
            
        overall_score = round((total_score / len(expert_results)) * 100, 1) if expert_results else 100
        
        # Override if total score is calculated but might be in 0.0-1.0 range initially
        if overall_score <= 1.0 and expert_results: 
            overall_score = round((total_score / len(expert_results)) * 100, 1)
        # Cap score for 100% just in case
        if overall_score > 100:
            if total_score <= len(expert_results):
                overall_score = round((total_score / len(expert_results)) * 100, 1)

        return {
            "category": category,
            "overall_score": overall_score,
            "flags_detected": total_flags,
            "report": final_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
