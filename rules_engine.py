import re

def check_hard_rules(text, category):
    """
    A deterministic logic gate that checks for illegal or highly predatory 
    phrases that should ALWAYS trigger a red flag, regardless of what the AI says.
    """
    text_lower = text.lower()
    triggered_rules = []

    # --- LEGAL HARD RULES ---
    if category == "LEGAL":
        legal_red_flags = {
            "Arbitration Trap": ["binding arbitration", "waive right to jury", "class action waiver"],
            "Liability Shield": ["as-is", "as is", "hold harmless", "indemnify"],
            "Financial Trap": ["non-refundable deposit", "accelerated rent", "liquidated damages"],
            "Sudden Termination": ["terminate without cause", "immediate eviction"]
        }
        
        for rule_name, phrases in legal_red_flags.items():
            for phrase in phrases:
                if phrase in text_lower:
                    triggered_rules.append(f"Hard Rule Triggered ({rule_name}): Found phrase '{phrase}'")

    # --- MEDICAL HARD RULES ---
    elif category == "MEDICAL":
        medical_red_flags = {
            "Blanket Consent": ["unforeseen circumstances", "additional procedures at discretion", "any and all treatments"],
            "Financial Liability": ["out-of-pocket", "patient responsible for all costs", "not covered by insurance"],
            "Data Privacy Waiver": ["share with third parties", "marketing purposes", "commercial use"]
        }
        
        for rule_name, phrases in medical_red_flags.items():
            for phrase in phrases:
                if phrase in text_lower:
                    triggered_rules.append(f"Hard Rule Triggered ({rule_name}): Found phrase '{phrase}'")

    # Return results
    if triggered_rules:
        return {
            "passed": False,
            "flags": triggered_rules
        }
    else:
        return {
            "passed": True,
            "flags": ["No hard rules violated."]
        }

# --- Test it! ---
if __name__ == "__main__":
    test_clause = "OTHER THAN THE SELLER'S WARRANTY OF OWNERSHIP STATED ABOVE, THE BUYER TAKES THE VEHICLE WITH AS-IS ALL FAULTS..."
    
    print("Testing Rule Engine...")
    result = check_hard_rules(test_clause, "LEGAL")
    
    if not result["passed"]:
        print("🚨 RED FLAGS DETECTED BY GUARDRAIL 🚨")
        for flag in result["flags"]:
            print("-", flag)
    else:
        print("✅ Clause passed hard rules.")