"""
=============================================================
  Egyptian Legal Contract Clause — LLaMA Fine-Tune Converter
  Version: 1.0
  
  What this script does:
    1. Loads one or more filtered JSON dataset files
    2. Converts each entry to LLaMA instruction-response format
    3. Derives enhancement suggestions from legal_basis text
    4. Splits into train (85%) and test (15%) sets
    5. Saves everything + runs full verification checks
    6. Prints a detailed report at the end
=============================================================
"""

import json
import random
import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURATION — edit these paths only
# ─────────────────────────────────────────────

# List all your filtered JSON files here
INPUT_FILES = [
   r"C:\Users\Mohamed\Desktop\better_call_maro\generated_clauses\clauses_with_legal_basis.json"
]

OUTPUT_DIR = "llama_training_data"   # folder where outputs will be saved
TRAIN_RATIO = 0.85                   # 85% train, 15% test
RANDOM_SEED = 42                     # for reproducibility

# ─────────────────────────────────────────────
#  SYSTEM PROMPT — this is what the model learns
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert legal analyst specializing in Egyptian commercial law. 
Your task is to analyze contract clauses and determine whether they comply with Egyptian law.

For each clause you will:
1. Determine if the clause is legally VALID or INVALID
2. If INVALID: identify the specific violation type
3. If INVALID: explain which article of Egyptian law is violated and why
4. If INVALID: suggest a corrected version of the clause that would make it compliant

Be precise, structured, and base all analysis strictly on Egyptian civil and commercial law."""


# ─────────────────────────────────────────────
#  ENHANCEMENT SUGGESTION GENERATOR
#  Derives a fix suggestion from legal_basis text
# ─────────────────────────────────────────────

def derive_enhancement(violation_type: str, legal_basis: str, clause_text: str) -> str:
    """
    Derives an enhancement suggestion from the violation type and legal basis.
    This is rule-based so it stays grounded in your training data logic.
    """

    violation_lower = violation_type.lower()
    basis_lower = legal_basis.lower()

    # Extract article reference from legal_basis if present
    article_ref = ""
    import re
    article_match = re.search(r"article\s+\d+[a-z]?", legal_basis, re.IGNORECASE)
    if article_match:
        article_ref = article_match.group(0)

    # Map violation types to enhancement templates
    enhancement_map = {
        "dissolution_without_notice": (
            "Revise the clause to require prior written notice to the defaulting party "
            "before dissolution takes effect, unless notice is expressly waived in a "
            "specific written provision of the agreement."
        ),
        "exclusion_of_damages": (
            "Remove the exclusion of damages. The non-defaulting party must retain the "
            "statutory right to claim compensation for losses resulting from non-performance, "
            f"as required by {article_ref}."
        ),
        "dissolution_without_express_notice_waiver": (
            "If the parties wish to waive the notice requirement, this waiver must be "
            "explicitly and specifically stated in the contract. A general or implied waiver "
            "is insufficient under Egyptian law."
        ),
        "payment_decoupled_from_delivery": (
            "Link payment directly to delivery of the work. Payment should become due upon "
            f"formal delivery and acceptance of the completed work, per {article_ref}. "
            "If milestone payments are needed, each installment must be tied to a specific delivery event."
        ),
        "unlimited_post_delivery_liability": (
            "Cap the liability period to a maximum of ten years from delivery, consistent "
            f"with {article_ref}. Unlimited or excessively long liability periods exceeding "
            "the statutory framework are void."
        ),
        "inverted_document_hierarchy": (
            "Restore the correct document hierarchy. The main Contract body and its formal "
            "annexes must prevail over all incorporated documents such as technical proposals "
            f"and financial offers, per {article_ref}."
        ),
        "impermissible_interpretive_deviation": (
            "Remove any provision allowing reinterpretation of clear contract terms. "
            f"Per {article_ref}, where contract language is clear and unambiguous, "
            "it must be applied as written without deviation under the pretext of interpretation."
        ),
        "missing_essential_terms_in_preliminary_commitment": (
            "Specify all essential terms — including scope, price, deliverables, and timeline — "
            f"before the contract becomes binding, as required by {article_ref}. "
            "A binding commitment cannot be made without these terms being fully defined."
        ),
        "foreign_law_substitution": (
            "Replace any reference to foreign law with Egyptian law. All obligations under "
            f"this contract must be governed by Egyptian law per {article_ref}."
        ),
        "foreign_jurisdiction": (
            "Replace the foreign jurisdiction clause with Egyptian courts or Egypt-seated "
            f"arbitration. Per {article_ref}, Egyptian courts have mandatory jurisdiction "
            "over disputes of this nature."
        ),
        "tax_obligation_waiver": (
            "Remove the tax waiver. All applicable Egyptian taxes must be paid in accordance "
            f"with Egyptian tax law as required by {article_ref}. Tax obligations cannot be "
            "contractually eliminated."
        ),
        "tax_liability_reversal": (
            "Restore the correct tax liability allocation in accordance with Egyptian law. "
            f"Per {article_ref}, the employer bears the statutory tax payment obligation "
            "and this cannot be reversed by contract."
        ),
        "waiver_of_statutory_termination_right": (
            "Remove the termination waiver. Either party to this type of contract retains "
            f"the statutory right to terminate at any time per {article_ref}. "
            "This right cannot be irrevocably waived."
        ),
        "exclusion_of_statutory_termination_compensation": (
            "Restore the compensation obligation. Where termination occurs without prior "
            "notice or at a commercially inconvenient time, the terminating party must pay "
            f"compensation as required by {article_ref}."
        ),
        "exclusion_of_ten_year_structural_liability": (
            "Restore the full ten-year joint liability period for the architect and contractor "
            f"for structural defects and collapse of fixed installations, per {article_ref}. "
            "This period cannot be reduced by contract."
        ),
        "waiver_of_rescission_right": (
            "Restore the employer's right to seek rescission following defective performance "
            f"that cannot be rectified, per {article_ref}. This statutory remedy cannot be "
            "contractually eliminated."
        ),
        "waiver_of_right_to_substitute_completion": (
            "Restore the employer's right to engage a substitute contractor to complete the "
            "work at the first contractor's expense following failure to rectify defective "
            f"performance, per {article_ref}."
        ),
        "liability_exemption_for_fraud_and_gross_negligence": (
            "Remove the exemption from liability for fraud or gross negligence. Per Egyptian "
            "law, any clause exempting a party from liability for its own fraud or gross "
            "negligence, or that of persons it employs, is void."
        ),
        "liability_exemption_for_unlawful_acts": (
            "Remove the exemption from liability for unlawful acts entirely. Any clause "
            "exempting liability for unlawful acts is absolutely void under Egyptian law "
            "and cannot be cured by agreement."
        ),
        "exclusion_of_natural_consequence_damages": (
            "Remove the exclusion of natural-consequence damages. Compensation must include "
            "all losses and missed gains that are a natural and foreseeable consequence of "
            f"non-performance, per {article_ref}."
        ),
        "exclusion_of_moral_damages": (
            "Remove the exclusion of moral damages. Compensation under Egyptian law includes "
            f"moral damage per {article_ref}. A blanket waiver of moral damage rights "
            "is contrary to the statute."
        ),
        "compensation_without_notice_requirement": (
            "Restore the notice requirement before compensation becomes due. Per Egyptian law, "
            "compensation is not owed until the debtor has been formally served with notice, "
            "unless a valid statutory exception applies."
        ),
        "deferred_payment_despite_delivery": (
            "Remove the payment deferral. Remuneration must become due upon delivery of the "
            f"work per {article_ref}. Post-delivery withholding or deferral of payment "
            "is not permitted without a valid agreed and proportionate exception."
        ),
        "unjustified_refusal_of_delivery": (
            "Remove the clause enabling unjustified refusal of delivery. Once the contractor "
            "has completed the work and issued formal notification, the employer must take "
            f"delivery promptly per {article_ref}. Unjustified refusal triggers deemed delivery."
        ),
        "deemed_delivery_waiver": (
            "Restore the deemed delivery protection. Where the employer unjustifiably refuses "
            "delivery after formal notification, the work must be deemed delivered by operation "
            f"of law per {article_ref}. This protection cannot be contractually waived."
        ),
        "failure_to_be_in_writing": (
            "Execute this agreement as a formal written contract signed by authorized "
            f"representatives of both parties per {article_ref}. Oral or informal "
            "agreements do not satisfy the statutory writing requirement and are void."
        ),
        "restriction_on_importer_freedom": (
            "Remove the restriction on the technology importer's freedom. Per Egyptian "
            "Commercial Code, the importer retains the right to use, develop, define, "
            "and advertise the production. Such restrictions may be annulled by Egyptian courts."
        ),
        "waiver_of_statutory_renegotiation_right": (
            "Restore the five-year renegotiation and termination right. Either party to a "
            "technology transfer contract may request termination or renegotiation after five "
            "years if economic conditions require. This right cannot be contractually waived."
        ),
        "shortened_annulment_period": (
            "Replace the shortened annulment period with the statutory ninety-day period "
            "running from the date of notification of the award, as required by Egyptian "
            "Arbitration Law. Contractually shortening this period deprives parties of "
            "their statutory rights."
        ),
        "unauthorized_disclosure_of_undisclosed_information": (
            "Replace with a mutual confidentiality obligation prohibiting disclosure of "
            "undisclosed information to third parties without authorization. Per Egyptian law, "
            "unauthorized disclosure of undisclosed information constitutes unfair competition."
        ),
        "unfair_commercial_exploitation_of_undisclosed_information": (
            "Remove the authorization to use undisclosed information for competitive purposes. "
            "Using another party's undisclosed information to gain commercial advantage "
            "constitutes an act contrary to fair commercial practices under Egyptian law."
        ),
        "delayed_domicile_change_notification": (
            "Replace the delayed notification requirement with a prompt notification obligation "
            "(standard practice is forty-eight hours written notice). Domicile must be kept "
            f"current for valid service of legal documents per {article_ref}."
        ),
        "prohibition_on_domicile_change": (
            "Remove the prohibition on domicile changes. Parties must be permitted to update "
            "their domicile address with prompt written notice. Prohibiting address changes "
            f"undermines valid service of legal documents per {article_ref}."
        ),
    }

    # Try to match violation type to a template
    # Handle compound violation types (e.g. "dissolution_without_notice, exclusion_of_damages")
    violations = [v.strip() for v in violation_type.split(",")]
    suggestions = []

    for v in violations:
        v_lower = v.lower().strip()
        matched = False
        for key, suggestion in enhancement_map.items():
            if key in v_lower or v_lower in key:
                suggestions.append(suggestion)
                matched = True
                break
        if not matched:
            # Generic fallback using legal_basis
            suggestions.append(
                f"Revise this clause to comply with {article_ref if article_ref else 'applicable Egyptian law'}. "
                f"The current provision violates the statutory requirement. "
                f"Consult the relevant article to restructure the clause appropriately."
            )

    return " Additionally, ".join(suggestions)


# ─────────────────────────────────────────────
#  FORMAT A SINGLE ENTRY
# ─────────────────────────────────────────────

def format_entry(entry: dict, idx: int) -> dict:
    """
    Converts one JSON entry to LLaMA instruction-response format.
    Returns a dict with 'instruction', 'input', and 'output' fields.
    """

    clause_text = entry.get("text", "").strip()
    label = entry.get("label", -1)
    violation_type = entry.get("violation_type", "none").strip()
    legal_basis = entry.get("legal_basis", "").strip()

    # ── Validate required fields ──
    if not clause_text:
        raise ValueError(f"Entry {idx}: missing 'text' field")
    if label not in [0, 1]:
        raise ValueError(f"Entry {idx}: label must be 0 or 1, got '{label}'")
    if not legal_basis:
        raise ValueError(f"Entry {idx}: missing 'legal_basis' field")

    # ── Build the instruction ──
    instruction = (
        "Analyze the following contract clause under Egyptian commercial law. "
        "Determine whether it is legally valid or invalid. "
        "If invalid, identify the violation type, explain which law is violated and why, "
        "and provide a corrected version of the clause."
    )

    # ── Build the output based on label ──
    if label == 0:
        # Compliant clause
        output = (
            f"VERDICT: VALID\n\n"
            f"VIOLATION TYPE: None\n\n"
            f"LEGAL ANALYSIS: {legal_basis}\n\n"
            f"ENHANCEMENT: No changes required. This clause is compliant with Egyptian law."
        )
    else:
        # Violating clause
        enhancement = derive_enhancement(violation_type, legal_basis, clause_text)
        output = (
            f"VERDICT: INVALID\n\n"
            f"VIOLATION TYPE: {violation_type}\n\n"
            f"LEGAL ANALYSIS: {legal_basis}\n\n"
            f"ENHANCEMENT: {enhancement}"
        )

    return {
        "system": SYSTEM_PROMPT,
        "instruction": instruction,
        "input": clause_text,
        "output": output,
        # Keep original fields for traceability
        "_original_label": label,
        "_original_violation_type": violation_type,
        "_source_index": idx
    }


# ─────────────────────────────────────────────
#  LOAD AND MERGE ALL INPUT FILES
# ─────────────────────────────────────────────

def load_all_files(file_paths: list) -> list:
    all_entries = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"  [WARNING] File not found: {path} — skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  [WARNING] {path} is not a JSON array — skipping")
            continue
        print(f"  Loaded {len(data)} entries from {path}")
        all_entries.extend(data)
    return all_entries


# ─────────────────────────────────────────────
#  VERIFICATION CHECKS
# ─────────────────────────────────────────────

def run_verification(original: list, converted: list) -> dict:
    """
    Runs all verification checks and returns a report dict.
    """
    report = {
        "passed": True,
        "checks": []
    }

    def add_check(name, passed, detail=""):
        status = "✅ PASS" if passed else "❌ FAIL"
        report["checks"].append({"name": name, "status": status, "detail": detail})
        if not passed:
            report["passed"] = False

    # Check 1: Count match
    add_check(
        "Entry count matches",
        len(original) == len(converted),
        f"Original: {len(original)} | Converted: {len(converted)}"
    )

    # Check 2: No empty outputs
    empty_outputs = [i for i, e in enumerate(converted) if not e.get("output", "").strip()]
    add_check(
        "No empty outputs",
        len(empty_outputs) == 0,
        f"Empty outputs at indices: {empty_outputs[:10]}" if empty_outputs else "All outputs populated"
    )

    # Check 3: No empty inputs
    empty_inputs = [i for i, e in enumerate(converted) if not e.get("input", "").strip()]
    add_check(
        "No empty inputs",
        len(empty_inputs) == 0,
        f"Empty inputs at indices: {empty_inputs[:10]}" if empty_inputs else "All inputs populated"
    )

    # Check 4: Label 0 entries marked VALID
    label0_wrong = [
        i for i, e in enumerate(converted)
        if e.get("_original_label") == 0 and "VERDICT: VALID" not in e.get("output", "")
    ]
    add_check(
        "All label=0 entries marked VALID",
        len(label0_wrong) == 0,
        f"Wrong entries: {label0_wrong[:10]}" if label0_wrong else "All correct"
    )

    # Check 5: Label 1 entries marked INVALID
    label1_wrong = [
        i for i, e in enumerate(converted)
        if e.get("_original_label") == 1 and "VERDICT: INVALID" not in e.get("output", "")
    ]
    add_check(
        "All label=1 entries marked INVALID",
        len(label1_wrong) == 0,
        f"Wrong entries: {label1_wrong[:10]}" if label1_wrong else "All correct"
    )

    # Check 6: Label 0 entries have no violation type in output
    label0_has_violation = [
        i for i, e in enumerate(converted)
        if e.get("_original_label") == 0
        and "VIOLATION TYPE: None" not in e.get("output", "")
    ]
    add_check(
        "All label=0 entries have no violation type",
        len(label0_has_violation) == 0,
        f"Wrong entries: {label0_has_violation[:10]}" if label0_has_violation else "All correct"
    )

    # Check 7: Label 1 entries have violation type
    label1_no_violation = [
        i for i, e in enumerate(converted)
        if e.get("_original_label") == 1
        and e.get("_original_violation_type", "none").lower() == "none"
    ]
    add_check(
        "All label=1 entries have a violation type",
        len(label1_no_violation) == 0,
        f"Suspicious entries: {label1_no_violation[:10]}" if label1_no_violation else "All correct"
    )

    # Check 8: Legal basis present in all outputs
    no_legal_basis = [
        i for i, e in enumerate(converted)
        if "LEGAL ANALYSIS:" not in e.get("output", "")
    ]
    add_check(
        "Legal analysis present in all outputs",
        len(no_legal_basis) == 0,
        f"Missing at indices: {no_legal_basis[:10]}" if no_legal_basis else "All present"
    )

    # Check 9: Enhancement present in all outputs
    no_enhancement = [
        i for i, e in enumerate(converted)
        if "ENHANCEMENT:" not in e.get("output", "")
    ]
    add_check(
        "Enhancement present in all outputs",
        len(no_enhancement) == 0,
        f"Missing at indices: {no_enhancement[:10]}" if no_enhancement else "All present"
    )

    # Check 10: Spot check — first, middle, last entries
    spot_checks = [0, len(original)//2, len(original)-1]
    spot_ok = True
    spot_detail = []
    for idx in spot_checks:
        orig = original[idx]
        conv = converted[idx]
        if orig.get("text", "").strip() != conv.get("input", "").strip():
            spot_ok = False
            spot_detail.append(f"Index {idx}: text mismatch")
    add_check(
        "Spot check (first/middle/last entries text preserved)",
        spot_ok,
        "; ".join(spot_detail) if spot_detail else "All match"
    )

    # Statistics
    label0_count = sum(1 for e in converted if e.get("_original_label") == 0)
    label1_count = sum(1 for e in converted if e.get("_original_label") == 1)
    report["stats"] = {
        "total": len(converted),
        "label_0_compliant": label0_count,
        "label_1_violating": label1_count,
        "balance_ratio": f"{label0_count/max(label1_count,1):.2f}:1 (compliant:violating)"
    }

    return report


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Egyptian Legal LLaMA Conversion Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ── Step 1: Load all input files ──
    print("\n[1] Loading input files...")
    all_entries = load_all_files(INPUT_FILES)

    if len(all_entries) == 0:
        print("\n[ERROR] No entries loaded. Check your INPUT_FILES paths.")
        sys.exit(1)

    print(f"  Total entries loaded: {len(all_entries)}")

    # ── Step 2: Convert all entries ──
    print("\n[2] Converting to LLaMA instruction format...")
    converted = []
    errors = []

    for idx, entry in enumerate(all_entries):
        try:
            converted_entry = format_entry(entry, idx)
            converted.append(converted_entry)
        except ValueError as e:
            errors.append(str(e))

    print(f"  Successfully converted: {len(converted)}")
    if errors:
        print(f"  Conversion errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")

    # ── Step 3: Run verification ──
    print("\n[3] Running verification checks...")
    verification = run_verification(all_entries, converted)

    for check in verification["checks"]:
        print(f"  {check['status']} — {check['name']}")
        if check["detail"]:
            print(f"             {check['detail']}")

    if not verification["passed"]:
        print("\n[ERROR] Verification failed. Fix issues before proceeding.")
        sys.exit(1)

    print(f"\n  Dataset stats:")
    for k, v in verification["stats"].items():
        print(f"    {k}: {v}")

    # ── Step 4: Shuffle and split ──
    print(f"\n[4] Shuffling and splitting (train={TRAIN_RATIO*100:.0f}% / test={100-TRAIN_RATIO*100:.0f}%)...")
    random.seed(RANDOM_SEED)
    random.shuffle(converted)

    split_idx = int(len(converted) * TRAIN_RATIO)
    train_set = converted[:split_idx]
    test_set = converted[split_idx:]

    print(f"  Training set: {len(train_set)} entries")
    print(f"  Test set:     {len(test_set)} entries")

    # ── Step 5: Save outputs ──
    print(f"\n[5] Saving output files to '{OUTPUT_DIR}/'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Remove internal tracking fields before saving
    def clean_entry(e):
        return {k: v for k, v in e.items() if not k.startswith("_")}

    train_clean = [clean_entry(e) for e in train_set]
    test_clean = [clean_entry(e) for e in test_set]
    all_clean = [clean_entry(e) for e in converted]

    train_path = os.path.join(OUTPUT_DIR, "train.json")
    test_path = os.path.join(OUTPUT_DIR, "test.json")
    full_path = os.path.join(OUTPUT_DIR, "full_dataset.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_clean, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_clean, f, ensure_ascii=False, indent=2)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_clean, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {train_path}")
    print(f"  Saved: {test_path}")
    print(f"  Saved: {full_path}")

    # ── Step 6: Save a human-readable sample for manual review ──
    sample_path = os.path.join(OUTPUT_DIR, "sample_for_review.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("SAMPLE ENTRIES FOR MANUAL REVIEW\n")
        f.write("="*60 + "\n\n")
        sample_indices = [0, 1, len(converted)//4, len(converted)//2, len(converted)-2, len(converted)-1]
        for i in sample_indices:
            e = converted[i]
            f.write(f"--- Entry {i} (original label: {e['_original_label']}) ---\n")
            f.write(f"CLAUSE:\n{e['input']}\n\n")
            f.write(f"RESPONSE:\n{e['output']}\n\n")
            f.write("="*60 + "\n\n")

    print(f"  Saved: {sample_path}  ← READ THIS to manually verify output looks correct")

    # ── Final summary ──
    print("\n" + "="*60)
    print("  CONVERSION COMPLETE")
    print("="*60)
    print(f"  Total entries:    {len(converted)}")
    print(f"  Training set:     {len(train_set)}")
    print(f"  Test set:         {len(test_set)}")
    print(f"  Verification:     {'✅ ALL PASSED' if verification['passed'] else '❌ FAILED'}")
    print(f"  Output folder:    {OUTPUT_DIR}/")
    print("\n  Files to use for training:")
    print(f"    → {train_path}")
    print(f"    → {test_path}")
    print("\n  Next step: open sample_for_review.txt and verify 5-6 entries look correct.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()