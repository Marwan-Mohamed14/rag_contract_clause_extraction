"""
Egyptian Legal Contract Dataset Alignment Tool
==============================================
Version: 2.0
Purpose: Align teammates' JSON datasets with established legal rules and RAG standards.

Rules embedded:
  - All legal classification rules established during dataset generation
  - All RAG noise exclusion rules (construction law, cross-border data, etc.)
  - Semantic similarity matching (not exact string) for AI-generated clause variations

Usage:
  python dataset_alignment_tool.py --input friend_dataset.json --output aligned_dataset.json
  python dataset_alignment_tool.py --input friend_dataset.json --report-only

Requirements:
  pip install sentence-transformers scikit-learn pydantic --break-system-packages
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator, model_validator

# ─────────────────────────────────────────────────────────────
#  PYDANTIC SCHEMA — catches format errors instantly
# ─────────────────────────────────────────────────────────────

VALID_VIOLATION_TYPES = {
    "none",
    # Liability / warranty
    "liability_exemption_for_gross_negligence",
    "liability_exemption_for_unlawful_acts",
    "notice_channel_restriction",
    "notice_period_restriction",
    "contractor_consent_required_for_compensation",
    "compensation_scope_limitation",
    "moral_damage_exemption",
    "liability_cap_overriding_gross_negligence",
    "partial_liability_exemption",
    "unilateral_failure_determination",
    "removal_of_legal_remedies",
    "liability_exemption_for_employed_persons",
    # IP
    "unauthorized_use_of_ideas_without_consent_or_remuneration",
    "unauthorized_ip_ownership_transfer",
    "implied_consent_for_ip_transfer",
    "waiver_of_remuneration_right",
    "unauthorized_use_of_ideas_without_remuneration",
    "unilateral_ip_use_without_author_consent",
    # Data privacy
    "unauthorized_data_use_without_consent",
    "unauthorized_data_use_without_remuneration",
    # Representative liability
    "exemption_from_representative_liability",
    "conditional_exemption_from_representative_liability",
    "secondary_rather_than_primary_representative_liability",
    "time_limited_exemption_from_representative_liability",
    "representative_liability_cap_without_gross_negligence_override",
    "representative_liability_scope_limitation",
    # Contract formation
    "unilateral_contract_formation",
    "implied_consent_substituting_written_consent",
    "retroactive_contract_formation",
    "incomplete_preliminary_agreement",
    # Severability / interpretation
    "invalidity_of_part_voids_whole_agreement",
    "unilateral_interpretation_in_drafter_favor",
    "exclusion_of_intent_based_interpretation",
    "unilateral_determination_of_severability",
    "exclusion_of_implied_obligations_after_partial_invalidity",
    "exclusion_of_good_faith_obligation",
    "waiver_of_modification_right_after_partial_invalidity",
    # Payment / delivery
    "late_payment_penalty_missing",
    "unilateral_payment_modification",
    "delivery_timeline_missing",
    "unilateral_delivery_modification",
    # Employment
    "liability_cap",
    # Multi-type (comma-separated) handled below
}


class ClauseEntry(BaseModel):
    text: str
    label: int
    violation_type: str
    legal_basis: str

    @field_validator("label")
    @classmethod
    def label_must_be_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError(f"label must be 0 or 1, got {v}")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("text is too short or empty")
        return v

    @field_validator("legal_basis")
    @classmethod
    def legal_basis_not_empty(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("legal_basis is empty")
        return v

    @model_validator(mode="after")
    def label_violation_consistency(self):
        if self.label == 0 and self.violation_type != "none":
            raise ValueError(
                f"label=0 but violation_type='{self.violation_type}'. "
                f"Compliant entries must have violation_type='none'."
            )
        if self.label == 1 and self.violation_type == "none":
            raise ValueError(
                "label=1 but violation_type='none'. "
                "Violating entries must have a specific violation_type."
            )
        return self


# ─────────────────────────────────────────────────────────────
#  LEGAL RULES ENGINE
#  All rules established during dataset generation sessions.
#  Uses keyword signals — not exact matches.
# ─────────────────────────────────────────────────────────────

LEGAL_RULES = [
    # ── RULE 1: Liability cap alone is VALID (Article 225) ──────────────────
    {
        "id": "R01",
        "name": "Liability cap mislabeled as violation",
        "description": (
            "A pure liability cap (e.g. 'shall not exceed EGP X') is VALID under "
            "Article 225. It only becomes a violation if the cap explicitly survives "
            "gross negligence or fraud. Check: does the clause say 'even in cases of "
            "gross negligence' or similar? If not → label must be 0."
        ),
        "check": lambda entry: (
            entry["label"] == 1
            and "liability_cap" in entry["violation_type"]
            and not any(
                phrase in entry["text"].lower()
                for phrase in [
                    "gross negligence",
                    "even in cases of",
                    "including negligence",
                    "regardless of negligence",
                    "fraud",
                ]
            )
        ),
        "severity": "ERROR",
        "fix": "Change label to 0 and violation_type to 'none'. "
               "Add 'even in cases of gross negligence' to the clause text if you want it to be label 1.",
    },

    # ── RULE 2: Article 653 is construction law only ─────────────────────────
    {
        "id": "R02",
        "name": "Article 653 misused for software contract",
        "description": (
            "Article 653 applies ONLY to buildings and architects. "
            "It must never appear as the sole legal basis for a software contract violation."
        ),
        "check": lambda entry: (
            entry["label"] == 1
            and "article 653" in entry["legal_basis"].lower()
            and "gross negligence" not in entry["legal_basis"].lower()
        ),
        "severity": "ERROR",
        "fix": "Remove Article 653 from legal_basis. Replace with the correct article "
               "(e.g. Civil Law p.49 for gross negligence exemption, Article 221 for "
               "compensation scope, Article 225 for liability caps).",
    },

    # ── RULE 3: Notice channel restriction alone is VALID ───────────────────
    {
        "id": "R03",
        "name": "Notice channel restriction mislabeled",
        "description": (
            "A notice channel restriction ('notify via email only') is VALID per "
            "Article 219 — parties can agree on notice method. "
            "It only becomes a violation if claims are PERMANENTLY BARRED for "
            "non-compliance with the channel."
        ),
        "check": lambda entry: (
            entry["label"] == 1
            and entry["violation_type"] in [
                "notice_channel_restriction",
                "notice_period_restriction",
            ]
            and not any(
                phrase in entry["text"].lower()
                for phrase in [
                    "permanently barred",
                    "permanently waived",
                    "shall be forfeited",
                    "shall lapse",
                    "no claim shall",
                    "waives all rights",
                    "shall be deemed waived",
                ]
            )
        ),
        "severity": "ERROR",
        "fix": "Change label to 0 and violation_type to 'none'. "
               "The violation only exists if claims are permanently barred — "
               "not just because a channel or timing preference is specified.",
    },

    # ── RULE 4: Submission/participation ≠ IP consent ───────────────────────
    {
        "id": "R04",
        "name": "Submission treated as implied IP consent — should be label 1",
        "description": (
            "If a clause says submission of ideas or participation constitutes "
            "authorization for IP use, this is implied_consent_for_ip_transfer "
            "and must be label 1. Article 149 requires EXPLICIT written authorization."
        ),
        "check": lambda entry: (
            entry["label"] == 0
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "submission constitutes",
                    "by submitting",
                    "participation constitutes",
                    "deemed to have consented",
                    "by participating",
                    "implied consent",
                ]
            )
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "right to use",
                    "may use",
                    "authorization",
                    "freely available",
                    "without further consent",
                ]
            )
        ),
        "severity": "WARNING",
        "fix": "Review this entry. If the clause treats submission as IP authorization "
               "without explicit written consent, change label to 1 and set "
               "violation_type to 'implied_consent_for_ip_transfer'.",
    },

    # ── RULE 5: Marking burden on author is a violation ──────────────────────
    {
        "id": "R05",
        "name": "Burden on author to mark ideas as proprietary — should be label 1",
        "description": (
            "Article 149 says economic rights are retained by default. "
            "A clause that requires the author to mark ideas as 'proprietary' "
            "or 'confidential' to preserve their rights shifts the burden unlawfully."
        ),
        "check": lambda entry: (
            entry["label"] == 0
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "must be clearly marked",
                    "identified as proprietary",
                    "marked as confidential",
                    "unless marked",
                    "unless identified",
                    "not marked",
                    "fail to mark",
                ]
            )
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "right to use",
                    "freely available",
                    "deemed",
                    "considered",
                    "may use",
                ]
            )
        ),
        "severity": "WARNING",
        "fix": "Review this entry. If marking is required to preserve IP rights, "
               "this is label 1 (unauthorized_use_of_ideas_without_consent_or_remuneration). "
               "Economic rights are retained by default — no marking required.",
    },

    # ── RULE 6: Data processing without consent must be label 1 ─────────────
    {
        "id": "R06",
        "name": "Data use without explicit consent labeled as compliant",
        "description": (
            "Article 2 of Data Protection Law requires EXPLICIT consent for "
            "data processing. Any clause that uses data 'in any manner' or "
            "'freely' without consent mechanism must be label 1."
        ),
        "check": lambda entry: (
            entry["label"] == 0
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "in any manner",
                    "in any manner it may deem",
                    "without restriction",
                    "freely and without",
                    "at no cost",
                    "without seeking",
                    "without authorization",
                    "without consent",
                ]
            )
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "use information",
                    "use data",
                    "use materials",
                    "use any information",
                    "use submitted",
                    "use ideas",
                ]
            )
        ),
        "severity": "WARNING",
        "fix": "Review this entry. If data/information is used without explicit "
               "consent mechanism, this should be label 1 "
               "(unauthorized_data_use_without_consent).",
    },

    # ── RULE 7: Subcontractor exemption must be label 1 ─────────────────────
    {
        "id": "R07",
        "name": "Subcontractor liability exemption labeled as compliant",
        "description": (
            "Article 662 — contractor remains liable for subcontractor work. "
            "Any clause exempting a party from subcontractor breaches must be label 1."
        ),
        "check": lambda entry: (
            entry["label"] == 0
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "not liable for",
                    "not responsible for",
                    "no liability for",
                    "exempt from liability",
                ]
            )
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "subcontractor",
                    "sub-contractor",
                    "third party",
                    "agent",
                    "representative",
                ]
            )
        ),
        "severity": "WARNING",
        "fix": "Review this entry. If the clause exempts a party from liability "
               "for subcontractor or representative breaches, this should be label 1 "
               "(exemption_from_representative_liability).",
    },

    # ── RULE 8: Unilateral contract formation must be label 1 ───────────────
    {
        "id": "R08",
        "name": "Unilateral contract formation labeled as compliant",
        "description": (
            "Article 89 — contract requires mutual consent from both parties. "
            "Any clause allowing one party to make agreement effective without "
            "the other's signature must be label 1."
        ),
        "check": lambda entry: (
            entry["label"] == 0
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "signature is not required",
                    "signature is optional",
                    "signature is confirmatory",
                    "without being signed",
                    "bound without signing",
                    "deemed to have accepted",
                    "failure to object",
                    "silence constitutes",
                ]
            )
        ),
        "severity": "ERROR",
        "fix": "Change label to 1 and violation_type to 'unilateral_contract_formation'. "
               "Article 89 requires mutual consent — one party cannot bind the other "
               "without their signature or explicit consent.",
    },

    # ── RULE 9: Waiver of Article 149 rights is void ─────────────────────────
    {
        "id": "R09",
        "name": "Waiver of court modification rights (Art 149) — should be label 1",
        "description": (
            "Article 149 explicitly states 'any agreement to the contrary shall be void'. "
            "Any clause waiving the right to seek court modification of unfair terms "
            "is void and must be label 1."
        ),
        "check": lambda entry: (
            entry["label"] == 0
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "waives any right to seek modification",
                    "waive the right to challenge",
                    "no right to seek court",
                    "waives article 149",
                    "waive modification rights",
                ]
            )
        ),
        "severity": "ERROR",
        "fix": "Change label to 1 and violation_type to "
               "'waiver_of_modification_right_after_partial_invalidity'. "
               "This waiver is explicitly void under Article 149.",
    },

    # ── RULE 10: Good faith exclusion must be label 1 ───────────────────────
    {
        "id": "R10",
        "name": "Exclusion of good faith obligation labeled as compliant",
        "description": (
            "Article 148 — contract must be performed in good faith. "
            "This is mandatory and cannot be contracted out of. "
            "Any clause excluding good faith must be label 1."
        ),
        "check": lambda entry: (
            entry["label"] == 0
            and any(
                phrase in entry["text"].lower()
                for phrase in [
                    "no duty of good faith",
                    "no obligation of good faith",
                    "without good faith",
                    "excluding good faith",
                    "no implied obligation",
                    "no duty of honesty",
                ]
            )
        ),
        "severity": "ERROR",
        "fix": "Change label to 1 and violation_type to "
               "'exclusion_of_good_faith_obligation'. "
               "Article 148 good faith is mandatory and cannot be excluded.",
    },
]


# ─────────────────────────────────────────────────────────────
#  RAG NOISE RULES
#  Flags entries whose legal_basis cites articles that are
#  noise in the context of software contracts.
# ─────────────────────────────────────────────────────────────

RAG_NOISE_RULES = [
    {
        "id": "N01",
        "name": "Construction law cited for software/IT contract",
        "noise_phrases": [
            "article 651", "article 652", "article 653", "article 654",
            "collapse of the buildings",
            "ten years for any total",
            "fixed installations they construct",
            "defects in the buildings",
        ],
        # v2.1 fix: only flag Art 651-654 when clause text is clearly a software/IT
        # clause — not when it genuinely involves architects or fixed installations.
        # If the clause mentions architect, structural, collapse, or fixed installation
        # it is a legitimate construction clause and Art 651 is the correct law.
        "clause_must_not_contain": [
            "architect",
            "structural",
            "collapse",
            "fixed installation",
            "contractor shall be jointly",
            "jointly liable for",
            "structural defect",
            "structural collapse",
        ],
        "message": (
            "Articles 651-654 apply ONLY to physical construction (buildings/architects). "
            "They are not valid legal basis for software/IT contract violations. "
            "Remove from legal_basis and replace with the correct article."
        ),
    },
    {
        "id": "N02",
        "name": "Cross-border data transfer law cited for general data clause",
        "noise_phrases": [
            # v2.1 fix: use "article 16 " (with trailing space) to prevent
            # matching "article 164" (elected domicile) as a false positive.
            # The original "article 16" matched inside "article 164" incorrectly
            # causing 59 legitimate domicile entries to be wrongly removed.
            "article 14 ",
            "article 16 ",
            "cross border movement",
            "transferring or sharing or storing of personal data",
            "controllers outside of the arab republic",
        ],
        "message": (
            "Cross-border data transfer articles (Art 14, 16) only apply when "
            "the clause specifically involves transferring data outside Egypt. "
            "For general data use/consent clauses, use Article 2 instead."
        ),
    },
    {
        "id": "N03",
        "name": "Industrial design / WTO registration law cited",
        "noise_phrases": [
            "trade registry department",
            "industrial design",
            "world trade organization",
            "wto member",
            "register an industrial design",
        ],
        "message": (
            "Industrial design registration and WTO membership articles are not "
            "relevant to IP ownership clauses in software contracts. "
            "Use Articles 149-150 of IP Law instead."
        ),
    },
    {
        "id": "N04",
        "name": "Copyright duration articles cited",
        "noise_phrases": [
            "50 years from the death",
            "economic rights relating to works of joint authorship",
            "lives of all co-authors",
            "50 years from the date on which the work was published",
        ],
        "message": (
            "Copyright duration articles (Art 160-162) define how long protection lasts. "
            "They are not relevant to IP ownership or transfer violations. "
            "Use Articles 149-150 of IP Law instead."
        ),
    },
    {
        "id": "N05",
        "name": "Transport/carrier law cited for software contract",
        "noise_phrases": [
            "fraud in transport matters",
            "carrier is not liable",
            "carrier may not disclaim responsibility",
            "destruction, damage, or delay of the goods during transport",
            "article 213", "article 214", "article 215", "article 216",
        ],
        "message": (
            "Transport/carrier law articles are not relevant to software contracts. "
            "Remove from legal_basis."
        ),
    },
    {
        "id": "N06",
        "name": "Arbitration annulment grounds cited for interpretation clause",
        "noise_phrases": [
            "arbitral award failed to apply",
            "composition of the arbitral tribunal",
            "arbitration procedures affecting the award",
            "ipso jure annul",
            "nullity affects exclusively",
        ],
        "message": (
            "Arbitration award annulment grounds are not relevant to contract "
            "interpretation or severability clauses. Remove from legal_basis."
        ),
    },
    {
        "id": "N07",
        "name": "Insolvency / fraudulent transaction law cited",
        "noise_phrases": [
            "debtor's insolvency",
            "fraudulent intent by the debtor",
            "debtor's assets or increased his liabilities",
            "declared unenforceable against him",
        ],
        "message": (
            "Insolvency and fraudulent transaction articles are not relevant to "
            "warranty/support clauses in software contracts. Remove from legal_basis."
        ),
    },
    {
        "id": "N08",
        "name": "Compulsory licensing articles cited",
        "noise_phrases": [
            "reproduction or translation, or both",
            "without the authorization of the author and for the purposes",
            "equitable remuneration to the author or his successor",
        ],
        "message": (
            "Compulsory licensing articles are about government-forced reproduction rights. "
            "Not relevant to IP ownership clauses between private parties. "
            "Use Articles 149-150 instead."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
#  SEMANTIC CONTRADICTION DETECTOR
#  Finds similar clauses with different labels across the file.
# ─────────────────────────────────────────────────────────────

def find_semantic_contradictions(entries: list, threshold: float = 0.88) -> list:
    """
    Find pairs of entries with similar clause text but different labels.
    Uses sentence embeddings for semantic similarity — handles AI paraphrasing.
    Returns list of contradiction pairs with similarity scores.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("  ⚠️  sentence-transformers not installed. Skipping semantic check.")
        print("      Run: pip install sentence-transformers scikit-learn --break-system-packages")
        return []

    print("  🔄 Loading embedding model for semantic contradiction check...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    texts = [e["text"] for e in entries]
    print(f"  🔄 Encoding {len(texts)} clauses...")
    embeddings = model.encode(texts, show_progress_bar=False)

    contradictions = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim >= threshold and entries[i]["label"] != entries[j]["label"]:
                contradictions.append({
                    "entry_i": i,
                    "entry_j": j,
                    "similarity": float(sim),
                    "text_i": entries[i]["text"][:120],
                    "text_j": entries[j]["text"][:120],
                    "label_i": entries[i]["label"],
                    "label_j": entries[j]["label"],
                    "violation_i": entries[i]["violation_type"],
                    "violation_j": entries[j]["violation_type"],
                })

    return contradictions


# ─────────────────────────────────────────────────────────────
#  MAIN VALIDATOR
# ─────────────────────────────────────────────────────────────

def validate_dataset(filepath: str, output_path: Optional[str] = None,
                     report_only: bool = False, skip_semantic: bool = False) -> dict:

    print(f"\n{'='*70}")
    print(f"  EGYPTIAN LEGAL DATASET ALIGNMENT TOOL v2.1")
    print(f"{'='*70}")
    print(f"  Input:  {filepath}")
    if output_path:
        print(f"  Output: {output_path}")
    print(f"{'='*70}\n")

    # ── Load file ──────────────────────────────────────────────────────────
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        sys.exit(1)

    if not isinstance(raw, list):
        print("❌ File must contain a JSON array of clause entries.")
        sys.exit(1)

    print(f"📂 Loaded {len(raw)} entries from {Path(filepath).name}\n")

    # ── Track results ──────────────────────────────────────────────────────
    schema_errors    = []   # structural errors: missing fields, wrong label type, etc.
    legal_errors     = []   # confirmed legal misclassifications — REMOVE from output
    legal_warnings   = []   # possible misclassifications — flag but KEEP in output
    noise_flags      = []   # wrong articles in legal_basis — REMOVE from output
    contradictions   = []   # similar clauses with opposite labels — REMOVE from output

    # Index sets for entries to remove from output
    remove_indices   = set()

    # ── Step 1: Pydantic schema check ─────────────────────────────────────
    # Only checks: label is 0/1, text not empty, legal_basis not empty,
    # label=0 has violation_type=none, label=1 has non-none violation_type.
    # Unknown violation_type values are IGNORED — they are new categories,
    # not errors. They pass through to the output file untouched.
    print("─" * 70)
    print("STEP 1: Schema Validation (Pydantic)")
    print("─" * 70)

    for i, entry in enumerate(raw):
        try:
            ClauseEntry(**entry)
        except Exception as e:
            schema_errors.append({
                "entry_index": i,
                "text_preview": str(entry.get("text", ""))[:80],
                "error": str(e),
            })
            remove_indices.add(i)

    if schema_errors:
        print(f"  ❌ {len(schema_errors)} structural schema error(s) — removed from output:")
        for err in schema_errors:
            print(f"\n  Entry #{err['entry_index']}: {err['text_preview']}...")
            print(f"  Error: {err['error']}")
    else:
        print(f"  ✅ All {len(raw)} entries pass schema validation.")

    # ── Step 2: Legal rules check ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("STEP 2: Legal Rules Check")
    print("─" * 70)

    for i, entry in enumerate(raw):
        for rule in LEGAL_RULES:
            try:
                if rule["check"](entry):
                    result = {
                        "entry_index": i,
                        "rule_id": rule["id"],
                        "rule_name": rule["name"],
                        "text_preview": entry.get("text", "")[:100],
                        "current_label": entry.get("label"),
                        "current_violation_type": entry.get("violation_type"),
                        "fix": rule["fix"],
                        "severity": rule["severity"],
                    }
                    if rule["severity"] == "ERROR":
                        legal_errors.append(result)
                        remove_indices.add(i)  # remove confirmed errors
                    else:
                        legal_warnings.append(result)
                        # warnings stay in output — flagged for manual review only
            except Exception:
                pass

    if legal_errors:
        print(f"  ❌ {len(legal_errors)} legal ERROR(s) — removed from output:\n")
        for err in legal_errors:
            print(f"  [{err['rule_id']}] Entry #{err['entry_index']}: {err['rule_name']}")
            print(f"  Text: {err['text_preview']}...")
            print(f"  Current: label={err['current_label']}, "
                  f"violation_type={err['current_violation_type']}")
            print(f"  Fix: {err['fix']}\n")
    else:
        print(f"  ✅ No legal errors found.")

    if legal_warnings:
        print(f"\n  ⚠️  {len(legal_warnings)} legal WARNING(s) — kept in output, review recommended:\n")
        for warn in legal_warnings:
            print(f"  [{warn['rule_id']}] Entry #{warn['entry_index']}: {warn['rule_name']}")
            print(f"  Text: {warn['text_preview']}...")
            print(f"  Current: label={warn['current_label']}, "
                  f"violation_type={warn['current_violation_type']}")
            print(f"  Fix: {warn['fix']}\n")
    else:
        print(f"  ✅ No legal warnings found.")

    # ── Step 3: RAG noise check ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("STEP 3: RAG Noise Check (legal_basis field)")
    print("─" * 70)

    for i, entry in enumerate(raw):
        legal_basis = entry.get("legal_basis", "").lower()
        clause_text  = entry.get("text", "").lower()
        for noise_rule in RAG_NOISE_RULES:
            # v2.1: if the rule has a clause_must_not_contain list, skip this
            # entry if the clause text contains any of those phrases — meaning
            # the article IS the correct law for this clause type.
            exempt_phrases = noise_rule.get("clause_must_not_contain", [])
            if any(p.lower() in clause_text for p in exempt_phrases):
                continue
            for phrase in noise_rule["noise_phrases"]:
                if phrase.lower() in legal_basis:
                    noise_flags.append({
                        "entry_index": i,
                        "noise_rule_id": noise_rule["id"],
                        "noise_rule_name": noise_rule["name"],
                        "matched_phrase": phrase,
                        "text_preview": entry.get("text", "")[:100],
                        "legal_basis_preview": entry.get("legal_basis", "")[:150],
                        "message": noise_rule["message"],
                    })
                    remove_indices.add(i)  # remove noise-contaminated entries
                    break

    if noise_flags:
        print(f"  ❌ {len(noise_flags)} RAG noise flag(s) — removed from output:\n")
        for flag in noise_flags:
            print(f"  [{flag['noise_rule_id']}] Entry #{flag['entry_index']}: "
                  f"{flag['noise_rule_name']}")
            print(f"  Text: {flag['text_preview']}...")
            print(f"  Legal basis: {flag['legal_basis_preview']}...")
            print(f"  Problem: {flag['message']}\n")
    else:
        print(f"  ✅ No RAG noise detected in legal_basis fields.")

    # ── Step 4: Semantic contradiction check ──────────────────────────────
    print(f"\n{'─'*70}")
    print("STEP 4: Semantic Contradiction Check (similar clauses, different labels)")
    print("─" * 70)

    if skip_semantic:
        print("  ⏭️  Skipped (--skip-semantic flag set).")
    else:
        contradictions = find_semantic_contradictions(raw)
        if contradictions:
            print(f"  ❌ {len(contradictions)} semantic contradiction(s) — both entries removed:\n")
            for c in contradictions:
                print(f"  Entries #{c['entry_i']} (label={c['label_i']}) "
                      f"and #{c['entry_j']} (label={c['label_j']}) "
                      f"— similarity: {c['similarity']:.3f}")
                print(f"  Text A: {c['text_i']}...")
                print(f"  Text B: {c['text_j']}...")
                print(f"  Violation A: {c['violation_i']}")
                print(f"  Violation B: {c['violation_j']}\n")
                remove_indices.add(c["entry_i"])
                remove_indices.add(c["entry_j"])
        else:
            print(f"  ✅ No semantic contradictions found.")

    # ── Build clean output ─────────────────────────────────────────────────
    clean_entries = [
        entry for i, entry in enumerate(raw)
        if i not in remove_indices
    ]
    removed_entries = [
        {"original_index": i, "entry": raw[i]}
        for i in sorted(remove_indices)
    ]

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ALIGNMENT REPORT SUMMARY")
    print(f"{'='*70}")
    print(f"  Total entries input    : {len(raw)}")
    print(f"  Schema errors removed  : {len(schema_errors)}  "
          f"{'❌' if schema_errors else '✅'}")
    print(f"  Legal errors removed   : {len(legal_errors)}  "
          f"{'❌' if legal_errors else '✅'}")
    print(f"  Legal warnings (kept)  : {len(legal_warnings)}  "
          f"{'⚠️ ' if legal_warnings else '✅'}")
    print(f"  RAG noise removed      : {len(noise_flags)}  "
          f"{'❌' if noise_flags else '✅'}")
    print(f"  Contradictions removed : {len(contradictions)}  "
          f"{'❌' if contradictions else '✅'}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total removed          : {len(remove_indices)}")
    print(f"  Clean entries output   : {len(clean_entries)}  ✅")

    total_errors = len(schema_errors) + len(legal_errors) + len(noise_flags) + len(contradictions)
    if total_errors == 0 and len(legal_warnings) == 0:
        print(f"\n  🟢 DATASET IS CLEAN — Ready for merging and training.")
    elif total_errors == 0:
        print(f"\n  🟡 DATASET HAS WARNINGS — Review flagged entries before merging.")
    else:
        print(f"\n  🔴 {len(remove_indices)} entries removed — "
              f"see report for details.")

    # ── Save clean output file ─────────────────────────────────────────────
    if output_path and not report_only:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean_entries, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 Clean dataset saved to: {output_path} "
              f"({len(clean_entries)} entries)")

    # ── Save JSON report ───────────────────────────────────────────────────
    report = {
        "file": filepath,
        "total_entries_input": len(raw),
        "total_entries_output": len(clean_entries),
        "total_removed": len(remove_indices),
        "summary": {
            "schema_errors_removed": len(schema_errors),
            "legal_errors_removed": len(legal_errors),
            "legal_warnings_kept": len(legal_warnings),
            "rag_noise_removed": len(noise_flags),
            "contradictions_removed": len(contradictions),
        },
        "removed_entries": removed_entries,
        "schema_errors": schema_errors,
        "legal_errors": legal_errors,
        "legal_warnings": legal_warnings,
        "noise_flags": noise_flags,
        "semantic_contradictions": contradictions,
    }

    report_path = Path(filepath).stem + "_alignment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  📄 Full report saved to:  {report_path}")
    print(f"{'='*70}\n")

    return report


# ─────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Egyptian Legal Contract Dataset Alignment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_alignment_tool.py --input friend1.json
  python dataset_alignment_tool.py --input friend2.json --skip-semantic
  python dataset_alignment_tool.py --input friend3.json --output aligned_friend3.json
        """,
    )
    parser.add_argument("--input", required=True, help="Path to the JSON dataset file to validate")
    parser.add_argument("--output", default=None, help="Optional: path to save aligned output file")
    parser.add_argument("--report-only", action="store_true", help="Only generate report, no output file")
    parser.add_argument("--skip-semantic", action="store_true",
                        help="Skip semantic contradiction check (faster, no GPU needed)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    validate_dataset(
        filepath=args.input,
        output_path=args.output,
        report_only=args.report_only,
        skip_semantic=args.skip_semantic,
    )


if __name__ == "__main__":
    main()