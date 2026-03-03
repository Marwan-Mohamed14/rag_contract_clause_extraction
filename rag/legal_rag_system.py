"""
Egyptian Legal Contract RAG System v4.5
========================================
FIXES over v4.1:

1. CONTENT FILTER (new):
   Added CONTENT_FILTERS dict — blocks specific chunk content patterns per category,
   regardless of score. This removes lease law (Art 578-581) from liability_warranty
   results. Lease articles kept appearing because they mention "liability" and "defects"
   which matched warranty queries semantically. Now any chunk containing "lessor",
   "lessee", "leased premises" etc. is blocked from liability_warranty results.
   Same filter blocks employment contract articles from delivery_timeline results.

2. LIABILITY_WARRANTY QUERIES REWRITTEN:
   Old queries anchored on Art 651 (10-year building guarantee) — this is construction
   law and kept pulling architecture/building chunks. New queries anchor on:
   - Art 653 (exemption null and void) — applies to ALL contracts
   - Art 652 (design vs execution liability)
   - Art 654 (3-year claim window) — note: construction context, software parties can agree shorter
   - Civil Law p.49 (unlawful acts / gross negligence exemption void) — applies to ALL contracts
   - Art 218-219 (compensation requires formal notice) — applies to ALL contracts

3. PAYMENT_TERMS THRESHOLD: 0.72 → 0.69
   Technology transfer articles (Art 71-72, 87-88) were scoring 0.698-0.700.
   Tightening to 0.69 cuts them out cleanly.

4. VERSION HEADER UPDATED throughout.

KEY LEGAL NOTE (documented here for dataset generation):
   Articles 651-654 are Egyptian Civil Law for CONSTRUCTION (buildings collapsing).
   For SOFTWARE contracts:
   - Art 653 principle (no liability exemption) → applies by analogy to all contracts
   - Art 654 (3-year claims window) → construction only; software parties CAN agree shorter periods
   - Art 651 (10-year guarantee) → construction only, NOT binding on software
   The universally applicable rules across all contract types are:
   - Exemption for unlawful acts is void (Civil Law p.49)
   - Exemption for gross negligence of employed persons is void (Civil Law p.49)
   - Compensation requires formal notice unless otherwise agreed (Art 218-219)
"""

import warnings
import hashlib
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


# ─────────────────────────────────────────────────────────────
#  TRIGGER KEYWORDS
# ─────────────────────────────────────────────────────────────
TRIGGER_KEYWORDS = {
    "arbitration_jurisdiction": [
        "arbitration", "jurisdiction", "governing law",
        "english law", "foreign law", "london court",
        "dispute resolution", "tribunal", "icc rules",
        "lcia", "uncitral", "settle disputes", "applicable law"
    ],
    "data_privacy": [
        "personal data", "user data", "customer data",
        "data protection", "data processing", "data transfer",
        "sell data", "monetize data", "share data", "collect data",
        "data consent", "data subject", "gdpr", "data controller",
        "data processor", "sensitive data", "cross-border data",
        "use information", "information submitted", "submitted information",
        "any manner it may deem", "deem appropriate", "confidential information",
        "disclose information", "right to use information"
    ],
    "electronic_signature": [
        "typed name", "e-sign", "electronic signature",
        "digital signature", "docusign", "email confirmation",
        "click to accept", "wet signature", "scanned signature",
        "whatsapp confirmation", "sms confirmation", "faxed signature",
        "sign electronically", "binding signature"
    ],
    "intellectual_property": [
        "intellectual property", "copyright", "source code ownership",
        "work product", "derivative works", "moral rights",
        "proprietary rights", "ip ownership", "license grant",
        "all rights reserved", "waive rights", "assign ownership",
        "vest in", "becomes property of",
        "proprietary", "ideas presented", "reserves the right to use",
        "right to use any", "use any and all ideas",
        "submitted ideas", "ownership of ideas", "unless identified as proprietary"
    ],
    "payment_terms": [
        "non-refundable", "late payment", "payment penalty",
        "invoice due", "payment schedule", "payment terms",
        "fee structure", "refund policy", "price adjustment",
        "payment obligation", "total value of the contract",
        "installment", "50%", "percent", "upon delivery",
        "upon going live", "sign off", "milestone payment",
        "taxes and duties", "total contract value"
    ],
    "delivery_timeline": [
        "implementation period", "delivery deadline", "days from",
        "completion date", "final delivery", "project timeline",
        "handover date", "milestone", "delivery schedule",
        "days from signing", "working days", "calendar days",
        "period of implementation", "project duration"
    ],
    "liability_warranty": [
        "warranty", "as-is", "as is", "limitation of liability",
        "cap on damages", "indemnify", "indemnification",
        "consequential damages", "liability waiver", "defect liability",
        "ten-year guarantee", "structural defect", "exempt from liability",
        "full liability", "assumes liability", "liable for damages",
        "responsible for damages", "free of charge", "no extra cost",
        "support included", "maintenance included",
        "liability for", "damages that may occur"
    ],
    "termination": [
        "termination", "terminate", "rescission", "cancellation",
        "notice of termination", "right to cancel", "end of contract",
        "early termination", "termination for cause", "termination fee",
        "notice period", "termination clause"
    ]
}

# ─────────────────────────────────────────────────────────────
#  BLOCKED SOURCES
# ─────────────────────────────────────────────────────────────
BLOCKED_SOURCES = [
    "nda template",
    "nda_template",
]

CONDITIONAL_SOURCES = {
    "investment": ["investment", "invest", "foreign direct"],
    "competition": [
        "monopoly", "dominant", "price fixing", "market share",
        "anti-competitive", "cartel", "economic concentration",
        "merger", "acquisition", "abuse of dominance"
    ],
}

# ─────────────────────────────────────────────────────────────
#  CONTENT FILTERS (NEW in v4.2)
#  Blocks chunks whose text matches known noise patterns per category.
#  This operates AFTER score filtering as a hard semantic block.
#
#  liability_warranty: blocks lease law (Art 578-581)
#    — "lessor"/"lessee" articles score well on warranty queries because
#      they mention "liability" and "defects" but are about physical leases.
#
#  delivery_timeline: blocks employment contract articles
#    — employment termination/remuneration articles mention "completion"
#      and "work" which matches delivery queries semantically.
# ─────────────────────────────────────────────────────────────
CONTENT_FILTERS = {
    # liability_warranty: block all non-contractor legal domains that appear
    # because they mention "liability" or "exemption" generically.
    # Each group below was identified from actual RAG output analysis.
    "liability_warranty": [
        # Lease law (Art 578-581) — blocked in v4.2
        "lessor", "lessee", "leased premises",
        "leased property", "lessee must use",
        "lessor's liability", "eviction",
        # Partnership law (Art 515-516) — blocked in v4.3
        "partner shall not share", "company's profits or losses",
        "partner whose contribution", "partner appointed as manager",
        "company contract shall be void",
        # Transport/carrier law (Commercial Law Art 213-215) — blocked in v4.3
        "carrier", "consignee", "sender", "transport risks",
        "derailment", "collision", "transport contract",
        "carrier's liability", "carrier from liability",
        # Interest rate / delay penalty articles (Art 225-227) — blocked in v4.3
        # These appear because they mention "compensation" and "gross negligence"
        # but are about monetary delay interest, not warranty/defect liability
        "rate of 4%", "rate of interest", "judicial demand",
        "delay in commercial matters", "interest accrues",
        "sum of money known in amount",
        # Construction law (Art 651-654) — blocked in v4.4
        # These apply ONLY to buildings/architects — NOT to software contracts.
        # Parties can legitimately agree on any warranty scope in software.
        # Art 653 ("no exemption for architect/contractor") is construction-specific.
        # The only universally applicable rules are Civil Law p.49 (gross negligence
        # and unlawful acts exemptions void) which are retrieved separately.
        "architect and the contractor shall be jointly liable",
        "partial collapse of the buildings",
        "fixed installations they construct",
        "ten years for any total",
        "claims arising from the guarantee mentioned above shall lapse",
        "defects in the buildings or installations",
        "architect's role was limited to preparing the design",
        "clause seeking to exempt or limit the liability of the architect",
        "collapse of the buildings",
        "ten-year period shall commence",
        # Insolvency/fraudulent transaction law (Art 237-238) — blocked in v4.5
        # These appear because they mention "debtor" and "fraud" but are about
        # creditor protection against insolvent debtors, not warranty/support
        "debtor's assets or increased his liabilities",
        "debtor's insolvency",
        "fraudulent intent by the debtor",
        "transaction conducted by his debtor",
        "declared unenforceable against him",
        # Creditor-default articles (Art 334-336) — blocked in v4.5
        # About what happens when creditor refuses performance — not warranty law
        "creditor unjustifiably refuses to accept",
        "creditor is in default",
        "interest shall cease to accrue",
        "deposit the thing at the creditor",
        # Data Protection Law penalty articles — blocked in v4.5
        # Penalty fines for legal representatives appear because they mention
        # "liability" and "obligations" — irrelevant to warranty/support clauses
        "data protection officer",
        "two hundred thousand egyptian pounds",
        "not exceeding two million",
        "legal representative of a juristic person",
        "penalized by a fine",
    ],
    "delivery_timeline": [
        "employment contract shall terminate",
        "worker's remuneration",
        "termination of the employment",
        "employer shall pay the worker",
        "employment contract",
    ],
}


class EgyptianLegalRAG:
    """
    Production-grade RAG for Egyptian legal contract analysis.
    v4.4: Construction law (Art 651-654) fully blocked from warranty results.
    """

    RELEVANCE_THRESHOLD = 0.68

    def __init__(self, data_folder: str, db_path: str = "./legal_rag_db"):
        self.data_folder = Path(data_folder)
        self.db_path = db_path
        self.vectorstore = None
        self._hash_file = Path(db_path) / "_pdf_hash.json"

        print("⚙️  Loading embedding model (first run may take ~30s)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("✅ Embedding model ready")

    # ──────────────────────────────────────────────
    #  DATABASE MANAGEMENT
    # ──────────────────────────────────────────────

    def _compute_pdf_hash(self) -> str:
        hasher = hashlib.md5()
        pdf_files = sorted(self.data_folder.glob("**/*.pdf"))
        for pdf in pdf_files:
            hasher.update(pdf.name.encode())
            hasher.update(str(pdf.stat().st_size).encode())
        return hasher.hexdigest()

    def _is_db_current(self) -> bool:
        if not Path(self.db_path).exists():
            return False
        if not self._hash_file.exists():
            return False
        try:
            saved = json.loads(self._hash_file.read_text())
            return saved.get("hash") == self._compute_pdf_hash()
        except Exception:
            return False

    def _save_pdf_hash(self):
        self._hash_file.write_text(
            json.dumps({"hash": self._compute_pdf_hash()})
        )

    def rebuild_database(self):
        if Path(self.db_path).exists():
            print("🗑️  Deleting old database...")
            shutil.rmtree(self.db_path)
        self._build_database()

    def _build_database(self):
        print("\n🔧 Building database from PDFs...")

        pdf_files = list(self.data_folder.glob("**/*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_folder}")

        print(f"📁 Found {len(pdf_files)} PDF files:")
        for f in pdf_files:
            print(f"   • {f.name}")

        loader = DirectoryLoader(
            str(self.data_folder),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"\n✅ Loaded {len(documents)} pages total")

        documents = [d for d in documents if len(d.page_content.strip()) > 150]
        print(f"✅ After filtering blank pages: {len(documents)} pages")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\nArticle", "\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        print("🗄️  Building vector database...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )

        self._save_pdf_hash()
        print(f"✅ Database ready — saved to {self.db_path}\n")

    def initialize(self):
        if self._is_db_current():
            print("📂 Loading existing database...")
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            print("✅ Ready!\n")
        else:
            if Path(self.db_path).exists():
                print("⚠️  PDF files changed — rebuilding database...")
            self._build_database()

    # ──────────────────────────────────────────────
    #  TRIGGER DETECTION
    # ──────────────────────────────────────────────

    def _detect_triggers(self, text: str) -> List[str]:
        text_lower = text.lower()
        triggered = []
        for category, keywords in TRIGGER_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                triggered.append(category)
        return triggered

    # ──────────────────────────────────────────────
    #  FILTERING
    # ──────────────────────────────────────────────

    def _is_source_allowed(self, source: str, clause_text: str) -> bool:
        source_lower = source.lower()

        for blocked in BLOCKED_SOURCES:
            if blocked in source_lower:
                return False

        for source_keyword, topic_triggers in CONDITIONAL_SOURCES.items():
            if source_keyword in source_lower:
                if not any(t in clause_text.lower() for t in topic_triggers):
                    return False

        return True

    def _is_content_allowed(self, content: str, category: str) -> bool:
        """
        Block chunks whose content matches noise patterns for a specific category.
        Introduced in v4.2 to block lease law from warranty results and
        employment law from delivery results.
        """
        blocked_phrases = CONTENT_FILTERS.get(category, [])
        content_lower = content.lower()
        for phrase in blocked_phrases:
            if phrase in content_lower:
                return False
        return True

    # ──────────────────────────────────────────────
    #  LAW RETRIEVAL
    # ──────────────────────────────────────────────

    def _search_with_score(self, query: str, k: int = 4) -> List[Tuple]:
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def _get_laws_for_category(self, category: str, clause_text: str = "") -> List[Tuple]:
        """
        Retrieve relevant laws for a category using targeted queries.

        Per-category thresholds (tuned from output analysis):
          liability_warranty : 0.65  — Art 651-654 score ~0.55-0.64
          payment_terms      : 0.69  — technology transfer noise scored 0.698-0.700, now cut
          delivery_timeline  : 0.69  — employment law noise scored 0.71, now cut
          all others         : 0.68  — default
        """
        category_thresholds = {
            "arbitration_jurisdiction": 0.68,
            "data_privacy":             0.68,
            "electronic_signature":     0.68,
            "intellectual_property":    0.68,
            "payment_terms":            0.69,
            "delivery_timeline":        0.69,
            "liability_warranty":       0.65,
            "termination":              0.68,
        }

        category_queries = {
            "arbitration_jurisdiction": [
                "Article 87 Egyptian courts jurisdiction technology transfer disputes null void",
                "Article 86 technology transfer termination reconsideration five years",
                "arbitration egypt commercial law disputes merits governed egyptian law"
            ],
            "data_privacy": [
                "personal data consent processor controller data subject egypt",
                "sensitive data collection disclosure consent prohibition egypt",
                "data protection personal information transfer cross border egypt"
            ],
            "electronic_signature": [
                "electronic signature certification authority egypt valid legally binding",
                "digital signature law egypt conditions requirements valid"
            ],
            "intellectual_property": [
                "copyright author moral rights ownership work egypt law 82",
                "intellectual property ownership derivative works modification egypt",
                "author rights prevent modification distortion work egypt"
            ],
            "payment_terms": [
                "civil law price agreed parties service contract remuneration egypt",
                "remuneration payable upon delivery work contractor civil law article 656",
                "civil law obligations payment installment parties agreed contract egypt",
                "commercial law payment obligation contractor employer agreed price"
            ],
            "delivery_timeline": [
                "contractor completed work delivery employer civil law egypt article 655",
                "civil law contractor fails complete work period rescission employer egypt",
                "contractor obligations deadline period implementation completion egypt",
                "civil law employer rescission contractor non-compliance work period"
            ],
            # v4.4: Removed ALL construction law queries (Art 651-654, Art 652).
            # These articles apply to buildings/architects ONLY and should not
            # appear in software contract analysis at all.
            # Only universally applicable civil law rules are now retrieved:
            # - Civil Law p.49: gross negligence exemption void, unlawful acts exemption void
            # - Art 218-220: compensation requires formal notice, notice channels
            # These apply to ALL contracts regardless of type.
            "liability_warranty": [
                "civil law exemption liability unlawful acts void gross negligence employed",
                "article 218 compensation due notice debtor served civil law egypt",
                "article 219 notice debtor formal warning any act serving warning post",
                "article 220 notice debtor not required unlawful act impossible performance",
            ],
            "termination": [
                "contract termination rescission notice period breach civil law egypt",
                "rescission contract contractor failure compliance period egypt"
            ]
        }

        threshold = category_thresholds.get(category, self.RELEVANCE_THRESHOLD)
        queries = category_queries.get(category, [])
        results = []
        seen_content = set()

        for query in queries:
            docs_scores = self._search_with_score(query, k=3)
            for doc, score in docs_scores:
                # 1. Score filter
                if score > threshold:
                    continue
                # 2. Source filter
                source = doc.metadata.get("source", "")
                if not self._is_source_allowed(source, clause_text):
                    continue
                # 3. Content filter — blocks lease/employment noise (v4.2)
                if not self._is_content_allowed(doc.page_content, category):
                    continue
                # 4. Deduplication
                content_key = doc.page_content[:80]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    results.append((doc, score))

        return results

    # ──────────────────────────────────────────────
    #  MAIN ANALYSIS
    # ──────────────────────────────────────────────

    def analyze_clause(self, clause_text: str, verbose: bool = True) -> Dict:
        if verbose:
            print("\n" + "="*80)
            print("CLAUSE ANALYSIS")
            print("="*80)
            print(f"\n📝 Clause: {clause_text[:200]}{'...' if len(clause_text)>200 else ''}")

        triggered = self._detect_triggers(clause_text)

        if verbose:
            if triggered:
                print(f"\n🎯 Triggered categories: {', '.join(triggered)}")
            else:
                print("\n🎯 No specific triggers — using general similarity search")

        all_results = []
        seen_keys = set()

        if triggered:
            for category in triggered:
                cat_results = self._get_laws_for_category(category, clause_text)
                if verbose:
                    print(f"  📋 {category}: found {len(cat_results)} relevant law(s)")
                for doc, score in cat_results:
                    key = (doc.metadata.get("source", ""), doc.page_content[:80])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_results.append((doc, score, category))
        else:
            docs_scores = self._search_with_score(clause_text, k=6)
            for doc, score in docs_scores:
                if score <= self.RELEVANCE_THRESHOLD:
                    source = doc.metadata.get("source", "")
                    if not self._is_source_allowed(source, clause_text):
                        continue
                    key = (doc.metadata.get("source", ""), doc.page_content[:80])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_results.append((doc, score, "general"))

        all_results.sort(key=lambda x: x[1])
        all_results = all_results[:8]

        if verbose:
            print(f"\n✅ Retrieved {len(all_results)} relevant law(s) "
                  f"(threshold={self.RELEVANCE_THRESHOLD})\n")

        formatted = self._format_output(all_results)

        if verbose:
            print("="*80)
            print("RELEVANT EGYPTIAN LAWS")
            print("="*80)
            print(formatted)
            print("="*80)
            print("✅ ANALYSIS COMPLETE")
            print("="*80 + "\n")

        return {
            "clause": clause_text,
            "triggered_categories": triggered,
            "laws": [
                {
                    "rank": i + 1,
                    "source": Path(doc.metadata.get("source", "Unknown")).name,
                    "page": doc.metadata.get("page", "Unknown"),
                    "score": round(score, 4),
                    "category": category,
                    "content": doc.page_content
                }
                for i, (doc, score, category) in enumerate(all_results)
            ],
            "formatted_output": formatted
        }

    def analyze_contract(self, contract_text: str, max_laws: int = 12) -> Dict:
        print("\n" + "="*80)
        print("FULL CONTRACT ANALYSIS")
        print("="*80)
        print(f"\n📄 Contract size: {len(contract_text):,} characters")

        clauses = self._extract_clauses(contract_text)
        print(f"✂️  Extracted {len(clauses)} clause(s)")

        all_triggers = self._detect_triggers(contract_text)
        print(f"🎯 Contract-level triggers: {', '.join(all_triggers) if all_triggers else 'none'}\n")

        all_laws = []
        seen_keys = set()

        for clause in clauses:
            result = self.analyze_clause(clause["text"], verbose=False)
            for law in result["laws"]:
                key = (law["source"], law["content"][:80])
                if key not in seen_keys:
                    seen_keys.add(key)
                    law["from_clause"] = clause["id"]
                    all_laws.append(law)

        all_laws.sort(key=lambda x: x["score"])
        final_laws = all_laws[:max_laws]

        print(f"✅ Total unique relevant laws found: {len(all_laws)}")
        print(f"✅ Showing top {len(final_laws)}\n")

        formatted = self._format_output_from_dicts(final_laws)

        print("="*80)
        print("RELEVANT EGYPTIAN LAWS")
        print("="*80)
        print(formatted)
        print("="*80)
        print("✅ CONTRACT ANALYSIS COMPLETE")
        print("="*80 + "\n")

        return {
            "triggered_categories": all_triggers,
            "laws": final_laws,
            "formatted_output": formatted
        }

    # ──────────────────────────────────────────────
    #  CLAUSE EXTRACTION
    # ──────────────────────────────────────────────

    def _extract_clauses(self, contract_text: str) -> List[Dict]:
        clauses = []
        pattern = r'(\d+\.\d+)\s+(.+?)(?=\d+\.\d+|\Z)'
        matches = re.findall(pattern, contract_text, re.DOTALL)

        if matches:
            for clause_id, content in matches:
                content = content.strip()
                if len(content) > 30:
                    clauses.append({"id": clause_id, "text": content})
        else:
            paragraphs = [p.strip() for p in contract_text.split('\n\n')
                         if len(p.strip()) > 30]
            for i, para in enumerate(paragraphs):
                clauses.append({"id": str(i + 1), "text": para})

        return clauses if clauses else [{"id": "1", "text": contract_text}]

    # ──────────────────────────────────────────────
    #  FORMATTING
    # ──────────────────────────────────────────────

    def _format_output(self, results: List[Tuple]) -> str:
        lines = []
        for i, (doc, score, category) in enumerate(results, 1):
            source = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page", "Unknown")
            lines.append("─" * 80)
            lines.append(f"📋 LAW #{i}  |  Category: {category}  |  Relevance Score: {score:.4f}")
            lines.append(f"📁 Source: {source}  |  Page: {page}")
            lines.append("─" * 80)
            lines.append(doc.page_content.strip())
            lines.append("")
        return "\n".join(lines)

    def _format_output_from_dicts(self, laws: List[Dict]) -> str:
        lines = []
        for law in laws:
            lines.append("─" * 80)
            lines.append(f"📋 LAW #{law['rank']}  |  Category: {law['category']}  |  Score: {law['score']}")
            lines.append(f"📁 Source: {law['source']}  |  Page: {law['page']}")
            lines.append("─" * 80)
            lines.append(law["content"].strip())
            lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  CLI INTERFACE
# ─────────────────────────────────────────────────────────────

def get_multiline_input(prompt: str = "") -> str:
    print("="*80)
    print(prompt or "PASTE YOUR TEXT BELOW")
    print("="*80)
    print("📝 Paste content, then type 'DONE' on a new line and press Enter")
    print("-"*80)
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "DONE":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def main():
    print("="*80)
    print("EGYPTIAN LEGAL CONTRACT ANALYZER v4.5")
    print("Content filter v4.4 | Insolvency+DataProtection+CreditorDefault blocked | Software warranty")
    print("="*80 + "\n")

    DATA_FOLDER = r"C:\Users\Mohamed\Desktop\data"

    rag = EgyptianLegalRAG(data_folder=DATA_FOLDER)
    rag.initialize()

    print("\nMode selection:")
    print("  1 — Analyze a single clause")
    print("  2 — Analyze a full contract")
    print("  3 — Force rebuild database (use after replacing PDFs)")

    mode = input("\nEnter mode (1/2/3): ").strip()

    if mode == "3":
        rag.rebuild_database()
        print("✅ Database rebuilt successfully.")
        return

    text = get_multiline_input("PASTE YOUR CLAUSE OR CONTRACT BELOW")

    if not text.strip():
        print("\n❌ No text provided. Exiting.")
        return

    if mode == "1" or len(text) <= 1000:
        rag.analyze_clause(text)
    else:
        rag.analyze_contract(text)


if __name__ == "__main__":
    main()