"""
Egyptian Legal Contract RAG System v4.1
FIXES over v4.0:
- Tighter trigger detection: requires intent-confirming phrases, not bare single words
  e.g. "sign" alone no longer fires electronic_signature — needs "typed name", "e-sign", etc.
- Threshold tightened: 1.2 → 0.68 to cut noisy low-relevance results
- NDA template filter restored (was accidentally removed in v4.0)
- Blocked sources list: NDA templates, Investment Law for non-investment clauses
- delivery_timeline queries now target contractor/software delivery specifically
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

# Use updated packages to avoid deprecation warnings
# Run once if needed: pip install -U langchain-huggingface langchain-chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


# ─────────────────────────────────────────────────────────────
#  TRIGGER KEYWORDS — controls conditional force-retrieval
#
#  DESIGN RULES (learned from output analysis):
#  1. No bare common words ("sign", "data", "days") — they appear
#     in every contract and cause false-positive category triggers.
#  2. Use INTENT-CONFIRMING phrases: 2+ words, or domain-specific
#     terms that only appear when the category truly applies.
#  3. Single-word triggers allowed ONLY if they are highly specific
#     (e.g. "arbitration", "copyright") and rarely appear by accident.
# ─────────────────────────────────────────────────────────────
TRIGGER_KEYWORDS = {
    # Fires only when the clause is actually about WHERE disputes go
    "arbitration_jurisdiction": [
        "arbitration", "jurisdiction", "governing law",
        "english law", "foreign law", "london court",
        "dispute resolution", "tribunal", "icc rules",
        "lcia", "uncitral", "settle disputes", "applicable law"
    ],
    # Fires when clause handles personal/user data OR claims broad rights over submitted info
    # Added softer patterns to catch "use information in any manner" type clauses
    # which are data usage grabs without consent — regulated by Egyptian Data Protection Law
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
    # Fires only when clause is specifically about HOW to sign
    "electronic_signature": [
        "typed name", "e-sign", "electronic signature",
        "digital signature", "docusign", "email confirmation",
        "click to accept", "wet signature", "scanned signature",
        "whatsapp confirmation", "sms confirmation", "faxed signature",
        "sign electronically", "binding signature"
    ],
    # Fires when clause deals with ownership of created/submitted work or ideas
    # Added: "proprietary", "ideas presented", "reserves the right to use"
    # to catch IP grabs phrased as "we own your submitted ideas unless marked otherwise"
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
    # Fires only for clauses specifically about money obligations
    # NOTE: "down payment" removed — it appears as a TIME REFERENCE in delivery clauses
    # ("17 days from receipt of down payment") and causes false triggers.
    # Only phrases that make the clause STRUCTURALLY about payment are kept.
    "payment_terms": [
        "non-refundable", "late payment", "payment penalty",
        "invoice due", "payment schedule", "payment terms",
        "fee structure", "refund policy", "price adjustment",
        "payment obligation", "total value of the contract",
        "installment", "50%", "percent", "upon delivery",
        "upon going live", "sign off", "milestone payment",
        "taxes and duties", "total contract value"
    ],
    # Fires for clauses about WHEN work is due / delivered
    "delivery_timeline": [
        "implementation period", "delivery deadline", "days from",
        "completion date", "final delivery", "project timeline",
        "handover date", "milestone", "delivery schedule",
        "days from signing", "working days", "calendar days",
        "period of implementation", "project duration"
    ],
    # Fires for clauses that limit, assign, or discuss fault/responsibility/damages
    # Added: "liability", "damages", "full liability" — specific enough to legal context
    # Added: "maintenance", "support" — to catch maintenance clauses that exclude liability
    # Added: "free of charge", "bugs", "defects" — software warranty language
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
    # Fires only for clauses that explicitly end the contract
    "termination": [
        "termination", "terminate", "rescission", "cancellation",
        "notice of termination", "right to cancel", "end of contract",
        "early termination", "termination for cause", "termination fee",
        "notice period", "termination clause"
    ]
}

# ─────────────────────────────────────────────────────────────
#  BLOCKED SOURCES — never return results from these files
# ─────────────────────────────────────────────────────────────
BLOCKED_SOURCES = [
    "nda template",       # Contract templates, not law
    "nda_template",
]

# Sources only returned when their topic keyword appears in the clause
CONDITIONAL_SOURCES = {
    "investment": ["investment", "invest", "foreign direct"],    # Investment Law
    "competition": [                                              # Competition Law
        "monopoly", "dominant", "price fixing", "market share",
        "anti-competitive", "cartel", "economic concentration",
        "merger", "acquisition", "abuse of dominance"
    ],
}


class EgyptianLegalRAG:
    """
    Production-grade RAG for Egyptian legal contract analysis.
    Supports smart conditional retrieval and relevance filtering.
    """

    # Similarity score threshold — results ABOVE this are dropped (0=identical, 2=very different)
    # Tightened from 1.2 → 0.68 based on output analysis:
    # Scores 0.43-0.68 = genuinely relevant
    # Scores 0.70+     = noise (Investment Law, NDA templates, off-topic articles)
    RELEVANCE_THRESHOLD = 0.68

    def __init__(self, data_folder: str, db_path: str = "./legal_rag_db"):
        self.data_folder = Path(data_folder)
        self.db_path = db_path
        self.vectorstore = None
        self._hash_file = Path(db_path) / "_pdf_hash.json"

        # Multilingual embeddings — handles Arabic/English legal text well
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
        """Hash all PDFs in data folder to detect changes"""
        hasher = hashlib.md5()
        pdf_files = sorted(self.data_folder.glob("**/*.pdf"))
        for pdf in pdf_files:
            hasher.update(pdf.name.encode())
            hasher.update(str(pdf.stat().st_size).encode())
        return hasher.hexdigest()

    def _is_db_current(self) -> bool:
        """Check if existing DB matches current PDFs"""
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
        """Save current PDF hash after building DB"""
        self._hash_file.write_text(
            json.dumps({"hash": self._compute_pdf_hash()})
        )

    def rebuild_database(self):
        """Force rebuild — deletes old DB and rebuilds from PDFs"""
        if Path(self.db_path).exists():
            print("🗑️  Deleting old database...")
            shutil.rmtree(self.db_path)
        self._build_database()

    def _build_database(self):
        """Build vector database from all PDFs in data folder"""
        print("\n🔧 Building database from PDFs...")

        pdf_files = list(self.data_folder.glob("**/*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_folder}")

        print(f"📁 Found {len(pdf_files)} PDF files:")
        for f in pdf_files:
            print(f"   • {f.name}")

        # Load documents
        loader = DirectoryLoader(
            str(self.data_folder),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"\n✅ Loaded {len(documents)} pages total")

        # Filter blank pages before chunking
        documents = [
            d for d in documents
            if len(d.page_content.strip()) > 150
        ]
        print(f"✅ After filtering blank pages: {len(documents)} pages")

        # Smaller chunks = cleaner article boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\nArticle", "\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        # Build Chroma with proper embeddings
        print("🗄️  Building vector database...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )

        self._save_pdf_hash()
        print(f"✅ Database ready — saved to {self.db_path}\n")

    def initialize(self):
        """Initialize or load database — auto-rebuilds if PDFs changed"""
        if self._is_db_current():
            print(f"📂 Loading existing database...")
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
    #  CLAUSE TRIGGER DETECTION
    # ──────────────────────────────────────────────

    def _detect_triggers(self, text: str) -> List[str]:
        """
        Detect which legal categories are relevant to this clause/contract.
        Returns list of triggered category names.
        """
        text_lower = text.lower()
        triggered = []
        for category, keywords in TRIGGER_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                triggered.append(category)
        return triggered

    # ──────────────────────────────────────────────
    #  TARGETED LAW RETRIEVAL
    # ──────────────────────────────────────────────

    def _search_with_score(self, query: str, k: int = 4) -> List[Tuple]:
        """Search and return (doc, score) pairs"""
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def _is_source_allowed(self, source: str, clause_text: str) -> bool:
        """
        Check if a source file should be included given the clause text.
        Blocks NDA templates always.
        Blocks Investment Law unless clause mentions investment topics.
        """
        source_lower = source.lower()

        # Always block template files
        for blocked in BLOCKED_SOURCES:
            if blocked in source_lower:
                return False

        # Block conditional sources unless their topic appears in clause
        for source_keyword, topic_triggers in CONDITIONAL_SOURCES.items():
            if source_keyword in source_lower:
                clause_lower = clause_text.lower()
                if not any(t in clause_lower for t in topic_triggers):
                    return False

        return True

    def _get_laws_for_category(self, category: str, clause_text: str = "") -> List[Tuple]:
        """
        Get laws relevant to a specific legal category.

        Per-category thresholds (tuned from output analysis):
          - liability_warranty : 0.65  — correct laws (Art 651-654) score ~0.61-0.64,
                                         but noisy results also score ~0.61-0.66,
                                         so we rely on SPECIFIC queries to find them
          - payment_terms      : 0.72  — correct laws exist just above 0.68
          - delivery_timeline  : 0.69  — tightened from 0.72: Laws #1,#2 score 0.53/0.58,
                                         employment law noise scored 0.71 — cut it out
          - all others         : 0.68  — default
        """

        category_thresholds = {
            "arbitration_jurisdiction": 0.68,
            "data_privacy":             0.68,
            "electronic_signature":     0.68,
            "intellectual_property":    0.68,
            "payment_terms":            0.72,
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
            # Queries anchor directly on Article numbers and exact content phrases
            # to ensure Articles 651-654 are retrieved and noise is excluded
            "liability_warranty": [
                "article 651 architect contractor jointly liable ten years collapse buildings",
                "civil law defect buildings contractor ten year guarantee period egypt",
                "article 653 clause exempt limit liability architect contractor null void",
                "article 654 claims guarantee lapse three years collapse discovery defect",
                "contractor liable defects strength safety buildings civil law egypt"
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
                if score > threshold:
                    continue
                source = doc.metadata.get("source", "")
                if not self._is_source_allowed(source, clause_text):
                    continue
                content_key = doc.page_content[:80]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    results.append((doc, score))

        return results

    # ──────────────────────────────────────────────
    #  MAIN ANALYSIS
    # ──────────────────────────────────────────────

    def analyze_clause(self, clause_text: str, verbose: bool = True) -> Dict:
        """
        Analyze a single contract clause.
        Returns dict with: triggered_categories, laws, formatted_output
        """
        if verbose:
            print("\n" + "="*80)
            print("CLAUSE ANALYSIS")
            print("="*80)
            print(f"\n📝 Clause: {clause_text[:200]}{'...' if len(clause_text)>200 else ''}")

        # Step 1: Detect relevant legal categories
        triggered = self._detect_triggers(clause_text)

        if verbose:
            if triggered:
                print(f"\n🎯 Triggered categories: {', '.join(triggered)}")
            else:
                print("\n🎯 No specific triggers — using general similarity search")

        # Step 2: Retrieve laws for each triggered category
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
            # Fallback: direct similarity search
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

        # Step 3: Sort by relevance score (lower = more similar)
        all_results.sort(key=lambda x: x[1])

        # Step 4: Cap at 8 laws max
        all_results = all_results[:8]

        if verbose:
            print(f"\n✅ Retrieved {len(all_results)} relevant law(s) "
                  f"(threshold={self.RELEVANCE_THRESHOLD})\n")

        # Step 5: Format output
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
        """
        Analyze a full contract by splitting into clause sections first.
        Returns combined results across all clauses.
        """
        print("\n" + "="*80)
        print("FULL CONTRACT ANALYSIS")
        print("="*80)
        print(f"\n📄 Contract size: {len(contract_text):,} characters")

        # Extract individual clauses
        clauses = self._extract_clauses(contract_text)
        print(f"✂️  Extracted {len(clauses)} clause(s)")

        # Detect triggers across whole contract
        all_triggers = self._detect_triggers(contract_text)
        print(f"🎯 Contract-level triggers: {', '.join(all_triggers) if all_triggers else 'none'}\n")

        # Analyze each clause
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

        # Sort by score, take top max_laws
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
        """Extract numbered clauses or fallback to paragraph chunks"""
        clauses = []

        # Try to match numbered clauses like "1.1", "2.3", etc.
        pattern = r'(\d+\.\d+)\s+(.+?)(?=\d+\.\d+|\Z)'
        matches = re.findall(pattern, contract_text, re.DOTALL)

        if matches:
            for clause_id, content in matches:
                content = content.strip()
                if len(content) > 30:
                    clauses.append({"id": clause_id, "text": content})
        else:
            # Fallback: split by double newlines
            paragraphs = [p.strip() for p in contract_text.split('\n\n')
                         if len(p.strip()) > 30]
            for i, para in enumerate(paragraphs):
                clauses.append({"id": str(i + 1), "text": para})

        return clauses if clauses else [{"id": "1", "text": contract_text}]

    # ──────────────────────────────────────────────
    #  FORMATTING
    # ──────────────────────────────────────────────

    def _format_output(self, results: List[Tuple]) -> str:
        """Format (doc, score, category) tuples into readable output"""
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
        """Format law dicts into readable output"""
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
    """Get multi-line input ending with DONE"""
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
    print("EGYPTIAN LEGAL CONTRACT ANALYZER v4.1")
    print("Tighter triggers | Source filtering | Threshold 0.68 | Multilingual embeddings")
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