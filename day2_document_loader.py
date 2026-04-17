"""
╔══════════════════════════════════════════════════════════════╗
║         RAG Bootcamp — Day 2 Task                            ║
║         Document Loader, Splitter, Metadata & Filter         ║
║         Nunnari Academy                                       ║
╚══════════════════════════════════════════════════════════════╝

Exercises Covered:
  1. Load PDFs using PyPDFLoader
  2. Split into chunks using RecursiveCharacterTextSplitter
  3. Attach metadata (filename, page_number, upload_date, source_type)
  4. Build a filter_chunks() function
  5. Test the filters
"""

import os
from datetime import date
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────────────────────
# PDF Configuration
# Define your 2 PDF files and their metadata here
# Just change the "path" to point to your actual PDF files
# ─────────────────────────────────────────────────────────────

PDF_CONFIG = [
    {
        "path": "paper1.pdf",           # ← Change to your PDF file path
        "source_type": "research_paper",
    },
    {
        "path": "paper2.pdf",           # ← Change to your PDF file path
        "source_type": "textbook",
    },
]


# ─────────────────────────────────────────────────────────────
# Exercise 1 — Load PDFs
# ─────────────────────────────────────────────────────────────

def load_pdfs(pdf_configs: list[dict]) -> list:
    """
    Load all PDF files using PyPDFLoader.
    Returns a list of (documents, config) tuples.
    """
    all_loaded = []

    for config in pdf_configs:
        path = config["path"]

        if not os.path.exists(path):
            print(f"⚠️  File not found: {path} — skipping.")
            continue

        print(f"📄 Loading: {path}")
        loader = PyPDFLoader(path)
        documents = loader.load()
        print(f"   ✅ Loaded {len(documents)} page(s)")

        all_loaded.append((documents, config))

    return all_loaded


# ─────────────────────────────────────────────────────────────
# Exercise 2 — Split into Chunks
# ─────────────────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    chunk_size=1000, chunk_overlap=200 as required.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ─────────────────────────────────────────────────────────────
# Exercise 3 — Attach Metadata
# ─────────────────────────────────────────────────────────────

def attach_metadata(chunks: list, config: dict) -> list:
    """
    Attach the following metadata to each chunk:
      - filename      : name of the PDF file
      - page_number   : page the chunk came from
      - upload_date   : today's date
      - source_type   : type of document (paper, textbook, notes, etc.)
    """
    filename    = os.path.basename(config["path"])
    source_type = config["source_type"]
    upload_date = str(date.today())         # e.g. "2025-07-14"

    for chunk in chunks:
        chunk.metadata["filename"]    = filename
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1  # 1-based
        chunk.metadata["upload_date"] = upload_date
        chunk.metadata["source_type"] = source_type

    return chunks


# ─────────────────────────────────────────────────────────────
# Exercise 4 — Filter Function
# ─────────────────────────────────────────────────────────────

def filter_chunks(chunks: list, **filters) -> list:
    """
    Return only the chunks that match ALL given metadata filters.

    Usage examples:
      filter_chunks(chunks, filename="paper1.pdf")
      filter_chunks(chunks, filename="paper1.pdf", page_number=3)
      filter_chunks(chunks, source_type="textbook")
      filter_chunks(chunks, upload_date="2025-07-14")
    """
    if not filters:
        print("⚠️  No filters provided — returning all chunks.")
        return chunks

    result = []
    for chunk in chunks:
        match = all(
            chunk.metadata.get(key) == value
            for key, value in filters.items()
        )
        if match:
            result.append(chunk)

    return result


# ─────────────────────────────────────────────────────────────
# Helper — Pretty print chunks
# ─────────────────────────────────────────────────────────────

def print_chunks(chunks: list, label: str = "Results", max_display: int = 3) -> None:
    """Print chunk details in a readable format."""
    print(f"\n{'─'*60}")
    print(f"  {label}  →  {len(chunks)} chunk(s) found")
    print(f"{'─'*60}")

    for i, chunk in enumerate(chunks[:max_display], start=1):
        m = chunk.metadata
        print(f"\n  Chunk #{i}")
        print(f"  ├─ filename    : {m.get('filename')}")
        print(f"  ├─ page_number : {m.get('page_number')}")
        print(f"  ├─ upload_date : {m.get('upload_date')}")
        print(f"  ├─ source_type : {m.get('source_type')}")
        print(f"  └─ text preview: {chunk.page_content[:120].strip()}...")

    if len(chunks) > max_display:
        print(f"\n  ... and {len(chunks) - max_display} more chunk(s).")


# ─────────────────────────────────────────────────────────────
# Exercise 5 — Main: Load → Split → Metadata → Filter → Test
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("   RAG Bootcamp Day 2 — Document Loader & Filter")
    print("   Nunnari Academy")
    print("═"*60)

    # ── Step 1: Load PDFs ──────────────────────────────────────
    print("\n📂 STEP 1: Loading PDFs...")
    loaded_pairs = load_pdfs(PDF_CONFIG)

    if not loaded_pairs:
        print("\n❌ No PDFs were loaded.")
        print("   Please update the 'path' values in PDF_CONFIG")
        print("   to point to real PDF files on your computer.\n")
        return

    # ── Step 2 & 3: Split + Attach Metadata ───────────────────
    print("\n✂️  STEP 2 & 3: Splitting and attaching metadata...")
    all_chunks = []

    for documents, config in loaded_pairs:
        chunks = split_documents(documents)
        chunks = attach_metadata(chunks, config)
        all_chunks.extend(chunks)
        print(f"   ✅ {os.path.basename(config['path'])} → {len(chunks)} chunk(s)")

    print(f"\n   📦 Total chunks across all PDFs: {len(all_chunks)}")

    # ── Step 4 & 5: Test the filter function ──────────────────
    print("\n🔍 STEP 4 & 5: Testing filter_chunks()...")

    # Get real values from loaded data for testing
    first_filename  = os.path.basename(PDF_CONFIG[0]["path"])
    second_filename = os.path.basename(PDF_CONFIG[1]["path"]) if len(PDF_CONFIG) > 1 else None

    # Test 1 — Filter by filename (first PDF)
    test1 = filter_chunks(all_chunks, filename=first_filename)
    print_chunks(test1, label=f"Filter → filename='{first_filename}'")

    # Test 2 — Filter by filename (second PDF)
    if second_filename:
        test2 = filter_chunks(all_chunks, filename=second_filename)
        print_chunks(test2, label=f"Filter → filename='{second_filename}'")

    # Test 3 — Filter by page_number = 1
    test3 = filter_chunks(all_chunks, page_number=1)
    print_chunks(test3, label="Filter → page_number=1")

    # Test 4 — Filter by source_type
    test4 = filter_chunks(all_chunks, source_type="research_paper")
    print_chunks(test4, label="Filter → source_type='research_paper'")

    # Test 5 — Filter by source_type textbook
    test5 = filter_chunks(all_chunks, source_type="textbook")
    print_chunks(test5, label="Filter → source_type='textbook'")

    # Test 6 — Multiple filters combined
    test6 = filter_chunks(all_chunks, filename=first_filename, page_number=1)
    print_chunks(test6, label=f"Filter → filename='{first_filename}' AND page_number=1")

    # Test 7 — Filter by upload_date
    today = str(date.today())
    test7 = filter_chunks(all_chunks, upload_date=today)
    print_chunks(test7, label=f"Filter → upload_date='{today}'")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  ✅  All exercises completed successfully!")
    print(f"  📦  Total chunks loaded   : {len(all_chunks)}")
    print(f"  📄  PDFs processed        : {len(loaded_pairs)}")
    print(f"  🔍  Filter tests run      : 7")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
