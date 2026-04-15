import os
from datetime import date
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_CONFIG = [
    {
        "path": "Python.pdf",           
        "source_type": "research_paper",
    },
    {
        "path": "java.pdf",          
        "source_type": "textbook",
    },
]



def load_pdfs(pdf_configs: list[dict]) -> list:
    
    all_loaded = []

    for config in pdf_configs:
        path = config["path"]

        if not os.path.exists(path):
            print(f"  File not found: {path} — skipping.")
            continue

        print(f" Loading: {path}")
        loader = PyPDFLoader(path)
        documents = loader.load()
        print(f"    Loaded {len(documents)} page(s)")

        all_loaded.append((documents, config))

    return all_loaded



def split_documents(documents: list) -> list:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks



def attach_metadata(chunks: list, config: dict) -> list:
 
    filename    = os.path.basename(config["path"])
    source_type = config["source_type"]
    upload_date = str(date.today())         # e.g. "2025-07-14"

    for chunk in chunks:
        chunk.metadata["filename"]    = filename
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1  # 1-based
        chunk.metadata["upload_date"] = upload_date
        chunk.metadata["source_type"] = source_type

    return chunks




def filter_chunks(chunks: list, **filters) -> list:
  
    if not filters:
        print("  No filters provided — returning all chunks.")
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




def print_chunks(chunks: list, label: str = "Results", max_display: int = 3) -> None:
   
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



def main():
    print("\n" + "═"*60)
    print("   RAG Bootcamp Day 2 — Document Loader & Filter")
    print("   Nunnari Academy")
    print("═"*60)

   
    print("\n STEP 1: Loading PDFs...")
    loaded_pairs = load_pdfs(PDF_CONFIG)

    if not loaded_pairs:
        print("\n No PDFs were loaded.")
        print("   Please update the 'path' values in PDF_CONFIG")
        print("   to point to real PDF files on your computer.\n")
        return

    print("\n  STEP 2 & 3: Splitting and attaching metadata...")
    all_chunks = []

    for documents, config in loaded_pairs:
        chunks = split_documents(documents)
        chunks = attach_metadata(chunks, config)
        all_chunks.extend(chunks)
        print(f"    {os.path.basename(config['path'])} → {len(chunks)} chunk(s)")

    print(f"\n    Total chunks across all PDFs: {len(all_chunks)}")

    print("\n STEP 4 & 5: Testing filter_chunks()...")

    first_filename  = os.path.basename(PDF_CONFIG[0]["path"])
    second_filename = os.path.basename(PDF_CONFIG[1]["path"]) if len(PDF_CONFIG) > 1 else None

    test1 = filter_chunks(all_chunks, filename=first_filename)
    print_chunks(test1, label=f"Filter → filename='{first_filename}'")

    if second_filename:
        test2 = filter_chunks(all_chunks, filename=second_filename)
        print_chunks(test2, label=f"Filter → filename='{second_filename}'")

    test3 = filter_chunks(all_chunks, page_number=1)
    print_chunks(test3, label="Filter → page_number=1")

    test4 = filter_chunks(all_chunks, source_type="research_paper")
    print_chunks(test4, label="Filter → source_type='research_paper'")

    test5 = filter_chunks(all_chunks, source_type="textbook")
    print_chunks(test5, label="Filter → source_type='textbook'")

    test6 = filter_chunks(all_chunks, filename=first_filename, page_number=1)
    print_chunks(test6, label=f"Filter → filename='{first_filename}' AND page_number=1")

    today = str(date.today())
    test7 = filter_chunks(all_chunks, upload_date=today)
    print_chunks(test7, label=f"Filter → upload_date='{today}'")

    print(f"\n{'═'*60}")
    print("    All exercises completed successfully!")
    print(f"    Total chunks loaded   : {len(all_chunks)}")
    print(f"    PDFs processed        : {len(loaded_pairs)}")
    print(f"    Filter tests run      : 7")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
