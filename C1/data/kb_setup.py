"""
kb_setup.py - Build the CRRA policy knowledge base in ChromaDB.

1. Reads every .md file in data/kb/
2. Splits each file into chunks at '## ' headings (the '# ' title line is skipped)
3. Loads the chunks into the 'crra_policy' collection (recreated on every run)
4. Prints per-file chunk counts and a total
5. Runs sample BizOps queries and prints the best match for each

Usage (from the project root):
    python .\\data\\kb_setup.py
"""

import re
import sys
from pathlib import Path

import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
KB_DIR = SCRIPT_DIR / "kb"                 # data/kb/
CHROMA_DIR = SCRIPT_DIR / "chroma_db"      # data/chroma_db/ (persistent store)
COLLECTION_NAME = "crra_policy"

TEST_QUERIES = [
    "Who approves a 60 lakh contract?",
    "Can we award a contract to a single vendor without competitive bidding?",
    "What checks are required before onboarding a vendor that handles customer data?",
    "How much notice do we need to give before a contract auto-renews?",
    "What happens if a vendor repeatedly misses its SLA?",
    "can I merge 2 contracts",
    "What approval does 110 lakh overdue contract require",
]

H2_PATTERN = re.compile(r"^##\s+(.+?)\s*#*\s*$")   # matches '## Heading' only, not '###'


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def split_into_chunks(md_path: Path) -> list[dict]:
    """Split one markdown file into chunks, one per '## ' section.

    Anything before the first '## ' heading (the '# ' title and any intro
    text) is skipped. '###' sub-headings stay inside their parent chunk.
    """
    text = md_path.read_text(encoding="utf-8")
    chunks = []
    heading = None
    body_lines: list[str] = []

    def flush():
        if heading is None:
            return
        body = "\n".join(body_lines).strip()
        if not body:
            return  # skip headings with no content
        chunks.append({
            "heading": heading,
            # Keep the heading in the text too - it improves retrieval.
            "text": f"## {heading}\n\n{body}",
        })

    for line in text.splitlines():
        match = H2_PATTERN.match(line)
        if match:
            flush()
            heading = match.group(1).strip()
            body_lines = []
        elif heading is not None:
            body_lines.append(line)
        # else: still in the title / preamble -> ignore

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not KB_DIR.is_dir():
        sys.exit(f"ERROR: knowledge base folder not found: {KB_DIR}")

    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        sys.exit(f"ERROR: no .md files found in {KB_DIR}")

    # --- 1 & 2: read and chunk -------------------------------------------
    ids, documents, metadatas = [], [], []
    per_file_counts: dict[str, int] = {}

    for md_path in md_files:
        chunks = split_into_chunks(md_path)
        per_file_counts[md_path.name] = len(chunks)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{md_path.stem}::{idx:02d}")
            documents.append(chunk["text"])
            metadatas.append({
                "source": md_path.name,
                "heading": chunk["heading"],
                "chunk_index": idx,
            })

    if not documents:
        sys.exit("ERROR: no '## ' sections found in any file - nothing to load.")

    # --- 3: (re)create the collection and load ---------------------------
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass  # collection did not exist yet

    # Cosine distance, so (1 - distance) is the cosine similarity.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # --- 4: chunk counts -------------------------------------------------
    print(f"\nLoaded into '{COLLECTION_NAME}' at {CHROMA_DIR}\n")
    name_width = max(len(name) for name in per_file_counts) + 2
    print(f"{'File':<{name_width}}Chunks")
    print("-" * (name_width + 6))
    for name, count in per_file_counts.items():
        print(f"{name:<{name_width}}{count:>6}")
    print("-" * (name_width + 6))
    print(f"{'TOTAL':<{name_width}}{len(documents):>6}")
    print(f"(collection.count() = {collection.count()})")

    # --- 5: test queries -------------------------------------------------
    print("\n" + "=" * 70)
    print("Test queries (best match)")
    print("=" * 70)

    results = collection.query(
        query_texts=TEST_QUERIES,
        n_results=1,
        include=["metadatas", "distances", "documents"],
    )

    for i, query in enumerate(TEST_QUERIES):
        meta = results["metadatas"][i][0]
        distance = results["distances"][i][0]
        confidence = 1 - distance
        snippet = results["documents"][i][0].replace("\n", " ")[:120]

        print(f"\nQ{i + 1}: {query}")
        print(f"    Source     : {meta['source']}")
        print(f"    Section    : {meta['heading']}")
        print(f"    Confidence : {confidence:.3f}")
        print(f"    Snippet    : {snippet}...")


if __name__ == "__main__":
    main()