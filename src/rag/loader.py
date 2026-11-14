"""Document loader module implementing LangChain DirectoryLoader and TextLoader."""

import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path so loader.py can be executed directly from any folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

class ProfileDirectoryLoader:
    """Loads and splits all profile data files from a directory using LangChain DirectoryLoader."""

    def __init__(self, dir_path: Optional[str] = None):
        if dir_path:
            p = Path(dir_path)
            self.dir_path = p if p.is_absolute() else (PROJECT_ROOT / p)
        else:
            # Default to src/data if it exists, otherwise data/
            candidates = [
                PROJECT_ROOT / "src" / "data",
                PROJECT_ROOT / "data"
            ]
            self.dir_path = next((c for c in candidates if c.exists()), candidates[0])

    def load_documents(self) -> List[Document]:
        """Loads all documents from the directory using LangChain DirectoryLoader."""
        if not self.dir_path.exists():
            logger.warning(f"Directory not found: {self.dir_path}")
            return []

        all_docs: List[Document] = []

        # 1. Load all text and markdown files via DirectoryLoader + TextLoader
        for pattern in ["**/*.txt", "**/*.md", "**/*.json"]:
            try:
                text_dir_loader = DirectoryLoader(
                    path=str(self.dir_path),
                    glob=pattern,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                    silent_errors=True
                )
                loaded = text_dir_loader.load()
                if loaded:
                    logger.info(f"DirectoryLoader loaded {len(loaded)} file(s) for pattern '{pattern}'")
                    all_docs.extend(loaded)
            except Exception as e:
                logger.warning(f"DirectoryLoader pattern '{pattern}' note: {e}")

        # 2. Try loading PDF files (e.g. resumes) if PyPDFLoader is available
        try:
            from langchain_community.document_loaders import PyPDFLoader
            pdf_dir_loader = DirectoryLoader(
                path=str(self.dir_path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                silent_errors=True
            )
            pdf_docs = pdf_dir_loader.load()
            print(pdf_docs)
            if pdf_docs:
                logger.info(f"DirectoryLoader loaded {len(pdf_docs)} page(s) from PDFs")
                all_docs.extend(pdf_docs)
        except ImportError:
            logger.info("pypdf not installed; skipping PDF files or install pypdf to parse .pdf")
        except Exception as e:
            logger.warning(f"DirectoryLoader PDF loading note: {e}")

        # Normalize metadata
        for doc in all_docs:
            src = doc.metadata.get("source", "")
            doc.metadata["filename"] = Path(src).name if src else "unknown"
            doc.metadata["file_type"] = Path(src).suffix if src else ""

        return all_docs

    def load_and_split(self) -> List[Document]:
        """Loads all documents from directory and splits into semantic chunks."""
        raw_docs = self.load_documents()
        if not raw_docs:
            return []

        # Split into readable chunks preserving context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        )
        split_chunks = text_splitter.split_documents(raw_docs)

        for i, chunk in enumerate(split_chunks, 1):
            chunk.metadata["doc_id"] = f"{chunk.metadata.get('filename', 'doc')}_chunk_{i}"
            if "section" not in chunk.metadata:
                chunk.metadata["section"] = "profile_knowledge"

        logger.info(f"DirectoryLoader loaded and split {len(raw_docs)} documents into {len(split_chunks)} chunks.")
        return split_chunks

# Backward compatibility alias
ProfileDocumentLoader = ProfileDirectoryLoader

def load_all_data_sources() -> List[Document]:
    """Helper to load from all existing data directories (both src/data and data)."""
    docs = []
    directories = [
        PROJECT_ROOT / "src" / "data",
        PROJECT_ROOT / "data"
    ]
    seen_files = set()
    for d in directories:
        if d.exists():
            loader = ProfileDirectoryLoader(dir_path=str(d))
            for chunk in loader.load_and_split():
                file_source = chunk.metadata.get("source")
                # Deduplicate if same file appears in multiple places
                key = (chunk.metadata.get("filename"), chunk.page_content.strip()[:60])
                if key not in seen_files:
                    seen_files.add(key)
                    docs.append(chunk)
    return docs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "="*60)
    print("📂 RUNNING LANGCHAIN DIRECTORY LOADER FOR PROFILE DATA")
    print("="*60)

    # Test loading from src/data
    src_data_dir = PROJECT_ROOT / "src" / "data"
    data_dir = PROJECT_ROOT / "data"

    target_dir = src_data_dir if src_data_dir.exists() else data_dir
    print(f"Target Directory: {target_dir}\n")

    loader = ProfileDirectoryLoader(dir_path=str(target_dir))
    raw_documents = loader.load_documents()

    print(f"📄 Total raw documents loaded by DirectoryLoader: {len(raw_documents)}")
    for i, doc in enumerate(raw_documents, 1):
        print(f"  [{i}] Source: {doc.metadata.get('filename')} | Length: {len(doc.page_content)} chars")

    print("\n✂️ Splitting documents with RecursiveCharacterTextSplitter...")
    chunks = loader.load_and_split()
    print(f"✅ Total chunks produced: {len(chunks)}")

    print("\n--- Preview of First 2 Chunks ---")
    for idx, c in enumerate(chunks[:2], 1):
        print(f"\n[Chunk #{idx}] Metadata: {c.metadata}")
        print(f"Content:\n{c.page_content.strip()}")
        print("-" * 40)
    print("="*60 + "\n")
