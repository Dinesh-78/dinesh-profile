"""Profile data chunking and vector store indexer using LangChain Document abstractions."""

import sys
import json
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.config import settings
from src.rag.embeddings import get_embeddings
from src.rag.loader import ProfileDocumentLoader

logger = logging.getLogger(__name__)

class ProfileIndexer:
    """Chunks structured profile information into LangChain Documents and indexes them into ChromaDB."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.embeddings = get_embeddings()
        self.vector_store = None
        self.documents: List[Document] = []

    def load_and_chunk(self) -> List[Document]:
        """Extract and structure chunks as LangChain Document objects from profile JSON and Markdown."""
        docs: List[Document] = []

        # 1. Chunk structured JSON
        json_path = Path(settings.DATA_PATH)
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Personal info
            p_info = data.get("personal_info", {})
            
            docs.append(Document(
                page_content=(
                    f"Full Name: {p_info.get('full_name')} ({p_info.get('preferred_name')})\n"
                    f"Professional Title: {p_info.get('title')}\n"
                    f"Contact Email: {p_info.get('email')}\n"
                    f"GitHub: {p_info.get('github')}\n"
                    f"LinkedIn: {p_info.get('linkedin')}\n"
                    f"Location: {p_info.get('location')}\n"
                    f"Professional Summary: {p_info.get('summary')}"
                ),
                metadata={
                    "doc_id": "personal_info_01",
                    "section": "personal_info",
                    "title": "Dinesh Personal Info & Summary",
                    "source": "profile_data.json"
                }
            ))

            # Career preferences
            prefs = data.get("career_preferences", {})
            docs.append(Document(
                page_content=(
                    f"Target Roles: {', '.join(prefs.get('target_roles', []))}\n"
                    f"Work Preference: {prefs.get('work_preference')}\n"
                    f"Notice Period: {prefs.get('notice_period')}\n"
                    f"Relocation Preference: {prefs.get('relocation')}\n"
                    f"Preferred Domains: {', '.join(prefs.get('preferred_domains', []))}"
                ),
                metadata={
                    "doc_id": "career_preferences_01",
                    "section": "preferences",
                    "title": "Career Preferences & Availability",
                    "source": "profile_data.json"
                }
            ))

            # Technical skills
            skills = data.get("skills", {})
            for category, skill_list in skills.items():
                cat_name = category.replace("_", " ").title()
                docs.append(Document(
                    page_content=(
                        f"Skill Category: {cat_name}\n"
                        f"Key Technologies & Competencies: {', '.join(skill_list)}"
                    ),
                    metadata={
                        "doc_id": f"skill_{category}",
                        "section": "skills",
                        "category": category,
                        "title": f"Technical Skills - {cat_name}",
                        "source": "profile_data.json"
                    }
                ))

            # Work experience
            for i, exp in enumerate(data.get("work_experience", [])):
                resps = "\n- ".join(exp.get("responsibilities", []))
                docs.append(Document(
                    page_content=(
                        f"Role: {exp.get('role')}\n"
                        f"Company: {exp.get('company')} ({exp.get('period')}, {exp.get('location')})\n"
                        f"Responsibilities and Achievements:\n- {resps}"
                    ),
                    metadata={
                        "doc_id": f"work_exp_{i+1}",
                        "section": "experience",
                        "company": exp.get("company"),
                        "title": f"Work Experience at {exp.get('company')}",
                        "source": "profile_data.json"
                    }
                ))

            # Featured projects
            for i, proj in enumerate(data.get("featured_projects", [])):
                highlights = "\n- ".join(proj.get("key_highlights", []))
                docs.append(Document(
                    page_content=(
                        f"Project: {proj.get('title')}\n"
                        f"Tech Stack: {', '.join(proj.get('tech_stack', []))}\n"
                        f"Description: {proj.get('description')}\n"
                        f"Key Highlights:\n- {highlights}"
                    ),
                    metadata={
                        "doc_id": f"project_{i+1}",
                        "section": "projects",
                        "title": proj.get("title"),
                        "source": "profile_data.json"
                    }
                ))

            # Recruiter FAQ
            for i, faq in enumerate(data.get("recruiter_faq", [])):
                docs.append(Document(
                    page_content=(
                        f"Recruiter Question: {faq.get('question')}\n"
                        f"Verified Answer: {faq.get('answer')}"
                    ),
                    metadata={
                        "doc_id": f"faq_{i+1}",
                        "section": "faq",
                        "title": f"Recruiter FAQ: {faq.get('question')[:40]}...",
                        "source": "profile_data.json"
                    }
                ))

        # 2. Load all data files using LangChain DirectoryLoader
        from src.rag.loader import load_all_data_sources
        dir_docs = load_all_data_sources()
        docs.extend(dir_docs)

        self.documents = docs
        logger.info(f"Loaded and chunked {len(self.documents)} LangChain Document objects.")
        return docs

    def build_index(self):
        """Indexes all LangChain Documents into ChromaDB using LangChain vectorstore abstractions."""
        if not self.documents:
            self.load_and_chunk()

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Clear any stale collection to guarantee 100% fresh, accurate data
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            try:
                client.delete_collection(name="dinesh_profile")
                logger.info("Cleared prior Chroma collection for fresh re-indexing.")
            except Exception:
                pass
        except Exception:
            pass

        try:
            # Prefer LangChain's vectorstore integration
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                from langchain_chroma import Chroma

            ids = [doc.metadata.get("doc_id", f"doc_{i}") for i, doc in enumerate(self.documents)]

            vector_store = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="dinesh_profile",
                ids=ids
            )
            self.vector_store = vector_store
            logger.info(f"Successfully indexed {len(self.documents)} documents in Chroma via LangChain.")
            return vector_store
        except Exception as e:
            logger.warning(f"LangChain Chroma wrapper init note ({e}). Falling back to native chromadb client.")
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            collection = client.get_or_create_collection(
                name="dinesh_profile",
                metadata={"description": "Dinesh's Profile Knowledge Base"}
            )
            texts = [d.page_content for d in self.documents]
            metas = [d.metadata for d in self.documents]
            ids = [d.metadata.get("doc_id", f"doc_{i}") for i, d in enumerate(self.documents)]
            emb_list = self.embeddings.embed_documents(texts)
            collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=emb_list)
            self.vector_store = collection
            return collection

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 Initializing ProfileIndexer...")
    indexer = ProfileIndexer()
    docs = indexer.load_and_chunk()
    print(f"✅ Loaded {len(docs)} LangChain Documents.")
    indexer.build_index()
    print("✅ Successfully indexed into ChromaDB.")
