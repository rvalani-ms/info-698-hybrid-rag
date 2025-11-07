import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_chroma import Chroma
from langchain_docling.loader import ExportType
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

clean_title = lambda x : x.split("#")[2].strip()

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(documents)
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            else:
                return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"Error in embed_documents: {e}")
            # Return dummy embeddings
            return [[0.0] * 384 for _ in documents]

    def embed_query(self, query: str) -> List[float]:
        try:
            embedding = self.model.encode([query])
            # Ensure a flat list (shape: [dim]) regardless of backend
            if isinstance(embedding, (list, tuple)):
                # embedding is likely a list with a single inner list
                first = embedding[0]
                return first if isinstance(first, list) else list(first)
            if hasattr(embedding, 'shape') and getattr(embedding, 'ndim', 1) == 2:
                return embedding[0].tolist()
            if hasattr(embedding, 'tolist'):
                out = embedding.tolist()
                return out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
            # Fallback: coerce to list
            return list(embedding)
        except Exception as e:
            print(f"Error in embed_query: {e}")
            # Return dummy embedding
            return [0.0] * 384



class PDFLoad:
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 100):
                # Initialize embedding model with consistent dimensions
        # Use all-MiniLM-L6-v2 (384 dims) to match existing collections
        try:
            self.embedding_model = CustomEmbeddings(model_name="all-MiniLM-L6-v2")
            print("✅ Using all-MiniLM-L6-v2 (384 dimensions) for consistency")
        except Exception as e:
            print(f"Warning: Could not initialize all-MiniLM-L6-v2: {e}")
            # Try the other model as fallback
            try:
                self.embedding_model = CustomEmbeddings(model_name="all-mpnet-base-v2")
                print("⚠️  Using all-mpnet-base-v2 (768 dimensions) - may cause dimension mismatch")
            except Exception as e2:
                print(f"❌ Could not initialize any embedding model: {e2}")
                self.embedding_model = None

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            # length_function=len
        )

        # Initialize vectorstore with dimension handling
        try:
            self.vectorstore = Chroma(
                embedding_function=self.embedding_model,
                persist_directory="chroma",
                collection_name="pdf_qa",
            )
            print("✅ Vector store initialized successfully")
        except Exception as e:
            if "dimension" in str(e).lower() or "embedding" in str(e).lower():
                print(f"⚠️  Dimension mismatch detected: {e}")
                print("🔄 Clearing existing collection and creating new one...")
                try:
                    # Clear the existing collection
                    import shutil
                    if os.path.exists("chroma"):
                        shutil.rmtree("chroma")
                        print("✅ Cleared existing ChromaDB collection")

                    # Create new collection
                    self.vectorstore = Chroma(
                        embedding_function=self.embedding_model,
                        persist_directory="chroma",
                        collection_name="pdf_qa",
                    )
                    print("✅ New vector store created with correct dimensions")
                except Exception as e2:
                    print(f"❌ Could not create new vector store: {e2}")
                    self.vectorstore = None
            else:
                print(f"❌ Vector store initialization failed: {e}")
                self.vectorstore = None
        # Initialize tracking attributes
        self.processed_pdfs = set()


    def load_documents_from_dir(self, directory: str):
        # Load documents from the specified directory
        document_loader = PyPDFDirectoryLoader(directory)
        documents = document_loader.load()
        processed_files_for_titles = set() # To avoid processing title for every page of the same file

        print("\n--- Extracted Titles from Directory ---")
        for doc in documents:
            source_file = doc.metadata.get('source')
            if source_file and source_file not in processed_files_for_titles:
                title = doc.metadata.get('title')
                base_filename = os.path.basename(source_file)
                if title:
                    print(f"File: {base_filename}, Title (from metadata): {title}")
                    self.extracted_titles[base_filename] = title
                else:
                    # Fallback: use filename (without extension) if title metadata is missing
                    filename_title = os.path.splitext(base_filename)[0].replace('_', ' ').replace('-', ' ')
                    print(f"File: {base_filename}, Title (from filename): {filename_title}")
                    self.extracted_titles[base_filename] = filename_title
                processed_files_for_titles.add(source_file)
        if not documents:
            print("No PDF documents found or loaded from the directory.")
        print("-------------------------------------\n")

        return documents

    def load_document(self, file_path: str):
        try:
            # Try DoclingLoader first
            l = DoclingLoader(file_path, export_type=ExportType.MARKDOWN).load()

            if l and len(l) > 0:
                # Extract title
                title = clean_title(l[0].page_content.split("\n")[0])
                print("===========================")
                print("TITLE : ", title)
                print("===========================")

                # Update metadata
                for doc in l:
                    doc.metadata.update({"title": title})
                return l
            else:
                print(f"Warning: DoclingLoader returned empty results for {file_path}")
                return []

        except Exception as e:
            print(f"Error with DoclingLoader for {file_path}: {e}")
            print("Falling back to PyPDFLoader...")

            try:
                # Fallback to PyPDFLoader
                from langchain_community.document_loaders import PyPDFLoader
                document_loader = PyPDFLoader(file_path, extract_images=False)
                documents = document_loader.load()

                # Extract title from filename as fallback
                import os
                title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').replace('-', ' ')

                # Update metadata
                for doc in documents:
                    doc.metadata.update({"title": title})

                print(f"Loaded {len(documents)} pages using PyPDFLoader")
                return documents

            except Exception as e2:
                print(f"Error with PyPDFLoader for {file_path}: {e2}")
                return []


    def add_documents(self, documents, metadata={}):
        if documents:
            print(f"Adding {len(documents)} documents to vector store")
            print(f"First document metadata: {documents[0].metadata}")

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")

            # Add chunks to vectorstore
            if self.vectorstore:
                self.vectorstore.add_documents(chunks)
                print("✅ Documents added to vector store")
            else:
                print("⚠️  Vector store not available")

            # Update processed PDFs tracking
            if hasattr(documents[0], 'metadata') and 'source' in documents[0].metadata:
                source_file = documents[0].metadata['source']
                if source_file:
                    self.processed_pdfs.add(source_file)
                    print(f"📝 Added to processed PDFs: {source_file}")
        else:
            raise Exception("No documents to add.")


if __name__ == "__main__":

    # Example usage
    pdf_qa = PDFLoad()

    # Load documents from a directory
    # documents = pdf_qa.load_documents_from_dir("./papers")

    # Load each pdf file separately
    papers_dir = "./../papers"
    import glob
    pdf_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
    for document in pdf_files:
        print(f"Processing file: {document}")
        print("==========================")
        # print(f"Loading document: {document}")
        documents= pdf_qa.load_document(document)

        pdf_qa.add_documents(documents)
