import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType


clean_title = lambda x : x.split("#")[2].strip()

class PDFLoad:
    def __init__(self):
        pass
        
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

        # TODO add documents to vector store and write functions for that and build a qa chain