import os
import numpy as np
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_chroma import Chroma
from langchain_docling.loader import ExportType

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .embedding import CustomEmbeddings
from .graph_builder import GraphRetrival
from typing import List, Dict
from collections import defaultdict

clean_title = lambda x : x.split("#")[2].strip()



class PDFQnA:
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

        graph_retrival = GraphRetrival()

        self.retrieval_cache = {}
 
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

    def expand_query(self, query: str) -> List[str]:
        """Expand query using LLM paraphrases with embedding-based filtering."""
        # Pending LLM integration
        return [query]

    def _rank_with_diversity(self, results: List[Dict], top_n: int) -> List[Dict]:
        """Rank results with diversity consideration."""
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Apply diversity bonus
        selected = []
        source_counts = defaultdict(int)
        
        vec_candidates = [r for r in results if r.get('source') == 'vector_db']
        graph_candidates = [r for r in results if r.get('source') == 'citation_graph']
        print(f"DEBUG: Candidates before diversity - vector: {len(vec_candidates)}, graph: {len(graph_candidates)}")
        print(f"DEBUG: {vec_candidates=} {graph_candidates=}")
        for result in results:
            source = result['source']
            # Diversity penalty
            diversity_penalty = source_counts[source] * 0.1
            adjusted_score = result['relevance_score'] - diversity_penalty
            result['relevance_score'] = max(0, adjusted_score)
            selected.append(result)
            source_counts[source] += 1
            if len(selected) >= top_n:
                break

        # Guarantee at least one vector_db result if available
        if not any(r.get('source') == 'vector_db' for r in selected) and vec_candidates:
            best_vec = vec_candidates[0]
            # Replace the weakest non-vector if list is full, else append
            if len(selected) >= top_n:
                weakest_idx = min(range(len(selected)), key=lambda i: selected[i]['relevance_score'])
                selected[weakest_idx] = best_vec
            else:
                selected.append(best_vec)

        print(f"DEBUG: Selected after diversity - vector: {sum(1 for r in selected if r.get('source')=='vector_db')}, graph: {sum(1 for r in selected if r.get('source')=='citation_graph')}")
        print(f"DEBUG: {selected=}")
        return selected


    def fuse_and_rank_results(self, vector_results, graph_results, query_embedding, top_n=5):
        """Enhanced fusion with better scoring and diversity."""
        fused_results = []
        
        # Process vector results with enhanced scoring
        for i in range(len(vector_results['documents'][0])):
            distance = vector_results['distances'][0][i]
            content = vector_results['documents'][0][i]
            metadata = vector_results['metadatas'][0][i]
            
            # Base relevance score
            base_score = 1 - distance
            
            # Boost for recent papers
            year = metadata.get('publication_year')
            if isinstance(year, (int, float)) and year > 2020:
                base_score *= 1.1
            
            # Boost for longer content (more informative)
            content_length_bonus = min(len(content) / 1000, 0.2)  # Max 20% bonus
            base_score += content_length_bonus
            
            fused_results.append({
                "source": "vector_db",
                "content": content,
                "metadata": metadata,
                "relevance_score": min(1.0, base_score)
            })
        
        # Process graph results with enhanced scoring
        for paper in graph_results:
            label = paper.get('label', '')
            
            # Use the pre-computed scores from advanced retrieval
            relevance_score = paper.get('score', 0)
            
            # Add metadata from citation data
            paper_metadata = {
                "openalex_id": paper.get('id'),
                "type": paper.get('type'),
                "cited_by_count": paper.get('cited_by_count', 0),
                "publication_year": paper.get('publication_year'),
                "pagerank_score": paper.get('pagerank_score', 0),
                "betweenness_score": paper.get('betweenness_score', 0)
            }
            
            fused_results.append({
                "source": "citation_graph",
                "content": f"Title: {label}",
                "metadata": paper_metadata,
                "relevance_score": relevance_score
            })
        
        # Enhanced ranking with diversity
        ranked_results = self._rank_with_diversity(fused_results, top_n)
        return ranked_results

    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Re-rank results using cross-encoder or other methods."""
        # Simple re-ranking based on multiple factors
        for result in results:
            # Boost score for exact matches
            if query.lower() in result.get("content", "").lower():
                result["relevance_score"] *= 1.2
            
            # Boost score for recent papers
            year = result.get("metadata", {}).get("publication_year")
            if isinstance(year, (int, float)) and year > 2020:
                result["relevance_score"] *= 1.1
        
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)

    def _query_personalized_paths(self, question: str, top_k: int = 3):
        """Compute query-aware citation paths using personalized PageRank and shortest paths.
        Returns a list of human-readable path strings with edge labels and weights.
        """
        try:
            import numpy as np
            import networkx as nx
        except Exception:
            return []

        G = getattr(self, 'G', None)
        if G is None or not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
            return []

        # Identify root node
        root_candidates = [n for n, d in G.nodes(data=True) if d.get('type') == 'root']
        root = root_candidates[0] if root_candidates else None
        if root is None:
            return []

        # Compute query embedding
        try:
            q_emb = np.array(self.embedding_model.embed_query(question), dtype=float)
            if q_emb.ndim != 1:
                q_emb = q_emb.flatten()
        except Exception:
            q_emb = None

        def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
            try:
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na == 0 or nb == 0:
                    return 0.0
                return float(np.dot(a, b) / (na * nb))
            except Exception:
                return 0.0

        # Personalization by similarity
        if q_emb is not None:
            personalization = {}
            for n, d in G.nodes(data=True):
                emb = d.get('embedding')
                if emb is None:
                    personalization[n] = 0.0
                else:
                    try:
                        personalization[n] = max(0.0, safe_cos(q_emb, np.array(emb, dtype=float)))
                    except Exception:
                        personalization[n] = 0.0
        else:
            personalization = {n: 1.0 for n in G.nodes}

        # Personalized PageRank
        try:
            pr = nx.pagerank(G, weight='weight', personalization=personalization)
        except Exception:
            pr = {n: 1.0 / max(G.number_of_nodes(), 1) for n in G.nodes}

        # Select anchors (exclude root)
        sorted_nodes = [n for n, _ in sorted(pr.items(), key=lambda kv: kv[1], reverse=True)]
        anchors = [n for n in sorted_nodes if n != root][: min(10, len(sorted_nodes))]

        # Shortest paths to root using inverse-weight cost
        def edge_cost(u, v, e):
            w = e.get('weight', 1.0)
            return 1.0 / (w + 1e-6)

        paths = []
        for a in anchors:
            try:
                p = nx.shortest_path(G, source=a, target=root, weight=lambda u, v, e: edge_cost(u, v, e))
                paths.append(p)
            except Exception:
                continue

        # Deduplicate and score
        seen = set()
        unique_paths = []
        for p in paths:
            key = tuple(p)
            if key not in seen:
                unique_paths.append(p)
                seen.add(key)

        def path_score(p):
            return sum(pr.get(n, 0.0) for n in p)

        unique_paths.sort(key=path_score, reverse=True)
        unique_paths = unique_paths[:top_k]

        # Render readable paths
        rendered = []
        for p in unique_paths:
            hops = []
            for i, n in enumerate(p):
                nd = G.nodes[n]
                title = nd.get('label', str(n))
                year = nd.get('publication_year')
                cited = nd.get('cited_by_count')
                parts = [title]
                suffix = []
                if isinstance(year, (int, float)):
                    suffix.append(str(int(year)))
                if isinstance(cited, (int, float)):
                    suffix.append(f"citations:{int(cited)}")
                if suffix:
                    parts.append(f"({', '.join(suffix)})")
                hops.append(" ".join(parts))

                if i < len(p) - 1:
                    u, v = p[i], p[i+1]
                    ed = G.get_edge_data(u, v) or {}
                    etype = ed.get('etype', 'edge')
                    w = ed.get('weight', 0.0)
                    hops.append(f" —{etype} (w={w:.2f})→ ")
            rendered.append("".join(hops))

        return rendered


    def ask_question(self, question: str, use_best_first: bool = False):
        if not self.vectorstore:
            raise Exception("No documents in the vectorstore.")
        
        # Check cache first
        processed_pdfs_str = str(sorted(self.processed_pdfs)) if hasattr(self, 'processed_pdfs') and self.processed_pdfs else "no_pdfs"
        cache_key = f"{question}_{hash(processed_pdfs_str)}"
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        # Expand query for better retrieval
        expanded_queries = self.expand_query(question)
        print("DEBUG : expanded quries", expanded_queries)

        # Get vector results with expanded queries
        all_vector_results = []
        for expanded_query in expanded_queries:
            try:
                vector_results = self.vectorstore.similarity_search_with_score(expanded_query, k=3)
            except Exception as e:
                print(f"Vector search error (with_score) for '{expanded_query}': {e}")
                vector_results = []
            all_vector_results.extend(vector_results)
        
        # Remove duplicates and get top results
        seen_content = set()
        unique_vector_results = []
        for doc, score in all_vector_results:
            if doc.page_content not in seen_content:
                unique_vector_results.append((doc, score))
                seen_content.add(doc.page_content)
        
        # Fallback if empty: try without scores
        if not unique_vector_results:
            try:
                fallback_docs = self.vectorstore.similarity_search(question, k=5)
                unique_vector_results = [(doc, 0.5) for doc in fallback_docs]
                print(f"DEBUG: Fallback vector retrieval used, {len(unique_vector_results)} docs")
            except Exception as e:
                print(f"Vector search error (fallback no-score): {e}")
                unique_vector_results = []

        # Convert to expected format for fusion
        vector_results_formatted = {
            'documents': [[doc.page_content for doc, _ in unique_vector_results[:5]]],
            'metadatas': [[doc.metadata for doc, _ in unique_vector_results[:5]]],
            'distances': [[score for _, score in unique_vector_results[:5]]]
        }
        print(f"DEBUG: Vector candidates kept: {len(vector_results_formatted['documents'][0])}")
        
        # # Get graph results (optionally use best-first traversal)
        # if use_best_first:
        #     try:
        #         graph_results = best_first_graph_retrieval(
        #             self.G,
        #             question,
        #             embedding_model=self.embedding_model,
        #             top_k=5,
        #             max_hops=3,
        #         )
        #     except Exception as e:
        #         print(f"Best-first retrieval failed, falling back to advanced retrieval: {e}")
        #         graph_results = self.advanced_graph_retrieval(question, top_k=5, max_hops=3)
        # else:
        graph_results = self.advanced_graph_retrieval(question, top_k=5, max_hops=3)
        query_embedding = self.embedding_model.embed_query(question)
        
        #  fusion with better weighting
        fused_results = self.fuse_and_rank_results(
            vector_results_formatted, graph_results, query_embedding, top_n=7
        )
        
        # Re-rank results
        fused_results = self.rerank_results(fused_results, question)
        
     

if __name__ == "__main__":

    # Example usage
    pdf_qa = PDFQnA()

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
