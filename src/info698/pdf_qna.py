import os
import numpy as np
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_docling import DoclingLoader

from langchain_chroma import Chroma
from langchain_docling.loader import ExportType
from sklearn.metrics.pairwise import cosine_similarity

from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding import CustomEmbeddings
from graph_builder import GraphRetrieval
from typing import List, Dict
from collections import defaultdict
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import simplejson as json

clean_title = lambda x : x.split("#")[2].strip()

from tracing import *

class PDFQnA:
    def __init__(self, model="llama3.2:3b", chunk_size: int = 4000, chunk_overlap: int = 100):
                # Initialize embedding model with consistent dimensions
        # Use all-MiniLM-L6-v2 (384 dims) to match existing collections
        self.llm = ChatOllama(model=model, temperature=0.3)
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

        self.graph_retrieval = GraphRetrieval()

        self.retrieval_cache = {}
        
        # Setup QA chain
        self.qa_chain = self._setup_qa_chain()
 
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


                # Set as root title if this is the first document
                if not hasattr(self, 'root_title') or not self.root_title:
                    self.root_title = title
                    print(f"Set root title to: {title}")
            

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

            #   # If this is the first document and we have a root title, ensure it's in the graph
            # if not hasattr(self, 'documents_added') and hasattr(self, 'root_title') and self.root_title:
            #     root_id = f"uploaded_{hash(self.root_title) & 0xffffffff}"
            #     if not self.graph_retrieval.G.has_node(root_id):
            #         self.graph_retrieval.G.add_node(
            #             root_id,
            #             label=self.root_title,
            #             title=self.root_title,
            #             type="root",  # Changed from "uploaded_paper" to "root" to match existing code
            #             is_root=True
            #         )
            #         print(f"Added root node for uploaded paper: {self.root_title}")

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
        variants: List[str] = [query]
        try:

            prompt = ChatPromptTemplate.from_template(
                "Generate up to 4 concise paraphrases or closely related query variants for: '{q}'.\n"
                "Return a JSON array of strings only."
            )
            chain = prompt | self.llm | StrOutputParser()
            raw = chain.invoke({"q": query})
            try:
                candidates = json.loads(raw)
                if not isinstance(candidates, list):
                    candidates = [str(raw).strip()]
            except Exception:
                candidates = [l.strip("- *\n ") for l in str(raw).splitlines() if l.strip()]
            base_emb = None
            try:
                base_emb = np.array(self.embedding_model.embed_query(query), dtype=float)
            except Exception:
                base_emb = None
            kept = []
            seen = {query.strip().lower()}
            for c in candidates:
                if not isinstance(c, str):
                    continue
                s = c.strip()
                if not s or s.lower() in seen or len(s) < 3:
                    continue
                if base_emb is not None:
                    try:
                        emb = np.array(self.embedding_model.embed_query(s), dtype=float)
                        sim = float(cosine_similarity([base_emb], [emb])[0][0])
                        if sim < 0.6 or sim > 0.98:
                            continue
                    except Exception:
                        pass
                kept.append(s)
                seen.add(s.lower())
                if len(kept) >= 4:
                    break
            variants.extend(kept)
        except Exception:
            print("DEBUG: Unable to expand query, returning, initial query.")
        print("DEBUG : variants", variants)
        return variants
    
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
        # Compute semantic similarity and apply additional signals
        try:
            q_emb = np.array(self.embedding_model.embed_query(query), dtype=float)
        except Exception:
            q_emb = None

        seen_titles = set()
        for r in results:
            base = float(r.get("relevance_score", 0.0))
            txt = str(r.get("content", ""))[:800]
            sim = 0.0
            if q_emb is not None and txt:
                try:
                    emb = np.array(self.embedding_model.embed_query(txt), dtype=float)
                    na = np.linalg.norm(q_emb); nb = np.linalg.norm(emb)
                    if na > 0 and nb > 0:
                        sim = float(np.dot(q_emb, emb) / (na * nb))
                except Exception:
                    sim = 0.0

            # Exact/partial match boost
            if query.lower() in txt.lower():
                base *= 1.15

            # Recency boost
            year = r.get("metadata", {}).get("publication_year")
            if isinstance(year, (int, float)) and year >= 2021:
                base *= 1.08

            # Source-aware smoothing
            src = r.get("source")
            if src == "vector_db":
                base = 0.8 * base + 0.2 * sim
            else:  # citation_graph
                base = 0.7 * base + 0.3 * sim

            # Light duplicate suppression by title text
            title_key = txt.split("\n", 1)[0].strip().lower()
            if title_key in seen_titles and title_key:
                base *= 0.95
            else:
                if title_key:
                    seen_titles.add(title_key)

            r["relevance_score"] = float(base)

        return sorted(results, key=lambda x: x.get("relevance_score", 0.0), reverse=True)


    def _setup_qa_chain(self, root_title="Attention is All You Need"):
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise research assistant. Answer strictly using the provided context. "
                "If the context is insufficient, say you don't know.\n\n"
                "Requirements:\n"
                "- Prefer concise bullet points (3–5).\n"
                "- Include inline citations using the form [Title, Year] when available.\n"
                "- Do not fabricate facts or citations.\n\n"
                "<context>\n{context}\n</context>"
            )),
            ("human", "{question}")
        ])
        
        chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
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
        graph_results = self.graph_retrieval.advanced_graph_retrieval(question, top_k=5, max_hops=3)
        query_embedding = self.embedding_model.embed_query(question)
        
        #  fusion with better weighting
        fused_results = self.fuse_and_rank_results(
            vector_results_formatted, graph_results, query_embedding, top_n=7
        )
        
        # Re-rank results
        fused_results = self.rerank_results(fused_results, question)
    
        # Prepare context for LLM with better formatting
        context_parts = []
        for i, res in enumerate(fused_results[:5]):
            source = res["source"]
            content = res["content"]
            metadata = res["metadata"]
            
            if source == "vector_db":
                title = metadata.get("title") or metadata.get("source") or "Document"
                year = metadata.get("publication_year")
                header = f"Title: {title}"
                if isinstance(year, (int, float)):
                    header += f" ({int(year)})"
                context_parts.append(f"{header}\nSnippet: {content}")
            else:
                title = res["content"].replace("Title: ", "")
                year = metadata.get("publication_year") if isinstance(metadata, dict) else None
                cited = metadata.get("cited_by_count") if isinstance(metadata, dict) else None
                suffix = []
                if isinstance(year, (int, float)):
                    suffix.append(str(int(year)))
                if isinstance(cited, (int, float)):
                    suffix.append(f"citations: {int(cited)}")
                extra = f" ({', '.join(suffix)})" if suffix else ""
                context_parts.append(f"Paper {i+1}: {title}{extra}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced citation path
        citation_path = []
        # Build query-personalized citation paths for explainability
        citation_path = self.graph_retrieval._query_personalized_paths(question, top_k=3)

        # Generate answer with better prompt
        response = self.qa_chain.invoke({
            "question": question, 
            "context": context
        })
        
        # Enhanced confidence calculation
        vector_scores = [res["relevance_score"] for res in fused_results if res["source"] == "vector_db"]
        graph_scores = [res["relevance_score"] for res in fused_results if res["source"] == "citation_graph"]
        
        avg_vector_score = np.mean(vector_scores) if vector_scores else 0.5
        avg_graph_score = np.mean(graph_scores) if graph_scores else 0.5
        
        # Weighted confidence with diversity bonus
        diversity_bonus = 0.1 if len(set([res["source"] for res in fused_results])) > 1 else 0
        confidence = 0.5 * avg_vector_score + 0.4 * avg_graph_score + 0.1 * diversity_bonus
        confidence = min(1.0, confidence)
        
        # Build retrieved_chunks ensuring at least 2 items are shown
        retrieved_chunks = [
            {"content": res["content"], "metadata": res["metadata"], "score": res["relevance_score"]}
            for res in fused_results if res["source"] == "vector_db"
        ]
        if len(retrieved_chunks) < 2:
            needed = 2 - len(retrieved_chunks)
            for res in [r for r in fused_results if r["source"] == "citation_graph"][:needed]:
                retrieved_chunks.append({
                    "content": res["content"],
                    "metadata": res["metadata"],
                    "score": res["relevance_score"]
                })

        # Cache the result
        result = {
            "answer": response,
            "explanation": {
                "retrieved_chunks": retrieved_chunks,
                "citation_path": citation_path,
                "confidence": round(confidence, 2),
                "retrieval_stats": {
                    "vector_results": len(vector_scores),
                    "graph_results": len(graph_scores),
                    "total_results": len(fused_results)
                }
            }
        }
        self.retrieval_cache[cache_key] = result
        return result


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

        question = input("Enter your question: ")
        # Ask a question
        answer = pdf_qa.ask_question(question)
        print(answer)

        # question = input("Enter your question: ")
        # answer = pdf_qa.ask_question(question)
        print("Answer:", answer["answer"])
        print("\nExplanation:")
        print("Retrieved Chunks:")
        for chunk in answer["explanation"]["retrieved_chunks"]:
            print(f"- {chunk['content']} (Metadata: {chunk['metadata']})")
        print("Citation Path:", " → ".join(answer["explanation"]["citation_path"]))
        print(f"Confidence: {answer['explanation']['confidence']}")
