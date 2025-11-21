import streamlit as st
import os
from pdf_qna import PDFQuestionAnswering
from graph_builder import extract_id
import plotly
import plotly.express as px
import json
import traceback

# Streamlit page configuration
st.set_page_config(page_title="Hybrid RAG Citation Graph Assistant", layout="wide")

# Initialize session state
if 'pdf_qa' not in st.session_state:
    st.session_state.pdf_qa = None
if 'papers_loaded' not in st.session_state:
    st.session_state.papers_loaded = False
if 'citation_graph' not in st.session_state:
    st.session_state.citation_graph = None
if 'graph_edges' not in st.session_state:
    st.session_state.graph_edges = None
if 'root_id' not in st.session_state:
    st.session_state.root_id = None
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = set()

# Create papers directory if it doesn't exist
PAPERS_DIR = "./../papers"
if not os.path.exists(PAPERS_DIR):
    os.makedirs(PAPERS_DIR)

# Sidebar for PDF upload and controls
st.sidebar.title("Controls")
#use_best_first = st.sidebar.checkbox("Use best-first graph traversal", value=False)
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Papers", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    with st.sidebar.status("Processing uploaded PDFs..."):
        try:
            # Initialize PDFQuestionAnswering if not already done
            if st.session_state.pdf_qa is None:
                st.session_state.pdf_qa = PDFQuestionAnswering()
            
            # Process each uploaded PDF only if not already processed
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                if file_name not in st.session_state.processed_pdfs:
                    file_path = os.path.join(PAPERS_DIR, file_name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.write(f"Processing {file_name}...")
                    documents = st.session_state.pdf_qa.load_document(file_path)
                    if documents:
                        st.session_state.pdf_qa.add_documents(documents)
                        st.session_state.processed_pdfs.add(file_name)
                        # Also add to the RAG system's tracking
                        if hasattr(st.session_state.pdf_qa, 'processed_pdfs'):
                            st.session_state.pdf_qa.processed_pdfs.add(file_path)
                        st.write(f"✅ Loaded {file_name}")
                    else:
                        st.write(f"⚠️  No content extracted from {file_name}")
                else:
                    st.write(f"Skipping {file_name} (already processed)")
            
            st.session_state.papers_loaded = True
            st.success("All new PDFs processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            st.session_state.papers_loaded = False

# Main content
st.title("Hybrid RAG Citation Graph Assistant")
st.markdown("Upload academic papers and ask questions to get answers with citation context and explanations.")

# Display loaded PDFs
if st.session_state.processed_pdfs:
    st.markdown("**Loaded Papers**:")
    for pdf in st.session_state.processed_pdfs:
        st.markdown(f"- {pdf}")

# Query input
query = st.text_input("Enter your question:", placeholder="e.g., How is attention different from previous sequence models?")
submit_button = st.button("Submit Query")

if submit_button and query:
    if st.session_state.pdf_qa is None:
        st.error("Please upload at least one PDF to initialize the system.")
    elif not st.session_state.papers_loaded:
        st.error("No papers loaded. Please upload PDFs first.")
    else:
        with st.status("Processing query..."):
            try:
                # Query the system without reprocessing PDFs
                response = st.session_state.pdf_qa.ask_question(query)
                
                # Display answer
                st.subheader("Answer")
                st.markdown(response["answer"])
                
                # Display explanation with enhanced information
                st.subheader("Explanation")
                
                #TODO: Add retrieval stats
                
                # Display retrieved chunks with enhanced metadata
                for i, chunk in enumerate(response["explanation"]["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i}**:")
                    st.markdown(f"**Content**: {chunk['content']}")
                    st.markdown(f"**Metadata**: {chunk['metadata']}")
                    if 'score' in chunk:
                        st.markdown(f"**Relevance Score**: {chunk['score']:.3f}")
                    st.markdown("---")
            
                # Enhanced citation path display
                if response["explanation"]["citation_path"]:
                    st.markdown("**Citation Path**: " + " → ".join(response["explanation"]["citation_path"]))
                else:
                    st.markdown("**Citation Path**: No citations found")
                
                # Enhanced confidence display
                confidence = response['explanation']['confidence']
                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                st.markdown(f"**Confidence**: :{confidence_color}[{confidence:.2f}]")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.markdown("**Stack Trace** (for debugging):")
                st.code(traceback.format_exc())

    if st.session_state.pdf_qa is None or not st.session_state.papers_loaded:
        st.error("Load PDFs first to run evaluation.")
    else:
        with st.status("Running evaluation..."):
            try:
                evaluator = HybridRAGEvaluator(st.session_state.pdf_qa)
                test_queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
                if not test_queries:
                    test_queries = default_queries
                results = evaluator.run_comprehensive_evaluation(test_queries)
                st.subheader("Evaluation Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{results['overall_score']:.3f}")
                    st.metric("Avg Confidence", f"{results['retrieval_performance']['avg_confidence']:.3f}")
                with col2:
                    st.metric("Avg Retrieval Time", f"{results['retrieval_performance']['avg_retrieval_time']:.3f}s")
                    st.metric("Avg Diversity", f"{results['retrieval_performance']['avg_diversity']:.3f}")
                with col3:
                    st.metric("Fusion Improvement", f"{results['fusion_effectiveness']['avg_improvement']:.3f}")
                    st.metric("Avg Vector Score", f"{results['fusion_effectiveness']['avg_vector_score']:.3f}")
                st.markdown("---")
                st.markdown("**Graph Statistics**")
                gs = results['graph_quality']['graph_analysis']
                gcol1, gcol2, gcol3, gcol4 = st.columns(4)
                with gcol1:
                    st.metric("Nodes", gs.get('nodes', 0))
                with gcol2:
                    st.metric("Edges", gs.get('edges', 0))
                with gcol3:
                    st.metric("Density", f"{gs.get('density', 0):.3f}")
                with gcol4:
                    st.metric("Modularity", f"{gs.get('modularity', 0):.3f}")
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")