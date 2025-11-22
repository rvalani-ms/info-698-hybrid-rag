import streamlit as st
import os
from pdf_qna import PDFQuestionAnswering
from graph_builder import extract_id
import plotly.express as px
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

