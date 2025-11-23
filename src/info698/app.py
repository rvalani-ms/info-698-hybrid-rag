import streamlit as st
import os
from pdf_qna import PDFQuestionAnswering
from graph_builder import extract_id
import plotly.express as px
import traceback
from graph_builder import visualize_interactive, extract_id, analyze_graph_structure, find_influential_papers, find_temporal_trends

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
                st.session_state.pdf_qa = PDFQuestionAnswering(chunk_size=4000, chunk_overlap=300)
            
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
                
                # Show retrieval statistics if available
                if "retrieval_stats" in response["explanation"]:
                    stats = response["explanation"]["retrieval_stats"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Vector Results", stats.get("vector_results", 0))
                    with col2:
                        st.metric("Graph Results", stats.get("graph_results", 0))
                    with col3:
                        st.metric("Total Results", stats.get("total_results", 0))
                
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

# System Statistics (Enhanced)
if st.session_state.pdf_qa is not None:
    st.subheader("System Statistics")
    try:
        stats = st.session_state.pdf_qa.get_system_stats()
        
        # Graph statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Graph Nodes", stats['graph_stats']['nodes'])
        with col2:
            st.metric("Graph Edges", stats['graph_stats']['edges'])
        with col3:
            st.metric("Graph Density", f"{stats['graph_stats']['density']:.3f}")
        with col4:
            st.metric("Communities", stats['graph_stats']['communities'])
        
        # Vector database statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vector Collection Size", stats['vector_stats']['collection_size'])
        with col2:
            st.metric("Processed PDFs", stats['vector_stats']['processed_pdfs'])
        with col3:
            st.metric("Cache Hit Rate", f"{stats['cache_stats']['cache_hit_rate']:.3f}")
            
    except Exception as e:
        st.warning(f"Could not load system statistics: {e}")
        st.info("💡 This is usually due to missing data or initialization issues. The system should still work for basic operations.")

# Graph visualization
st.subheader("Citation Graph")
if st.session_state.citation_graph is not None and st.session_state.root_id is not None:
    try:
        # Check if graph has nodes
        if st.session_state.citation_graph.number_of_nodes() > 0:
            # Generate interactive graph
            fig = visualize_interactive(
                st.session_state.citation_graph, 
                st.session_state.graph_edges, 
                st.session_state.root_id
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Citation graph is empty (0 nodes).")
    except Exception as e:
        st.error(f"Error rendering citation graph: {str(e)}")
        st.info("💡 Try running `python3 data_collector.py` to generate citation data.")
else:
    st.info("📊 **Citation graph not available**")
    st.markdown("""
    **To enable citation graph visualization:**
    1. Run the data collector: `python3 data_collector.py`
    2. This will fetch citation data from OpenAlex API
    3. The data will be saved to `citations.json`
    4. Restart the app to load the graph
    """)

# Graph Analysis (Structure, Influential Papers, Temporal Trends)
if st.session_state.pdf_qa is not None and st.session_state.citation_graph is not None:
    with st.expander("📊 Graph Analysis", expanded=False):
        try:
            GA = analyze_graph_structure(st.session_state.pdf_qa.G)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes", GA.get('nodes', 0))
            with col2:
                st.metric("Edges", GA.get('edges', 0))
            with col3:
                st.metric("Density", f"{GA.get('density', 0):.3f}")
            with col4:
                st.metric("Modularity", f"{GA.get('modularity', 0):.3f}")

            if GA.get('year_range'):
                yr = GA['year_range']
                avg_year = GA.get('avg_year', 0)
                try:
                    avg_year_disp = int(round(avg_year))
                except Exception:
                    avg_year_disp = avg_year
                st.markdown(f"**Year Range**: {yr[0]} - {yr[1]} | **Avg Year**: {avg_year_disp}")
                st.markdown(f"**Avg Citations**: {GA.get('avg_citations', 0):.1f} | **Max Citations**: {GA.get('max_citations', 0)}")

            st.markdown("---")
            st.markdown("**🏆 Influential Papers (Top 5)**")
            infl = find_influential_papers(st.session_state.pdf_qa.G, top_k=5)
            for i, p in enumerate(infl, 1):
                st.markdown(f"{i}. {p.get('title','N/A')} (Year: {p.get('publication_year','N/A')}, Citations: {p.get('cited_by_count',0)})")

            st.markdown("---")
            st.markdown("**📅 Temporal Trends**")
            tt = find_temporal_trends(st.session_state.pdf_qa.G)
            if tt.get('years'):
                years = tt['years']
                st.markdown(f"Years covered: {min(years)} - {max(years)}")
                papers_per_year = tt.get('papers_per_year', [])
                try:
                    fig_ppy = px.bar(x=years, y=papers_per_year, labels={'x': 'Year', 'y': 'Papers'}, title='Papers per Year')
                    st.plotly_chart(fig_ppy, use_container_width=True)
                except Exception:
                    st.markdown(f"Papers per year: {papers_per_year}")
            else:
                st.markdown("No temporal data available.")
        except Exception as e:
            st.warning(f"Graph analysis unavailable: {e}")

# Instructions
st.sidebar.markdown("---")
st.sidebar.subheader("📖 Instructions")
st.sidebar.markdown("""
1. *Upload Papers*: Add one or more PDF papers using the file uploader above
2. *Ask Questions*: Enter research questions (e.g., "What is attention mechanism?")
3. *Get Enhanced Answers*: Receive answers with citation paths and confidence scores
4. *View Analytics*: Check system statistics and citation graph visualization
5. *Explore Features*: Try different types of questions to see the system's capabilities
""")    

# Cleanup option
if st.sidebar.button("Clear Papers and Reset"):
    try:
        # shutil.rmtree(PAPERS_DIR)
        # os.makedirs(PAPERS_DIR)
        st.session_state.pdf_qa = None
        st.session_state.papers_loaded = False
        st.session_state.citation_graph = None
        st.session_state.graph_edges = None
        st.session_state.root_id = None
        st.session_state.processed_pdfs = set()
        st.sidebar.success("Papers cleared and system reset.")
    except Exception as e:
        st.sidebar.error(f"Error resetting: {str(e)}")