# A Hybrid RAG System With Graph Based Contect Retrieval

## Overview

The Graph RAG Powered Academic Assistant is an AI system designed to help students and researchers navigate scholarly literature with explainable, citation-aware, and semantically grounded insights. It enhances traditional RAG systems by integrating semantic vector retrieval with citation-graph traversal, ensuring answers are relevant and academically verifiable.

## Key Capabilities

-   Hybrid Retrieval: Semantic + Citation Graph
-   Explainability Module with citation trails
-   PDF Parsing & Chunking
-   Interactive Web UI
-   Local, privacy-preserving architecture

## System Architecture

-   Document Ingestion & Parsing
-   Hybrid Retrieval Pipeline
-   LLM Reasoning
-   Explainability Layer
-   Streamlit UI

## Features

-   PDF Parsing & Chunking
-   Citation Graph Builder
-   Dual Retrieval (Semantic + Graph)
-   Explainability & Tracing
-   Web UI

## Methodology Summary

-   Upload PDF
-   Chunking
-   Embedding
-   Citation Graph Construction
-   Dual Retrieval
-   Fusion Scoring
-   LLM Reasoning
-   Explainability Output

## Evaluation Summary

-   +10–12% BLEU/ROUGE-L improvement
-   ~6% hallucination rate
-   87% citation accuracy

## Tech Stack

Python, ChromaDB, NetworkX, SentenceTransformers, Docling, LangChain, Ollama, Phoenix, Streamlit

## Installation & Setup

-   Clone Repository
    ```
    git clone <repo-url>
    cd project
    ```
-   Install Dependencies
    ```
    pip install -r requirements.txt
    ```
-   Start the Streamlit UI
    ```
    streamlit run app.py
    ```

## Future Work

-   AWS deployment
-   Multi-agent architecture
-   Explore different LLM models to improve retrieval accuracy, reasoning quality, and citation grounding

## Team

Pant Divyansh
Singh Gagan Preet
Tiwari Sourav
Valani Rameshkumar Premji

Advisor: Dr. Xiao Hu
