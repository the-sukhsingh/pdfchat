import streamlit as st
import pandas as pd
import numpy as np
import fitz # PyMuPDF
import torch
import re
import os
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
from tqdm.auto import tqdm
from time import perf_counter as timer
import textwrap
import tempfile
import pickle

# Page configuration
st.set_page_config(
    page_title="Local RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Utility functions
@st.cache_data
def text_formatter(text: str) -> str:
    """Clean and format text by removing newlines and extra spaces."""
    cleaned_text = text.replace("\n", "").strip()
    return cleaned_text

@st.cache_data
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF and return structured data."""
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for page_number, page in enumerate(doc):
        status_text.text(f"Processing page {page_number + 1}/{len(doc)}")
        progress_bar.progress((page_number + 1) / len(doc))
        
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_texts.append({
            'page_number': page_number,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text
        })
    
    progress_bar.empty()
    status_text.empty()
    return pages_and_texts

@st.cache_resource
def load_spacy_model():
    """Load spaCy model for sentence segmentation."""
    nlp = English()
    nlp.add_pipe("sentencizer")
    return nlp

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        model_name_or_path="all-mpnet-base-v2",
        device=device
    )
    return model

def split_list(input_list: list, slice_size: int = 10) -> list[list[str]]:
    """Split a list into chunks of specified size."""
    return [input_list[i: i + slice_size + 1] for i in range(0, len(input_list), slice_size)]

def process_pdf_to_chunks(pdf_path: str, chunk_size: int = 10):
    """Process PDF into text chunks with embeddings."""
    
    st.info("📄 Extracting text from PDF...")
    pages_and_texts = open_and_read_pdf(pdf_path)
    
    st.info("✂️ Segmenting sentences...")
    nlp = load_spacy_model()
    
    progress_bar = st.progress(0)
    for i, item in enumerate(pages_and_texts):
        progress_bar.progress((i + 1) / len(pages_and_texts))
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count"] = len(item["sentences"])
    progress_bar.empty()
    
    st.info("🔗 Creating text chunks...")
    for item in pages_and_texts:
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    
    # Create individual chunks
    pages_and_chunks = []
    for item in pages_and_texts:
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ","").strip()
            joined_sentence_chunk = re.sub(r'\.(A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
            pages_and_chunks.append(chunk_dict)
    
    st.info("🧠 Generating embeddings...")
    embedding_model = load_embedding_model()
    
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process embeddings in batches
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        status_text.text(f"Embedding batch {i//batch_size + 1}/{(len(text_chunks)//batch_size) + 1}")
        progress_bar.progress(min(1.0, (i + batch_size) / len(text_chunks)))
        
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True)
        all_embeddings.append(batch_embeddings)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # Add embeddings to chunks
    for i, item in enumerate(pages_and_chunks):
        item["embedding"] = embeddings[i].cpu().numpy()
    
    progress_bar.empty()
    status_text.empty()
    
    return pages_and_chunks, embeddings, embedding_model

def retrieve_relevant_resources(query: str, embeddings: torch.tensor, model: SentenceTransformer, n_resources_to_return: int = 5):
    """Retrieve most relevant text chunks for a query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()
    
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices, end_time - start_time

def print_wrapped(text, width=80):
    """Print text wrapped to specified width."""
    return textwrap.fill(text, width=width)

# Main application
st.title("📚 Local RAG PDF Chatbot")
st.markdown("Upload a PDF document and chat with it using local embeddings and semantic search!")

# Sidebar
st.sidebar.header("⚙️ Settings")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    st.success("✅ File uploaded successfully!")
    
    # Processing options
    st.sidebar.subheader("📊 Processing Options")
    chunk_size = st.sidebar.slider("Sentence chunk size", min_value=5, max_value=20, value=10, 
                                   help="Number of sentences per chunk")
    
    if st.sidebar.button("🚀 Process PDF", type="primary"):
        try:
            with st.spinner("Processing PDF... This may take a few minutes."):
                pages_and_chunks, embeddings, embedding_model = process_pdf_to_chunks(tmp_file_path, chunk_size)
                
                # Store in session state
                st.session_state.processed_data = pd.DataFrame(pages_and_chunks)
                st.session_state.embeddings = embeddings
                st.session_state.embedding_model = embedding_model
                
            st.success("🎉 PDF processed successfully!")
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chunks", len(pages_and_chunks))
            with col2:
                st.metric("Avg. Chunk Length", f"{st.session_state.processed_data['chunk_char_count'].mean():.0f} chars")
            with col3:
                st.metric("Total Pages", st.session_state.processed_data['page_number'].nunique())
            with col4:
                st.metric("Embedding Dimension", embeddings.shape[1])
                
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    # Chat interface
    if st.session_state.processed_data is not None:
        st.divider()
        st.subheader("💬 Chat with your PDF")
        
        # Search parameters
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Ask a question about your PDF:", placeholder="e.g., What is Python dictionary?")
        with col2:
            n_results = st.selectbox("Results to show:", [3, 5, 10], index=1)
        
        if query:
            try:
                scores, indices, search_time = retrieve_relevant_resources(
                    query=query, 
                    embeddings=st.session_state.embeddings, 
                    model=st.session_state.embedding_model, 
                    n_resources_to_return=n_results
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": query,
                    "results": [(scores[i].item(), indices[i].item()) for i in range(len(scores))],
                    "search_time": search_time
                })
                
                st.info(f"🔍 Search completed in {search_time:.4f} seconds")
                
                # Display results
                st.subheader("📋 Search Results")
                
                for i, (score, idx) in enumerate(zip(scores, indices)):
                    with st.expander(f"Result {i+1} (Score: {score.item():.4f})", expanded=i<2):
                        chunk_data = st.session_state.processed_data.iloc[int(idx)]
                        st.write("**Text:**")
                        st.write(chunk_data['sentence_chunk'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Page", int(chunk_data['page_number']) + 1)
                        with col2:
                            st.metric("Word Count", int(chunk_data['chunk_word_count']))
                        with col3:
                            st.metric("Similarity Score", f"{score.item():.4f}")
                
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
        
        # Chat history
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📝 Chat History")
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                    st.write(f"**Query:** {chat['query']}")
                    st.write(f"**Search Time:** {chat['search_time']:.4f} seconds")
                    st.write(f"**Top Result Score:** {chat['results'][0][0]:.4f}")
        
        # Data exploration
        st.divider()
        st.subheader("📊 Data Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📈 Statistics", "📋 Sample Data", "🔧 Export"])
        
        with tab1:
            st.write("**Chunk Statistics:**")
            st.dataframe(st.session_state.processed_data[['chunk_char_count', 'chunk_word_count', 'chunk_token_count', 'page_number']].describe())
            
            # Visualization
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.hist(st.session_state.processed_data['chunk_word_count'], bins=20, alpha=0.7)
            ax1.set_title('Distribution of Chunk Word Counts')
            ax1.set_xlabel('Word Count')
            ax1.set_ylabel('Frequency')
            
            ax2.scatter(st.session_state.processed_data['page_number'], st.session_state.processed_data['chunk_word_count'], alpha=0.6)
            ax2.set_title('Word Count vs Page Number')
            ax2.set_xlabel('Page Number')
            ax2.set_ylabel('Word Count')
            
            st.pyplot(fig)
        
        with tab2:
            st.write("**Sample chunks:**")
            sample_size = min(10, len(st.session_state.processed_data))
            sample_data = st.session_state.processed_data.sample(n=sample_size)[['page_number', 'sentence_chunk', 'chunk_word_count']]
            
            for idx, row in sample_data.iterrows():
                with st.expander(f"Page {int(row['page_number']) + 1} - {row['chunk_word_count']} words"):
                    st.write(row['sentence_chunk'])
        
        with tab3:
            st.write("**Export processed data:**")
            
            if st.button("💾 Download CSV"):
                csv = st.session_state.processed_data.drop(columns=['embedding']).to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="pdf_chunks.csv",
                    mime="text/csv"
                )
            
            st.info("💡 **Tip:** You can save the embeddings separately for faster future loading!")

else:
    st.warning("📤 Please upload a PDF file to start chatting.")
    
    # Instructions
    st.markdown("""
    ### 🔍 How it works:
    
    1. **Upload PDF**: Choose any PDF document
    2. **Process**: The app will:
       - Extract text from all pages
       - Split text into sentence chunks
       - Generate embeddings using sentence-transformers
    3. **Chat**: Ask questions about the content
    4. **Get Results**: Receive relevant text chunks ranked by similarity
    
    ### ✨ Features:
    - Local processing (no external APIs required)
    - Semantic search using sentence embeddings
    - Adjustable chunk sizes
    - Chat history
    - Data visualization and export
    - Fast similarity search with dot product scoring
    """)

# Footer
st.divider()
st.markdown("Built with ❤️ using Streamlit, sentence-transformers, and fitz")
