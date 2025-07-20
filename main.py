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
from google import genai
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# Page configuration
st.set_page_config(
    page_title="Local RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat bubbles and modal
st.markdown("""
<style>
.user-message {
    background-color: #E3F2FD;
    color: black;
    padding: 15px;
    margin: 10px 0;
    border-radius: 15px 15px 5px 15px;
    margin-left: 20%;
    border-left: 4px solid #2196F3;
}

.bot-message {
    background-color: #F1F8E9;
    padding: 15px;
    color: black;
    margin: 10px 0;
    border-radius: 15px 15px 15px 5px;
    margin-right: 20%;
    border-left: 4px solid #4CAF50;
}

.chunk-links {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #ddd;
    font-size: 0.85em;
    color: #666;
}

.chunk-links small {
    color: #555;
}

.chunk-link {
    display: inline-block;
    background-color: #E8F5E8;
    color: black;
    padding: 4px 8px;
    margin: 2px 4px;
    border-radius: 12px;
    text-decoration: none;
    border: 1px solid #4CAF50;
    font-size: 0.8em;
    cursor: pointer;
}

.chunk-link:hover {
    background-color: #C8E6C9;
    text-decoration: none;
}

.stTextInput > div > div > input {
    border-radius: 25px;
    border: 2px solid #E0E0E0;
    padding: 10px 20px;
}

.stButton > button {
    border-radius: 25px;
    height: 3rem;
}

.chunk-modal {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    color: black;
    padding: 8px;
    margin: 10px 0;
}

.chunk-header {
    background-color: #e9ecef;
    display: flex;
    align-items: center;
    justify-content: space-around;
    border-radius: 6px 6px 0 0;
    padding-top: 10px;
}

.chunk-content {
    background-color: white;
    padding: 15px;
    color: black;
    border-radius: 0 0 6px 6px;
    border-left: 4px solid #007bff;
    # margin: 10px 0;
}

.good {
    background-color: green;
    color: white;
    padding: 5px 8px;
    border-radius: 12px;
    
}

.moderate {
    background-color: yellow;
    color: black;
    padding: 5px 8px;
    border-radius: 12px;
    
}

.bad {
    background-color: red;
    color: white;
    padding: 5px 8px;
    border-radius: 12px;
    
}

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_chunks_modal' not in st.session_state:
    st.session_state.show_chunks_modal = {}
if 'selected_chunks' not in st.session_state:
    st.session_state.selected_chunks = {}

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
            joined_sentence_chunk = "".join(sentence_chunk).strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
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
    
    # Get the top-k chunks from the retrieved indices
    top_chunks = [st.session_state.processed_data.iloc[int(idx)]['sentence_chunk'] for idx in indices]
    
    content = f'''
    You are a helpful assistant. Your task is to answer questions based on the provided text chunks.
    Here are the most relevant text chunks based on the query: "{query}".
    Please provide concise and accurate answers based on the content of these chunks.
    The text chunks are as follows:
    {"\n".join([f"{i+1}. {text}" for i, text in enumerate(top_chunks)])}
    
    Query: {query}
    '''
    
    # Send the relevent resources to the client and return the model response
    response = client.models.generate_content(
    model="gemini-2.5-flash", contents=content)
    
    print(response.text)
    
    return scores, indices, end_time - start_time, response.text

def print_wrapped(text, width=80):
    """Print text wrapped to specified width."""
    return textwrap.fill(text, width=width)

# Main application
st.title("📚 Local RAG PDF Chatbot")
st.markdown("Upload a PDF document and chat with it using local embeddings and semantic search!")

# Sidebar
st.sidebar.header("⚙️ Settings")

# File upload in sidebar
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    st.sidebar.success("✅ File uploaded successfully!")
    
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
        st.subheader("💬 Chat with your PDF")
        
        # Chat settings in sidebar
        st.sidebar.subheader("🔧 Chat Settings")
        n_results = st.sidebar.selectbox("Results to show:", [3, 5, 10], index=1)
        
        # Chat history display
        if st.session_state.chat_history:
            st.markdown("### 📝 Conversation")
            
            # Display chat messages
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {chat['query']}
                </div>
                """, unsafe_allow_html=True)
                
                # Bot response with chunk links
                chunk_links_html = ""
                if 'results' in chat and chat['results']:
                    chunk_links = []
                    for j, (score, idx) in enumerate(chat['results']):
                        chunk_data = st.session_state.processed_data.iloc[int(idx)]
                        page_num = int(chunk_data['page_number']) + 1
                        chunk_links.append(f"📄 Page {page_num} ({score:.3f})")
                    
                
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Assistant:</strong> {chat['response']}
                </div>
                """, unsafe_allow_html=True)
                
                # Add buttons to view chunks
                if 'results' in chat and chat['results']:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(f"📋 View Source Chunks", key=f"chunks_btn_{i}", use_container_width=True):
                            st.session_state.show_chunks_modal[i] = True
                            st.session_state.selected_chunks[i] = chat['results']
                            st.rerun()
                
                # Display modal if requested
                if st.session_state.show_chunks_modal.get(i, False):
                    st.markdown("---")
                    st.markdown(f"### 📋 Source Chunks for Query {i+1}")
                    
                    try:
                        for j, (score, idx) in enumerate(st.session_state.selected_chunks[i]):
                            chunk_data = st.session_state.processed_data.iloc[int(idx)]
                            
                            st.markdown(f"""
                            <div class="chunk-modal">
                                <div class="chunk-header">
                                    <p>📄 Chunk {j+1} - Page {int(chunk_data['page_number']) + 1}</p>
                                    <p class={
                                        "good" if score > 0.7 else "moderate" if score > 0.4 else "bad"
                                        }><strong>Similarity Score:</strong> {score * 100:.2f}</p>
                                    <p><strong>Word Count:</strong> {int(chunk_data['chunk_word_count'])}</p>
                                </div>
                                <div class="chunk-content">
                                {chunk_data['sentence_chunk']}
                            </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error displaying chunks: {str(e)}")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button(f"❌ Close", key=f"close_modal_{i}", use_container_width=True):
                            st.session_state.show_chunks_modal[i] = False
                            st.rerun()
                    
                    st.markdown("---")
        
        # Divider before input
        st.divider()
        
        # Input area at bottom
        st.markdown("### 💭 Ask a question")
        
        # Create input form
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                query = st.text_input("", placeholder="Ask a question about your PDF...", label_visibility="collapsed")
            
            with col2:
                submit_button = st.form_submit_button("Send 📤", use_container_width=True)
        
        # Clear chat button in sidebar
        if st.session_state.chat_history:
            if st.sidebar.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.show_chunks_modal = {}
                st.session_state.selected_chunks = {}
                st.rerun()
        
        # Data exploration in sidebar
        with st.sidebar.expander("📊 Data Analysis", expanded=False):
            if st.button("📈 View Statistics"):
                st.write("**Chunk Statistics:**")
                stats_df = st.session_state.processed_data[['chunk_char_count', 'chunk_word_count', 'chunk_token_count', 'page_number']].describe()
                st.dataframe(stats_df, use_container_width=True)
            
            if st.button("💾 Export Data"):
                csv = st.session_state.processed_data.drop(columns=['embedding']).to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="pdf_chunks.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Process query
        if submit_button and query:
            try:
                with st.spinner("🔍 Searching for relevant information..."):
                    scores, indices, search_time, model_response = retrieve_relevant_resources(
                        query=query, 
                        embeddings=st.session_state.embeddings, 
                        model=st.session_state.embedding_model, 
                        n_resources_to_return=n_results
                    )
                
                # Add to chat history
                chat_index = len(st.session_state.chat_history)
                st.session_state.chat_history.append({
                    "query": query,
                    "response": model_response,
                    "results": [(scores[i].item(), indices[i].item()) for i in range(len(scores))],
                    "search_time": search_time
                })
                
                # Initialize modal state for new chat
                st.session_state.show_chunks_modal[chat_index] = False
                
                # Rerun to show updated chat
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
    
    # Show initial state if no processed data
    else:
        st.info("📤 Please upload and process a PDF file in the sidebar to start chatting.")
        
        # Instructions
        st.markdown("""
        ### 🔍 How it works:
        
        1. **Upload PDF**: Choose any PDF document in the sidebar
        2. **Process**: The app will:
           - Extract text from all pages
           - Split text into sentence chunks
           - Generate embeddings using sentence-transformers
        3. **Chat**: Ask questions about the content
        4. **Get Results**: Receive relevant answers based on document content
        
        ### ✨ Features:
        - Local processing (no external APIs required)
        - Semantic search using sentence embeddings
        - Adjustable chunk sizes
        - Chat history with message bubbles
        - Data visualization and export
        - Fast similarity search with dot product scoring
        """)

else:
    st.info("📤 Please upload a PDF file in the sidebar to start chatting.")
    
    # Instructions
    st.markdown("""
    ### 🔍 How it works:
    
    1. **Upload PDF**: Choose any PDF document in the sidebar
    2. **Process**: The app will:
       - Extract text from all pages
       - Split text into sentence chunks
       - Generate embeddings using sentence-transformers
    3. **Chat**: Ask questions about the content
    4. **Get Results**: Receive relevant answers based on document content
    
    ### ✨ Features:
    - Local processing (no external APIs required)
    - Semantic search using sentence embeddings
    - Adjustable chunk sizes
    - Chat history with message bubbles
    - Data visualization and export
    - Fast similarity search with dot product scoring
    """)

# Footer
st.divider()
st.markdown("Built with ❤️ using Streamlit, sentence-transformers, and fitz")
