# Local RAG PDF Chatbot 📚

A fully-featured Streamlit application that implements a local Retrieval-Augmented Generation (RAG) pipeline for PDF documents. Chat with your PDFs using semantic search and local embeddings - no external APIs required!

## 🌟 Features

- **Local Processing**: Complete PDF processing and search without external dependencies
- **Semantic Search**: Uses sentence-transformers for high-quality embeddings
- **Interactive Chat**: Natural language querying of PDF content
- **Real-time Results**: Fast similarity search with relevance scoring
- **Data Visualization**: Statistics and charts about your document chunks
- **Export Capabilities**: Download processed data as CSV
- **Chat History**: Keep track of your queries and results
- **Responsive UI**: Modern, user-friendly Streamlit interface

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Sentence Transformers**: For generating text embeddings
- **PyTorch**: Deep learning backend
- **PyMuPDF (fitz)**: PDF text extraction
- **spaCy**: Natural language processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization

## 📋 Requirements

- Python 3.7+
- Windows/Linux/macOS
- At least 2GB RAM (more recommended for large PDFs)
- Optional: CUDA-capable GPU for faster processing

## 🚀 Quick Start

### Option 1: Automatic Setup

1. **Clone or download** this repository
2. **Navigate** to the project directory
3. **Run the setup script**:
   ```bash
   python setup.py
   ```
4. **Start the application**:
   ```bash
   streamlit run app.py
   ```

### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📖 How to Use

### 1. Upload PDF
- Click "Browse files" or drag and drop your PDF
- Supported format: PDF files of any size

### 2. Configure Processing
- **Sentence chunk size**: Adjust how many sentences per chunk (5-20)
- Higher values = longer chunks, fewer total chunks
- Lower values = shorter chunks, more granular search

### 3. Process Document
- Click "🚀 Process PDF" to start
- Processing includes:
  - Text extraction from all pages
  - Sentence segmentation
  - Chunk creation with overlap
  - Embedding generation
- Progress bars show real-time status

### 4. Chat with Your PDF
- Enter questions in natural language
- Examples:
  - "What is a Python dictionary?"
  - "How do I handle exceptions?"
  - "Explain the main concepts"
- Adjust number of results (3, 5, or 10)

### 5. Explore Results
- **Search Results**: Ranked by similarity score
- **Page References**: Know exactly where information came from
- **Statistics**: Character, word, and token counts
- **Chat History**: Review previous queries

### 6. Analyze Data
- **Statistics Tab**: Numerical summaries and distributions
- **Sample Data Tab**: Browse random text chunks
- **Export Tab**: Download processed data as CSV

## 🧠 How It Works

### Text Processing Pipeline
1. **PDF Extraction**: PyMuPDF extracts raw text from each page
2. **Text Cleaning**: Remove extra whitespace and formatting
3. **Sentence Segmentation**: spaCy breaks text into sentences
4. **Chunking**: Group sentences into overlapping chunks
5. **Embedding**: sentence-transformers creates vector representations

### Search Algorithm
1. **Query Encoding**: Convert user question to embedding vector
2. **Similarity Calculation**: Compute dot product with all chunk embeddings
3. **Ranking**: Sort chunks by similarity score
4. **Results**: Return top-k most relevant chunks

### Performance Optimizations
- **Caching**: Streamlit caches models and processed data
- **Batch Processing**: Embeddings generated in efficient batches
- **GPU Support**: Automatically uses CUDA if available
- **Memory Management**: Efficient tensor operations with PyTorch

## 📊 Technical Specifications

### Embedding Model
- **Model**: `all-mpnet-base-v2`
- **Dimensions**: 768
- **Language**: English (extensible to multilingual)
- **Performance**: High quality sentence embeddings

### Chunking Strategy
- **Method**: Sentence-based with configurable overlap
- **Size**: 5-20 sentences per chunk (adjustable)
- **Overlap**: 1 sentence between consecutive chunks
- **Rationale**: Maintains context while enabling granular search

### Search Method
- **Algorithm**: Dot product similarity
- **Speed**: O(n) where n = number of chunks
- **Accuracy**: High precision for semantic similarity
- **Scalability**: Handles documents with thousands of chunks

## 🔧 Configuration Options

### Processing Parameters
- `chunk_size`: Number of sentences per chunk (default: 10)
- `batch_size`: Embedding batch size (default: 32)
- `device`: Processing device (auto-detected: CPU/CUDA)

### Search Parameters
- `n_resources_to_return`: Number of results to show (3, 5, 10)
- `similarity_threshold`: Minimum similarity score (configurable)

## 📈 Performance Tips

### For Large Documents
- Use larger chunk sizes (15-20 sentences)
- Enable GPU processing if available
- Consider processing in sections for very large PDFs

### For Better Search Results
- Use specific, detailed queries
- Try different phrasings of the same question
- Experiment with different chunk sizes

### Memory Optimization
- Close other applications when processing large PDFs
- Monitor memory usage during embedding generation
- Consider upgrading RAM for very large documents

## 🐛 Troubleshooting

### Common Issues

**"Import could not be resolved" errors:**
- Run `pip install -r requirements.txt`
- Ensure you're using Python 3.7+

**Slow processing:**
- Check if CUDA is properly installed for GPU acceleration
- Reduce batch size if running out of memory
- Use smaller chunk sizes for faster processing

**PDF extraction fails:**
- Ensure PDF is not password-protected
- Try a different PDF file
- Check if PDF contains extractable text (not just images)

**Out of memory errors:**
- Reduce batch size
- Use CPU instead of GPU
- Process smaller sections of large documents

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── setup.py              # Setup and installation script
├── README.md             # This documentation
├── Python.pdf            # Sample PDF for testing
└── 01.ipynb             # Original Jupyter notebook (reference)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Support for additional file formats (DOCX, TXT, etc.)
- Multiple language support
- Advanced search filters
- Integration with external LLMs
- Batch processing multiple documents
- Vector database integration

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Sentence Transformers**: For excellent embedding models
- **Streamlit**: For the amazing web app framework
- **PyMuPDF**: For robust PDF processing
- **spaCy**: For natural language processing tools

## 📞 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are properly installed
4. Try with a different PDF file to isolate the issue

---

**Happy chatting with your PDFs! 🎉**
