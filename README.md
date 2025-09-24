# 🐧 Antarctic Penguin Research Intelligence

An AI-powered RAG (Retrieval Augmented Generation) system specialized in Antarctic penguin research, remote sensing, and wildlife monitoring using satellite and UAV data.

![Antarctic Penguin Research](https://img.shields.io/badge/Antarctic-Research-blue)
![Remote Sensing](https://img.shields.io/badge/Remote-Sensing-green)
![AI RAG](https://img.shields.io/badge/AI-RAG-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 🌟 Features

- **🛰️ Multi-platform Analysis**: Satellite imagery, UAV/drone data, and AI/ML techniques
- **📚 Comprehensive Database**: 22 peer-reviewed scientific articles (2009-2025)
- **🐧 Species Coverage**: Emperor, Chinstrap, Adélie, and Gentoo penguins
- **🤖 AI-Powered**: Uses Ollama LLMs for intelligent question answering
- **🎨 Beautiful Interface**: Modern Streamlit UI with Antarctic theme
- **📖 Source Citations**: Inline references to research papers

## 🔬 Specializations

- Antarctic penguin colony monitoring
- Satellite image analysis & processing
- Drone (UAV) remote sensing
- Population estimation & trend analysis
- Machine learning in wildlife monitoring
- Climate change impact assessment

## 📋 Requirements

- Python 3.8+
- Ollama installed and running
- Required Python packages (see `requirements.txt`)

## 🚀 Quick Start

### 1. Install Ollama
```bash
# macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from: https://ollama.ai/download
```

### 2. Download Required Models
```bash
# Main LLM model (recommended)
ollama pull qwen3:14b

# Embedding model
ollama pull all-minilm:latest

# Alternative smaller models (if memory limited):
# ollama pull llama3.1:8b
# ollama pull nomic-embed-text:latest
```

### 3. Clone and Setup
```bash
git clone https://github.com/YOUR_USERNAME/antarctic-penguin-rag.git
cd antarctic-penguin-rag

# Create virtual environment
python -m venv penguin_rag_env
source penguin_rag_env/bin/activate  # On Windows: penguin_rag_env\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Application
```bash
# Make sure Ollama is running
ollama serve

# Start the Streamlit app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📚 Research Database

The system includes 22 peer-reviewed scientific articles covering:

- Remote sensing methodologies (2009-2025)
- Penguin population dynamics and trends
- Climate change impacts on Antarctic ecosystems
- UAV/drone applications in wildlife monitoring
- Machine learning techniques for colony detection
- Breeding biology and habitat studies

## 🖥️ System Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Streamlit UI  │    │   RAG Chain  │    │   Ollama    │
│                 │◄──►│              │◄──►│   Models    │
└─────────────────┘    └──────────────┘    └─────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────┐
         │              │  ChromaDB    │
         │              │  VectorStore │
         └──────────────┤              │
                        └──────────────┘
                                │
                                ▼
                      ┌──────────────┐
                      │   22 PDF     │
                      │  Research    │
                      │  Articles    │
                      └──────────────┘
```

## 📊 Configuration

Key settings in `app.py`:
- **LLM Model**: `qwen3:14b` (configurable)
- **Embedding Model**: `all-minilm:latest`
- **Chunk Size**: 1000 characters
- **Retrieval**: Top 7 most relevant chunks
- **Temperature**: 0.1 (focused responses)

## 🎯 Example Questions

- "What satellite sensors are best for monitoring penguin colonies?"
- "How do UAV data compare to satellite imagery for penguin detection?"
- "What are the climate change impacts on Adélie penguin populations?"
- "How is machine learning used for automated colony detection?"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Antarctic research community for the scientific articles
- Ollama team for the LLM infrastructure
- Streamlit team for the web framework

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out through GitHub.

---
