#!/usr/bin/env python3
"""
Antarctic Penguin Colonies RAG System
Remote Sensing Research Intelligence Assistant

This script creates a RAG (Retrieval Augmented Generation) system specialized in
Antarctic penguin colony research using remote sensing technologies.
"""

# Standard imports
import os
import logging
import tempfile
from typing import List
import re
import base64

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Streamlit for web interface
import streamlit as st

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration parameters
PERSIST_DIRECTORY = 'chroma_db'  # directory to store the vector database
CHUNK_SIZE = 1000  # characters per chunk for text splitting
CHUNK_OVERLAP = 50  # characters of overlap between chunks
PDF_URLS = [
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Belyaev et al. 2023.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Belyaev et al. 2024.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Bird et al. 2020.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Borowicz et al. 2018.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Boyer et al. 2025.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Firla et al. 2019.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Fretwell et al. 2009.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Fretwell et al. 2015.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Lin et al. 2025.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Lynch et al. 2012.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Naveen et al. 2012.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Pfeifer et al. 2019.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Ratcliffe et al. 2015.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Román et al. 2022.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Roman et al. 2024.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Rummler et al. 2018.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Rummler et al. 2021.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Schwaller et al. 2013.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Strycker et al. 2020.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Trathan et al. 2011.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Wehner et al. 2025.pdf',
    r'/Users/alejandroroman/Documents/repos/bootcamp/ragollamavenv/data/Zmarz et al. 2018.pdf'
]
LLM_MODEL = 'qwen3:14b'  # Ollama model for LLM
EMBEDDING_MODEL = 'all-minilm:latest'  # Ollama model for embeddings
TEMPERATURE = 0.1


class RAGSystem:
    def __init__(self, pdf_urls: List[str], persist_directory: str = PERSIST_DIRECTORY):
        self.pdf_urls = pdf_urls
        self.persist_directory = persist_directory
        self.documents = []
        self.vectorstore = None
        self.llm = None
        self.chain = None

        # Initialize the LLM with streaming capability
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            callback_manager=callback_manager,
            # Additional parameters to reduce internal thinking
            top_p=0.9,
            repeat_penalty=1.1
        )

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        logger.info(f"Initialized RAG system with {len(pdf_urls)} PDFs")

    def load_documents(self) -> None:
        """Load and split PDF documents"""
        logger.info("Loading and processing PDFs...")

        # Text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        all_pages = []  # List to hold all pages from all PDFs

        for url in self.pdf_urls:
            try:
                loader = PyPDFLoader(url)
                pages = loader.load()
                logger.info(f"Loaded {len(pages)} pages from {url}")
                all_pages.extend(pages)
            except Exception as e:
                logger.error(f"Error loading PDF from {url}: {e}")

        # Split the documents into chunks
        self.documents = text_splitter.split_documents(all_pages)
        logger.info(f"Created {len(self.documents)} document chunks")

    def create_vectorstore(self) -> None:
        """Create a fresh vector database"""
        # Remove any existing database
        if os.path.exists(self.persist_directory):
            import shutil
            logger.info(f"Removing existing vectorstore at {self.persist_directory}")
            shutil.rmtree(self.persist_directory, ignore_errors=True)

        # Create a new vectorstore
        logger.info("Creating new vectorstore...")
        if not self.documents:
            self.load_documents()

        # Create a temporary directory for the database
        # This helps avoid permission issues on some systems
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory for initial database creation: {temp_dir}")

        try:
            # First create in temp directory
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=temp_dir
            )

            # Now create the real directory
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory)

            # And create the final vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()

            logger.info(f"Vectorstore created successfully with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise
        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

    def setup_chain(self) -> None:
        """Set up the RAG chain for question answering"""
        if not self.vectorstore:
            self.create_vectorstore()

        # Create retriever with search parameters
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7}  # Return top 7 most relevant chunks
        )

        # Define the prompt template
        template = """
### INSTRUCTIONS:
        You are a postdoctoral researcher specialized in Antarctic ecology and remote sensing,
        with expertise in penguin population monitoring, satellite image analysis, and wildlife conservation.
        Your focus is on applying remote sensing technologies and machine learning techniques to study
        penguin colonies in Antarctica, their population dynamics, and environmental impacts.
        Base your answers strictly on the provided scientific publications (the context below).
        Be polite, professional, and avoid guessing or using outside sources.

        IMPORTANT: Provide ONLY your final answer. Do not show any internal thinking, reasoning process, or preliminary analysis. Go straight to the answer.

        (1) Be attentive to details: read the question and the context thoroughly before answering.
        (2) Begin your response with a friendly tone and briefly restate the user's question to confirm understanding.
        (3) If the context allows you to answer the question, write a detailed, clear, and rigorous response.
        - Use precise terminology from the publications (e.g., satellite sensors, multispectral imagery, population estimation, colony distribution, breeding sites, remote sensing techniques).
        - When relevant, include methodological explanations, processing workflows, or technical specifications.
        - Reference the sources **inline** (e.g., [Article A §2.3], [Dataset B Fig.5]) and ONLY cite content that appears in the provided context.
        - Keep the explanation accessible while preserving scientific accuracy.

       IF NOT: if you cannot find the answer, respond with an explanation starting with:
       "I couldn't find the information in the documents I have access to."

        (4) Below your response, list all referenced sources (document titles/IDs and exact sections/figures/tables that support your claims).
        (5) Review your answer to ensure it is accurate, well-structured, and professional. Use short paragraphs or bullet points where helpful.

        Additional constraints:
        - Do not invent citations or content outside the provided context.
        - Use appropriate units consistently (e.g., km² for colony areas, individuals for population counts, pixels for image resolution).
        - Include technical details about satellite sensors, image processing methods, or statistical analyses only if explicitly present in the context.
        - If the documents contain conflicting information, acknowledge the discrepancy and cite both sources.
        - Focus on practical applications of remote sensing for penguin monitoring and conservation.
        - DO NOT include any thinking process, analysis steps, or reasoning. Provide only the final answer.

        Answer the following question using the provided context.
        ### Question: {question} ###
        ### Context: {context} ###
        ### Helpful Answer with Sources:
        """

        prompt = PromptTemplate.from_template(template)

        # Create the chain
        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("RAG chain setup complete")

    def clean_answer(self, answer: str) -> str:
        """
        Clean the answer by removing any internal thinking tags or unwanted content

        Args:
            answer: Raw answer from the model

        Returns:
            Cleaned answer
        """
        import re

        # Remove content between <think> and </think> tags (case insensitive)
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.IGNORECASE | re.DOTALL)

        # Remove any other thinking patterns that might appear
        answer = re.sub(r'<thinking>.*?</thinking>', '', answer, flags=re.IGNORECASE | re.DOTALL)

        # Remove standalone thinking indicators
        answer = re.sub(r'^Okay,.*?Let me.*?\n', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'^First,.*?Let me.*?\n', '', answer, flags=re.MULTILINE)

        # Clean up extra whitespace
        answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)
        answer = answer.strip()

        return answer

    def answer_question(self, question: str) -> str:
        """
        Answer a question using the RAG chain

        Args:
            question: The question to answer

        Returns:
            The answer to the question
        """
        if not self.chain:
            self.setup_chain()

        logger.info(f"Answering question: {question}")
        try:
            answer = self.chain.invoke(question)
            # Clean the answer to remove any thinking content
            cleaned_answer = self.clean_answer(answer)
            return cleaned_answer
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error processing your question: {str(e)}"


def get_sample_questions():
    """Return categorized sample questions about Antarctic penguin colonies and remote sensing"""
    return {
        "🛰️ Remote Sensing & UAV Methodologies": [
            "What satellite sensors are commonly used for monitoring penguin colonies in Antarctica?",
            "How are high-resolution satellite images processed to identify penguin colonies?",
            "What are the advantages of using UAV/drone data compared to satellite imagery for penguin monitoring?",
            "How does temporal analysis of satellite and drone data help monitor colony population changes?",
            "What role does machine learning play in automated penguin colony detection from multi-platform data?"
        ],
        "🐧 Colony Distribution & Characteristics": [
            "How are penguin colonies distributed across the Antarctic Peninsula?",
            "What factors influence the location and size of penguin breeding colonies?",
            "How do environmental variables affect colony site selection in Antarctica?",
            "What are the main characteristics of emperor penguin colonies versus Adélie penguin colonies?",
            "How do sea ice conditions impact penguin colony accessibility and monitoring?"
        ],
        "📊 Population Monitoring & Trends": [
            "What methods are used to estimate penguin population sizes from satellite imagery?",
            "How has penguin colony population changed over the last decades according to remote sensing data?",
            "What are the main challenges in accurately counting penguin populations from space?",
            "How do researchers validate satellite-based population estimates with ground truth data?",
            "What temporal patterns are observed in penguin colony occupancy throughout breeding seasons?"
        ],
        "🌡️ Climate Change Impacts": [
            "How is climate change affecting penguin colony distribution in Antarctica?",
            "What evidence exists for penguin colony range shifts due to warming temperatures?",
            "How do changes in sea ice extent impact penguin breeding success and colony dynamics?",
            "What role does snow cover play in penguin colony monitoring and breeding habitat availability?",
            "How are extreme weather events affecting penguin colonies as observed through remote sensing?"
        ]
    }


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching for better performance"""
    try:
        with st.spinner("🔄 Initializing RAG system..."):
            rag_system = RAGSystem(pdf_urls=PDF_URLS)

            # Load documents and create vectorstore
            st.info("📚 Loading research documents...")
            rag_system.load_documents()

            st.info("🔍 Creating vector embeddings...")
            rag_system.create_vectorstore()

            st.success("✅ RAG system initialized successfully!")
            return rag_system
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {e}")
        return None


def get_base64_image(image_path: str) -> str:
    """Convert image to base64 string for HTML embedding"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return ""


def extract_sources_from_answer(answer: str) -> List[str]:
    """Extract source citations from the answer"""
    citations = re.findall(r'\[.*?\]', answer)
    return list(set(citations))


def main():
    """Main Streamlit application"""

    # Page configuration
    st.set_page_config(
        page_title="🐧 Antarctic Penguin Research Intelligence",
        page_icon="🐧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load background image
    background_image_b64 = get_base64_image("img/fondo.jpeg")

    # Custom CSS styling with background image
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    * {{
        font-family: 'Poppins', sans-serif;
    }}

    .main-header {{
        background:
            linear-gradient(135deg, rgba(15, 76, 117, 0.7) 0%, rgba(50, 130, 184, 0.7) 50%, rgba(187, 225, 250, 0.7) 100%),
            url('data:image/jpeg;base64,{background_image_b64}') center/cover no-repeat;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15, 76, 117, 0.3);
        position: relative;
        overflow: hidden;
    }}

    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="30" r="1.5" fill="rgba(255,255,255,0.1)"/><circle cx="40" cy="70" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="90" cy="80" r="2.5" fill="rgba(255,255,255,0.1)"/></svg>');
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0) rotate(0deg); }}
        50% {{ transform: translateY(-20px) rotate(180deg); }}
    }}

    .penguin-icon {{
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 2s ease-in-out infinite;
    }}

    @keyframes bounce {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}

    .question-container {{
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border: none;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }}

    .question-container::before {{
        content: '🐧';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 2rem;
        opacity: 0.1;
    }}

    .sidebar-content {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }}

    .feature-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #3282b8;
        transition: transform 0.3s ease;
    }}

    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }}

    .source-box {{
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1976d2;
        margin-top: 1.5rem;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.1);
    }}

    .tip-card {{
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.1);
    }}

    .warning-card {{
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.1);
    }}

    .category-header {{
        background: linear-gradient(135deg, #3282b8 0%, #0f4c75 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
    }}

    .stButton button {{
        background: linear-gradient(135deg, #3282b8 0%, #0f4c75 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(50, 130, 184, 0.3);
    }}

    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(50, 130, 184, 0.4);
    }}

    .antarctic-bg {{
        background: linear-gradient(180deg, #87ceeb 0%, #ffffff 30%, #f0f8ff 100%);
        min-height: 100vh;
    }}

    .ice-decoration {{
        position: absolute;
        width: 100px;
        height: 20px;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 50px;
        animation: drift 15s ease-in-out infinite;
    }}

    @keyframes drift {{
        0%, 100% {{ transform: translateX(-10px); }}
        50% {{ transform: translateX(10px); }}
    }}

    .penguin-gallery {{
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }}

    .penguin-card {{
        background: white;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        max-width: 200px;
        overflow: hidden;
    }}

    .penguin-card:hover {{
        transform: scale(1.05);
    }}

    .penguin-image {{
        width: 120px;
        height: 120px;
        object-fit: cover;
        border-radius: 50%;
        margin: 0 auto 0.5rem auto;
        display: block;
        border: 3px solid #3282b8;
        transition: border-color 0.3s ease;
    }}

    .penguin-card:hover .penguin-image {{
        border-color: #0f4c75;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Header with penguin animation and gallery
    st.markdown("""
    <div class="main-header">
        <span class="penguin-icon">🐧</span>
        <h1>Antarctic Penguin Research Intelligence</h1>
        <p style="font-size: 1.3rem; margin: 1rem 0; opacity: 0.95; font-weight: 300;">
            🤖 AI-Powered Remote Sensing Research Assistant 🛰️
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Penguin species gallery with real photos
    st.markdown("""
    <div class="penguin-gallery">
        <div class="penguin-card">
            <img src="data:image/jpg;base64,{}" class="penguin-image" alt="Emperor Penguin">
            <h4 style="color: #0f4c75; margin: 0;">Emperor</h4>
            <small style="color: #666;">Aptenodytes forsteri</small>
        </div>
        <div class="penguin-card">
            <img src="data:image/jpg;base64,{}" class="penguin-image" alt="Chinstrap Penguin">
            <h4 style="color: #0f4c75; margin: 0;">Chinstrap</h4>
            <small style="color: #666;">Pygoscelis antarcticus</small>
        </div>
        <div class="penguin-card">
            <img src="data:image/jpg;base64,{}" class="penguin-image" alt="Adélie Penguin">
            <h4 style="color: #0f4c75; margin: 0;">Adélie</h4>
            <small style="color: #666;">Pygoscelis adeliae</small>
        </div>
        <div class="penguin-card">
            <img src="data:image/jpg;base64,{}" class="penguin-image" alt="Gentoo Penguin">
            <h4 style="color: #0f4c75; margin: 0;">Gentoo</h4>
            <small style="color: #666;">Pygoscelis papua</small>
        </div>
    </div>
    """.format(
        get_base64_image("img/emperor.jpg"),
        get_base64_image("img/Chinstrap.jpg"),
        get_base64_image("img/adelie.jpg"),
        get_base64_image("img/gentoo.jpg")
    ), unsafe_allow_html=True)

    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2 style="text-align: center; margin-bottom: 1.5rem;">🐧 Antarctic Penguin Research Intelligence ❄️</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Specializations</h3>
            <ul style="margin: 0;">
                <li>🐧 Antarctic penguin colony monitoring</li>
                <li>🛰️ Satellite image analysis & processing</li>
                <li>🚁 Drone (UAV) remote sensing</li>
                <li>📈 Population estimation & trend analysis</li>
                <li>🤖 Machine learning in wildlife monitoring</li>
                <li>🌡️ Climate change impact assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>📚 Enhanced Knowledge Base</h3>
            <ul style="margin: 0;">
                <li>📄 22 peer-reviewed scientific articles (2009-2025)</li>
                <li>🔬 Remote sensing & UAV methodologies</li>
                <li>🧭 Antarctic research & conservation studies</li>
                <li>🥚 Population monitoring & breeding biology</li>
                <li>🌍 Climate change & environmental impacts</li>
                <li>🤖 AI/ML applications in wildlife research</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>✨ Advanced Features</h3>
            <ul style="margin: 0;">
                <li>📖 Inline citations from 22+ research articles</li>
                <li>🔬 Technical remote sensing & UAV terminology</li>
                <li>⚙️ Detailed methodological explanations</li>
                <li>📊 Multi-temporal population trend analysis</li>
                <li>🌡️ Climate change impact assessments</li>
                <li>🚁 Drone-based monitoring techniques</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # System configuration with enhanced styling
        st.markdown("---")
        st.markdown("""
        <div class="feature-card">
            <h3>⚙️ System Configuration</h3>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>🧠 LLM Model:</strong> {}<br>
                <strong>🔍 Embedding Model:</strong> {}<br>
                <strong>📚 Documents:</strong> {} research articles
            </div>
        </div>
        """.format(LLM_MODEL, EMBEDDING_MODEL, len(PDF_URLS)), unsafe_allow_html=True)

        # Add comprehensive research database info
        st.markdown("""
        <div class="feature-card">
            <h3>🌡️ Research Database Stats</h3>
            <div style="font-size: 0.9rem;">
                <p><strong>🏔️ Coverage:</strong> All Antarctic regions</p>
                <p><strong>🐧 Species:</strong> 4+ penguin species</p>
                <p><strong>📅 Time Span:</strong> 2009-2025 research</p>
                <p><strong>🛰️ Technologies:</strong> Satellites, UAVs, AI/ML</p>
                <p><strong>📚 Articles:</strong> 22 peer-reviewed papers</p>
                <p><strong>🔬 Methods:</strong> Multi-platform remote sensing</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = initialize_rag_system()

    # Check if system is ready
    if st.session_state.rag_system is None:
        st.error("❌ RAG system failed to initialize. Please check the logs and try again.")
        st.stop()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>💭 Ask Your Research Question</h2>
            <p style="color: #666; margin-top: 0.5rem;">
                🤔 What would you like to know about Antarctic penguin research and remote sensing?
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Question input with enhanced styling
        question = st.text_area(
            "✍️ Your Question:",
            placeholder="🔍 e.g., 'What satellite sensors are best for monitoring penguin colonies?' 🛰️",
            height=120,
            key="question_input",
            help="Ask about remote sensing techniques, penguin species, colony monitoring, or climate impacts! 🐧"
        )

        # Options with better styling
        col_submit, col_sources = st.columns([1, 1])

        with col_submit:
            submit_button = st.button("🔍 Analyze Research 🚀", type="primary", use_container_width=True)

        with col_sources:
            show_sources = st.checkbox("📚 Show source citations", value=True, help="Display references from research papers")

        # Example questions with enhanced categories
        st.markdown("""
        <div class="feature-card">
            <h2>🎯 Quick Example Questions</h2>
            <p style="color: #666; margin-top: 0.5rem;">
                Click on any question below to try it out! 👇
            </p>
        </div>
        """, unsafe_allow_html=True)

        sample_questions = get_sample_questions()

        for category, questions in sample_questions.items():
            with st.expander(f"{category} ❓"):
                for q in questions:
                    if st.button(f"❓ {q}", key=f"example_{hash(q)}", help="Click to use this question", use_container_width=True):
                        st.session_state.question_input = q
                        st.rerun()

    with col2:
        # Enhanced tips and information cards
        st.markdown("""
        <div class="tip-card">
            <h3>💡 Research Tips</h3>
            <p><strong>🎯 For best results:</strong></p>
            <ul style="margin: 0.5rem 0;">
                <li>🔬 Be specific in your questions</li>
                <li>⚙️ Ask about technical methods & processes</li>
                <li>🐧 Reference specific penguin species</li>
                <li>🗺️ Ask about temporal or spatial patterns</li>
                <li>📊 Inquire about data analysis methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ Important Notice</h3>
            <p style="margin: 0;">
                📚 All responses are based strictly on the loaded research documents.
                🚫 The system cannot access external information beyond the provided scientific articles.
                ✅ Perfect for academic research and citation accuracy!
            </p>
        </div>
        """, unsafe_allow_html=True)


    # Process question when submitted
    if submit_button and question.strip():
        st.markdown("""
        <div class="feature-card">
            <h2>🎓 Expert Analysis Results</h2>
            <p style="color: #666; margin-top: 0.5rem;">
                🧠 AI-powered analysis of Antarctic penguin research literature
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🐧 Analyzing penguin research data... 🔍"):
            try:
                # Get answer from RAG system
                answer = st.session_state.rag_system.answer_question(question)

                # Display answer with enhanced styling
                with st.container():
                    st.markdown('<div class="question-container">', unsafe_allow_html=True)
                    st.markdown(f"### 🐧 Research Analysis\n\n{answer}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Display sources if requested
                if show_sources and answer:
                    sources = extract_sources_from_answer(answer)
                    if sources:
                        st.markdown('<div class="source-box">', unsafe_allow_html=True)
                        st.markdown("### 📚 Scientific Sources Referenced:")
                        st.markdown("*The following research articles and datasets support this analysis:*")
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**[{i}]** {source}")
                        st.markdown('</div>', unsafe_allow_html=True)

                # Add feedback section
                st.markdown("""
                <div style="margin-top: 2rem; text-align: center;">
                    <p style="color: #666; font-size: 0.9rem;">
                        🎯 <strong>Analysis Complete!</strong> This response is based on {count} peer-reviewed research articles (2009-2025) about Antarctic penguin colonies, satellite monitoring, and UAV remote sensing.
                    </p>
                </div>
                """.format(count=len(PDF_URLS)), unsafe_allow_html=True)

            except Exception as e:
                st.markdown("""
                <div class="warning-card">
                    <h3>❌ Processing Error</h3>
                    <p><strong>Error:</strong> {}</p>
                    <p><strong>💡 Solution:</strong> Please check that Ollama is running and models are available:</p>
                    <ul>
                        <li>🔄 Run: <code>ollama serve</code></li>
                        <li>📥 Ensure models are downloaded: <code>ollama pull qwen3:14b</code></li>
                        <li>🧊 Check model availability: <code>ollama list</code></li>
                    </ul>
                </div>
                """.format(str(e)), unsafe_allow_html=True)

    elif submit_button and not question.strip():
        st.markdown("""
        <div class="tip-card">
            <h3>🚀 Ready to Start!</h3>
            <p style="margin: 0;">
                ✍️ Please enter a research question to begin your Antarctic penguin analysis!
                Try one of the example questions above or ask your own. 🐧
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-top: 3rem;">
        <h3 style="color: #0f4c75; margin: 0 0 1rem 0;">🐧 Antarctic Penguin Research Intelligence ❄️</h3>
        <p style="color: #666; margin: 0.5rem 0; font-size: 1rem;">
            🤖 Powered by <strong>LangChain</strong> + <strong>Ollama</strong> + <strong>Streamlit</strong>
        </p>
        <p style="color: #888; font-size: 0.9rem; margin: 1rem 0;">
            🔬 Specialized AI assistant for Antarctic ecology and remote sensing research<br>
            📚 Based on peer-reviewed scientific literature about penguin colony monitoring
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin: 1rem 0; flex-wrap: wrap;">
            <span style="background: #e3f2fd; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.8rem;">
                🛰️ Satellite Data Analysis
            </span>
            <span style="background: #f3e5f5; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.8rem;">
                🐧 Population Monitoring
            </span>
            <span style="background: #e8f5e8; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.8rem;">
                🌡️ Climate Impact Studies
            </span>
        </div>
        <p style="color: #aaa; font-size: 0.8rem; margin: 1rem 0 0 0;">
            🧊 Advancing Antarctic research through AI • © 2024 • Made with ❤️ for penguin conservation
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Check if we're running in Streamlit
    try:
        # This will only work if we're in a Streamlit app
        st.write()
        main()
    except:
        # If we're running directly, provide instructions
        print("\n" + "="*60)
        print("🐧 ANTARCTIC PENGUIN COLONIES RAG SYSTEM")
        print("   Remote Sensing Research Intelligence Assistant")
        print("="*60)
        print("\nTo run the Streamlit application, use:")
        print("streamlit run app.py")
        print("\nThis will launch the web interface at http://localhost:8501")
        print("="*60)