import os
import streamlit as st
st.set_page_config(
    page_title="Sahayak - ASHA Worker Assistant",
    page_icon="👩‍⚕️",
    layout="wide"
)
from dotenv import load_dotenv

#Cerebras
try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    st.warning("Cerebras SDK not installed. Please install it with: pip install cerebras-cloud-sdk")

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
CHROMA_DATA_DIR = os.path.join("data")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CEREBRAS_MODEL_NAME = "llama-4-scout-17b-16e-instruct"
TOP_K = 1

translations = {
    "en": {
        "title": "Sahayak - ASHA Worker Assistant",
        "ask_question": "Ask your question to ASHA assistant:",
        "submit_button": "Ask",
        "searching_db": "Searching vector DB...",
        "context_header": "Context from database",
        "no_context": "No relevant context found in vector DB.",
        "getting_answer": "Getting answer from Cerebras...",
        "response_header": "ASHA Assistant Response",
        "empty_question": "Please enter a non-empty question.",
        "api_key_error": "CEREBRAS_API_KEY not set in environment (or .env).",
        "chroma_error": "Failed to list Chroma collections: {}",
        "collection_error": "Could not open collection '{}': {}",
        "no_collection": "No collection found and could not create 'asha_temp'.",
        "chroma_query_error": "Chroma query failed: {}",
        "api_call_error": "Error calling Cerebras API: {}",
        "language_selector": "Select Language / भाषा चुनें",
        "english": "English",
        "hindi": "Hindi",
        "clear_button": "Clear",
        "recent_questions": "Recent Questions / हाल के प्रश्न",
        "learning_tab": "📚 Learning",
        "urgent_help_tab": "🚨 Urgent Help",
        "learning_placeholder": "Ask a question to learn about health topics...",
        "urgent_placeholder": "Ask for immediate help with a health emergency...",
    },
    "hi": {
        "title": "अशा कार्यकर्ता सहायक",
        "ask_question": "अशा सहायक से अपना प्रश्न पूछें:",
        "submit_button": "पूछें",
        "searching_db": "वेक्टर डेटाबेस खोजा जा रहा है...",
        "context_header": "डेटाबेस से संदर्भ",
        "no_context": "वेक्टर डेटाबेस में कोई प्रासंगिक संदर्भ नहीं मिला।",
        "getting_answer": "सीरेब्रास से उत्तर प्राप्त किया जा रहा है...",
        "response_header": "अशा सहायक प्रतिक्रिया",
        "empty_question": "कृपया एक गैर-रिक्त प्रश्न दर्ज करें।",
        "api_key_error": "पर्यावरण (या .env) में CEREBRAS_API_KEY निर्धारित नहीं है।",
        "chroma_error": "क्रोमा संग्रहों को सूचीबद्ध करने में विफल: {}",
        "collection_error": "संग्रह '{}' खोल नहीं सका: {}",
        "no_collection": "कोई संग्रह नहीं मिला और 'asha_temp' नहीं बना सका।",
        "chroma_query_error": "क्रोमा क्वेरी विफल: {}",
        "api_call_error": "सीरेब्रास एपीआई को कॉल करने में त्रुटि: {}",
        "language_selector": "भाषा चुनें",
        "english": "अंग्रेज़ी",
        "hindi": "हिंदी",
        "clear_button": "स्पष्ट करें",
        "recent_questions": "हाल के प्रश्न",
        "learning_tab": "📚 सीखना",
        "urgent_help_tab": "🚨 तत्काल सहायता",
        "learning_placeholder": "स्वास्थ्य विषयों के बारे में जानने के लिए प्रश्न पूछें...",
        "urgent_placeholder": "स्वास्थ्य आपातकाल के साथ तत्काल सहायता के लिए पूछें...",
    }
}

@st.cache_resource
def initialize_backend():
    # Cerebras client
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        st.error(translations["en"]["api_key_error"])
        return None, None, None

    if CEREBRAS_AVAILABLE:
        client = Cerebras(api_key=api_key)
    else:
        client = None

    # Embedding model
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Chroma persistent client
    chroma_path = os.path.join(CHROMA_DATA_DIR, os.listdir(CHROMA_DATA_DIR)[0]) 
    db = chromadb.PersistentClient(path=chroma_path)

    # Find an existing collection
    try:
        cols = db.list_collections()
    except Exception as e:
        st.error(translations["en"]["chroma_error"].format(e))
        return client, embedder, None

    collection = None
    if cols:
        first = cols[0]
        if hasattr(first, "name"):
            name = first.name
        elif isinstance(first, dict) and "name" in first:
            name = first["name"]
        elif isinstance(first, str):
            name = first
        else:
            name = str(first)

        try:
            collection = db.get_collection(name=name)
        except Exception:
            try:
                collection = db.get_or_create_collection(name=name)
            except Exception as e:
                st.warning(translations["en"]["collection_error"].format(name, e))
                collection = None
    else:
        # Create new collection
        try:
            collection = db.get_or_create_collection(name="asha_temp")
        except Exception:
            st.warning(translations["en"]["no_collection"])
            collection = None

    return client, embedder, collection

# Initialize backend components
cerebras_client, embedding_model, collection = initialize_backend()

# ---- HELPERS ----
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_relevant_docs_via_chroma(query: str, top_k: int = TOP_K):
    if not collection:
        return []
    if not query or not query.strip():
        return []
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs_for_query = results.get("documents", [[]])[0]
        return [d for d in docs_for_query if d and isinstance(d, str)]
    except Exception as e:
        st.warning(translations["en"]["chroma_query_error"].format(e))
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def ask_cerebras(_client, question: str, context_docs: list):
    if not CEREBRAS_AVAILABLE or not _client:
        return "Cerebras API is not available. Please install the SDK and set up your API key."
    
    context = "\n\n".join(context_docs) if context_docs else ""
    system_msg = "You are an ASHA worker assistant. Answer succinctly and clearly for a community health worker."
    user_msg = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        completion = _client.chat.completions.create(messages=messages, model=CEREBRAS_MODEL_NAME)
        try:
            return completion.choices[0].message.content
        except Exception:
            try:
                return completion.choices[0].message.get("content", str(completion))
            except Exception:
                return str(completion)
    except Exception as e:
        return translations["en"]["api_call_error"].format(e)

# ---- SESSION STATE ----
if "language" not in st.session_state:
    st.session_state.language = "en"

if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []


# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
    
    .question-card {
        background-color: #e1f0fa;
        border-left: 5px solid #3498db;
    }
    
    .context-card {
        background-color: #fef9e7;
        border-left: 5px solid #f1c40f;
    }
    
    .response-card {
        background-color: #d5f5e3;
        border-left: 5px solid #27ae60;
    }
    
    .recent-card {
        background-color: #ebdef0;
        border-left: 5px solid #8e44ad;
    }
    
    /* Learning section card */
    .learning-card {
        background-color: #e1f0fa;
        border-left: 5px solid #3498db;
    }
    
    /* Urgent help section card */
    .urgent-card {
        background-color: #fadbd8;
        border-left: 5px solid #e74c3c;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .clear-button>button {
        background-color: #e74c3c;
    }
    
    .clear-button>button:hover {
        background-color: #c0392b;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding: 10px;
        
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d6eaf8;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #3498db;
        padding: 0.8rem;
        font-size: 1.1rem;
    }
    
    /* Subheader styling */
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    
    /* Info and warning styling */
    .stAlert {
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Spinner styling */
    .stSpinner>div {
        color: #3498db;
        font-size: 1.1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Main header with language support
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
# st.markdown(f"<h1 class='main-title'>{translations[st.session_state.language]['title']}</h1>", unsafe_allow_html=True)
# Main Title
st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50;'>
        Sahayak – Your ASHA Companion 👩‍⚕️
    </h1>
    """,
    unsafe_allow_html=True
)
# Tagline
st.markdown(
    """
    <p style='text-align: center; font-size:18px; color: #555;'>
        Your trusted assistant for community health work <br>
        <span style='font-size:16px; color:#888;'>
        Built using <b>Cerebras API</b> + <b>LLaMA</b>, with the potential to impact <b>900,000 ASHA workers</b> across India.
        </span>
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Language selector in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.language = st.selectbox(
        translations["en"]["language_selector"],
        options=["en", "hi"],
        format_func=lambda x: translations[x]["english"] if x == "en" else translations[x]["hindi"],
        index=0 if st.session_state.language == "en" else 1
    )

    
    st.markdown("---")
    st.markdown("### 📝 About ASHA                (Accredited Social Health Activist) Assistant")
    if st.session_state.language == "en":
        st.markdown("""
        This assistant helps ASHA workers with:
        - Health information lookup
        - Medical guidance
        - Patient care recommendations
        - Community health resources
        """)
    else:
        st.markdown("""
        यह सहायक अशा कार्यकर्ताओं की मदद करता है:
        - स्वास्थ्य जानकारी खोज
        - चिकित्सा मार्गदर्शन
        - रोगी देखभाल की सिफारिशें
        - सामुदायिक स्वास्थ्य संसाधन
        """)

# Get translations for current language
t = translations[st.session_state.language]

# Recent questions in sidebar
with st.sidebar:
    if st.session_state.recent_questions:
        st.markdown("<div class='recent-card card'>", unsafe_allow_html=True)
        st.subheader(t["recent_questions"])
        for q in st.session_state.recent_questions[-5:]:  # Show last 5 questions
            st.markdown(f"- {q}")
        st.markdown("</div>", unsafe_allow_html=True)

# Main content area with tabs
col1, col2 = st.columns([5, 1])

with col1:
    # Create tabs for Learning and Urgent Help
    tab1, tab2 = st.tabs([t["learning_tab"], t["urgent_help_tab"]])
    
    with tab1:
        # st.markdown("<div class='learning-card card'>", unsafe_allow_html=True)
        st.markdown(f"<h2 class='section-header'>{t['learning_tab']}</h2>", unsafe_allow_html=True)
        
        with st.form("learning_form"):
            learning_query = st.text_input("", placeholder=t["learning_placeholder"], label_visibility="collapsed")
            learning_submitted = st.form_submit_button(t["submit_button"])
            
            if learning_submitted:
                if not learning_query or not learning_query.strip():
                    st.warning(t["empty_question"])
                else:
                    # Add to recent questions
                    if learning_query not in st.session_state.recent_questions:
                        st.session_state.recent_questions.append(learning_query)
                    
                    # Get relevant documents
                    with st.spinner(t["searching_db"]):
                        docs = get_relevant_docs_via_chroma(learning_query, TOP_K)
                    # Get answer from Cerebras
                    with st.spinner(t["getting_answer"]):
                        answer = ask_cerebras(cerebras_client, learning_query, docs)

                    # Display response in main content area
                    st.markdown(f"<h2 class='section-header'>{t['response_header']}</h2>", unsafe_allow_html=True)
                    st.markdown("<div class='response-card card'>", unsafe_allow_html=True)
                    st.write(answer)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Display context in main content area
                    st.markdown(f"<h2 class='section-header'>{t['context_header']}</h2>", unsafe_allow_html=True)
                    st.markdown("<div class='context-card card'>", unsafe_allow_html=True)
                    if docs:
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**{i}.** {d}")
                    else:
                        st.info(t["no_context"])
                    st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f"<h2 class='section-header'>{t['urgent_help_tab']}</h2>", unsafe_allow_html=True)
        
        with st.form("urgent_form"):
            urgent_query = st.text_input("", placeholder=t["urgent_placeholder"], label_visibility="collapsed")
            urgent_submitted = st.form_submit_button(t["submit_button"])
            
            if urgent_submitted:
                if not urgent_query or not urgent_query.strip():
                    st.warning(t["empty_question"])
                else:
                    # Add to recent questions
                    if urgent_query not in st.session_state.recent_questions:
                        st.session_state.recent_questions.append(urgent_query)
                    
                    # Get relevant documents
                    with st.spinner(t["searching_db"]):
                        docs = get_relevant_docs_via_chroma(urgent_query, TOP_K)

                    # Get answer from Cerebras
                    with st.spinner(t["getting_answer"]):
                        answer = ask_cerebras(cerebras_client, urgent_query, docs)

                    # Display response in main content area
                    st.markdown(f"<h2 class='section-header'>{t['response_header']}</h2>", unsafe_allow_html=True)
                    st.markdown("<div class='response-card card'>", unsafe_allow_html=True)
                    st.write(answer)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Display context in main content area
                    st.markdown(f"<h2 class='section-header'>{t['context_header']}</h2>", unsafe_allow_html=True)
                    st.markdown("<div class='context-card card'>", unsafe_allow_html=True)
                    if docs:
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**{i}.** {d}")
                    else:
                        st.info(t["no_context"])
                    st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 💡 Example Questions")
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.session_state.language == "en":
        st.markdown("""
        - What are the symptoms of malaria?
        - How to provide first aid for burns?
        - What vaccines are given to newborns?
        - How to identify malnutrition in children?
        - What are the danger signs during pregnancy?
        """)
    else:
        st.markdown("""
        - मलेरिया के लक्षण क्या हैं?
        - बर्न के लिए प्रथम चिकित्सा कैसे प्रदान करें?
        - नवजात शिशुओं को कौन से टीके दिए जाते हैं?
        - बच्चों में कुपोषण की पहचान कैसे करें?
        - गर्भावस्था के दौरान खतरे के संकेत क्या हैं?
        """)

    st.markdown("[Go to Resources & Books](https://nhm.gov.in/index1.php?lang=1&level=3&sublinkid=184&lid=257)", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
