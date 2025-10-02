# Sahayak - ASHA worker Companion
🚀 Live Demo: [Sahayak - ASHA Worker Companion](https://harsh-sahayak.streamlit.app/]


Sahayak is a Streamlit-based application designed to assist ASHA (Accredited Social Health Activist) workers in India. 
I developed a custom vector database by processing the official PDFs and guideline documents provided by government sources for ASHA workers. 
Using advanced embedding techniques, the documents were transformed into a structured, searchable knowledge base, enabling accurate and context-aware responses. This ensures that all information is sourced from authoritative government material rather than general internet data, making the system reliable, precise, and fully aligned with official guidelines.[Go to Government Health Resources](https://nhm.gov.in/index1.php?lang=1&level=3&sublinkid=184&lid=257)



## Features

- __Bilingual Support__: Available in both English and Hindi to cater to diverse users
- __Learning Tab__: For general health information queries and educational purposes
- __Urgent Help Tab__: For immediate assistance with health emergencies
- __Vector Database Integration__: Uses ChromaDB to store and retrieve relevant health information
- __AI-Powered Responses__: Leverages the Cerebras API with LLaMA model for accurate and contextual responses
- __Recent Questions Tracking__: Keeps a history of recently asked questions for quick reference
- __Example Questions__: Provides sample questions to help users understand how to use the application

## 💻 Technologies Used

### Generative AI / LLM
- **LLaMA** – Large Language Model for answering health questions  
- **Cerebras API** – Accessing LLaMA models   

### Semantic Search / Knowledge Retrieval
- **ChromaDB** – Vector database for storing and searching health PDFs  
- **SentenceTransformers** – Creating embeddings for documents  

### Web / Frontend
- **Streamlit** – Web app framework for interactive interface  
- **HTML, CSS** – Styling the web interface  

### Backend / Logic
- **Python** – Core language for LLM integration, document search, and API calls  
- **APIs / HTTP requests** – Connecting to Cerebras API for LLM responses  

### Data / Resources
- **Government Health PDFs** – Official resources for ASHA workers

## Installation

1. Clone the repository
2. Install the required dependencies
1. ```javascript
   pip install -r requirements.txt
   ```

2. Set up your Cerebras API key in a `.env` file:

   ```javascript
   CEREBRAS_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:

   ```javascript
   streamlit run app.py
   ```

2. Select your preferred language (English or Hindi) from the sidebar

3. Use either the Learning tab for general health queries or the Urgent Help tab for emergency assistance

4. Enter your question in the text input field

5. Click "Ask" to get AI-powered responses based on relevant health documents

## Potential Impact

This application has the potential to assist 900,000 ASHA workers across India, providing them with quick access to health information and guidance.

## Dependencies

# Core libraries
streamlit==1.50.0
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
scipy==1.15.3

# NLP / Embeddings
sentence-transformers==5.1.1
transformers==4.56.2
huggingface-hub==0.35.3

# LangChain
langchain==0.3.27
langchain-huggingface==0.3.1
langchain-community==0.3.30
langchain-core==0.3.76

# Vector DB
chromadb==1.1.0

# Backend / API
cerebras_cloud_sdk==1.50.1
python-dotenv==1.1.1
requests==2.32.5

# Optional / Utilities
tqdm==4.67.1
rich==14.1.0
