# Sahayak - ASHA worker Companion

Sahayak is a Streamlit-based application designed to assist ASHA (Accredited Social Health Activist) workers in India with health information lookup, medical guidance, patient care recommendations, and community health resources. This is taken via the help of books provided by the government. [Go to Government Health Resources](https://nhm.gov.in/index1.php?lang=1&level=3&sublinkid=184&lid=257)



## Features

- __Bilingual Support__: Available in both English and Hindi to cater to diverse users
- __Learning Tab__: For general health information queries and educational purposes
- __Urgent Help Tab__: For immediate assistance with health emergencies
- __Vector Database Integration__: Uses ChromaDB to store and retrieve relevant health information
- __AI-Powered Responses__: Leverages the Cerebras API with LLaMA model for accurate and contextual responses
- __Recent Questions Tracking__: Keeps a history of recently asked questions for quick reference
- __Example Questions__: Provides sample questions to help users understand how to use the application

## 💻 Technologies Used

### 1️⃣ Generative AI / LLM
- **LLaMA** – Large Language Model for answering health questions  
- **Cerebras API** – Accessing LLaMA models   

### 2️⃣ Semantic Search / Knowledge Retrieval
- **ChromaDB** – Vector database for storing and searching health PDFs  
- **SentenceTransformers** – Creating embeddings for documents  

### 3️⃣ Web / Frontend
- **Streamlit** – Web app framework for interactive interface  
- **HTML, CSS** – Styling the web interface  

### 4️⃣ Backend / Logic
- **Python** – Core language for LLM integration, document search, and API calls  
- **APIs / HTTP requests** – Connecting to Cerebras API for LLM responses  

### 5️⃣ Data / Resources
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

- streamlit==1.36.0
- python-dotenv
- chromadb==0.4.2
- sentence-transformers==2.2.2
- cerebras-cloud-sdk
