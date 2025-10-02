from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the PDF
pdf_files = ["book-no-0.pdf",  # Added missing book 0
             "book-no-1.pdf", 
             "book-no-2.pdf",
             "book-no-3.pdf",
             "book-no-4.pdf", 
             "book-no-5.pdf",
             "book-no-6.pdf",
             "book-no-7.pdf", 
             "book-no-8.pdf",
             "book-no-9.pdf",
             "book-no-10.pdf", 
             "book-no-11.pdf",
             "book-no-12.pdf"]  # add your PDFs here
documents = []

# Load documents with error handling
for pdf in pdf_files:
    try:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
        print(f"Successfully loaded {pdf}")
    except Exception as e:
        print(f"Error loading {pdf}: {e}")

print(f"Loaded {len(documents)} documents from {len(pdf_files)} PDFs")

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert texts to embeddings
try:
    embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
    print("Vector Embeddings created successfully")
except Exception as e:
    print(f"Error creating vector embeddings: {e}")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")

# Add documents to the vector store
vector_store.add_documents(documents=texts)
# Add documents if database is empty
if len(vector_store.get()) == 0:
    vector_store.add_documents(texts)
    vector_store.persist()  # <-- VERY IMPORTANT! Saves DB to disk
    print("Vector DB created and persisted successfully.")
else:
    print("Vector DB already exists, loaded from disk.")
# Validate the setup
try:
    # Test query to validate data retrieval
    test_query = "what is anti barrack movement?"
    results = vector_store.search(query=test_query, search_type='similarity')

    # Deduplicate results
    unique_results = OrderedDict()
    for doc in results:
        if doc.page_content not in unique_results:
            unique_results[doc.page_content] = doc

    # Convert unique results to a list and limit to top 3
    final_results = list(unique_results.values())[:3]
    print(f"Unique query results: {final_results}")
except Exception as e:
    print(f"Error during test query: {e}")
