# Phase 1 libraries
import os
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Phase 2 libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Phase 3 libraries (Updated Imports)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS  # Replacing Chroma with FAISS

# Disable warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Streamlit UI
st.title("Ask Chatbot!")

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Function to create a vector store
@st.cache_resource
def get_vectorstore():
    try:
        pdf_name = "./Binder & Hardeners.pdf"

        # Load PDF
        loader = PyPDFLoader(pdf_name)
        documents = loader.load()

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        # Create Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)  # Using FAISS

        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

# User input
prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Define Groq model and system prompt
    groq_prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant, always providing the best, most accurate, and precise answers.
        Answer the following Question: {user_prompt}. Start the answer directly."""
    )

    # Define Groq model
    model = "llama-3.3-70b-versatile"  # Use a valid model name

    try:
        # Initialize ChatGroq with API key
        groq_chat = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),  # Ensure API key is set in .env
            model=model
        )

        # Load vectorstore
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document.")
        else:
            # Create RetrievalQA chain
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )

            # Get response
            result = chain({"query": prompt})
            response = result["result"]

            # Display response
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {str(e)}")
