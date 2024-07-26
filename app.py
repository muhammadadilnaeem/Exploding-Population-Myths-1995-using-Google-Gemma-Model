import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time

load_dotenv()

## Load the GROQ And OpenAI API KEY 
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title=" 📋 Exploding Population Myths Q&A", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main-heading {
            text-align: center;
            font-size: 2.5em;
            color: #4CAF50;
        }
        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .embed-button, .submit-button {
            display: inline-block;
            margin: 20px 10px;
            background-color: #4CAF50; /* Green */
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 1.2em;
            cursor: pointer;
            text-align: center;
        }
        .embed-button:hover, .submit-button:hover {
            background-color: #45a049;
        }
        .question-input {
            text-align: center;
            margin-top: 20px;
        }
        .response-heading {
            color: #3498db;
            font-size: 1.5em;
            margin-top: 20px;
        }
        .response {
            margin-top: 20px;
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
            font-size: 1.1em;
        }
        .spinner {
            display: block;
            margin: 0 auto;
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-radius: 50%;
            border-top: 5px solid #3498db;
            animation: spin 2s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# Main heading
st.markdown('<h1 class="main-heading">Q&A on "📚 Exploding Population Myths" (1995) using Google Gemma Model 🤖</h1>', unsafe_allow_html=True)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./document")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# Embedding creation button
with st.container():
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("📚 Create Document Embeddings", key='embed_button', help='Click to create embeddings for documents'):
        with st.spinner('Embeddings in progress...'):
            vector_embedding()
        st.success('Embeddings completed You can now ask questions. ✅')
    st.markdown('</div>', unsafe_allow_html=True)

# Question input
prompt1 = st.text_input("🔍 Enter Your Question From Documents", key='question_input')

# Submit query button
with st.container():
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("💬 Submit Query", key='submit_button', help='Click to submit your question'):
        if prompt1:
            with st.spinner('Searching for your question...'):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                st.markdown(f"<div class='response-heading'>📋 Response</div>", unsafe_allow_html=True)
                st.write(f"<div class='response'><strong></strong> {response['answer']}</div>", unsafe_allow_html=True)
                st.write(f"<div class='response'>Response time: {time.process_time() - start:.2f} seconds</div>", unsafe_allow_html=True)

                with st.expander("📄 Document Similarity Search"):
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("<hr>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)