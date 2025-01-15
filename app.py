from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import streamlit as st
import time
import os

class RAGSystem:
    def __init__(self, model_name="llama2"):
        # Get API key from Streamlit secrets
        self.api_key = st.secrets["OPENAI_API_KEY"]
        
        # Use OpenAI instead of Ollama
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = OpenAI(openai_api_key=self.api_key)
        self.vector_store = None
        
        self.prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know.
        
        Context: {context}
        Question: {question}
        Answer: Let me provide a detailed response based on the context provided:
        """

    #STEP 1: TEXT PROCESSING
    def process_text(self, pdf_paths, status_placeholder):
        documents = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
            status_placeholder.write("📄 Text extracted from PDF...")
            time.sleep(0.5)  # Simulate processing time
        
        status_placeholder.write("✅ Text extraction complete")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        status_placeholder.write("🔄 Creating semantic chunks...")
        time.sleep(0.5)  # Simulate processing time
        status_placeholder.write("✅ Semantic chunking complete")
        return chunks
    
    #STEP 2: Embedding Generation
    def generate_embeddings(self, chunks, status_placeholder):
        status_placeholder.write("🔄 Generating embeddings with OpenAI...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        time.sleep(0.5)
        status_placeholder.write("✅ Embeddings generated and stored in ChromaDB")
        return self.vector_store

    #STEP 3: User Query Processing
    def process_query(self, query, status_placeholder):
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process documents first.")
        status_placeholder.write("🔄 Processing query...")
        time.sleep(0.5)  # Simulate processing time
        status_placeholder.write("✅ Query processed")
        return query

    #STEP 4: Similarity Matching
    def similarity_matching(self, processed_query, status_placeholder):
        status_placeholder.write("🔄 Finding relevant documents...")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        similar_chunks = retriever.invoke(processed_query)
        time.sleep(0.5)  # Simulate processing time
        status_placeholder.write("✅ Relevant documents retrieved")
        return similar_chunks

    #STEP 5: Context Integeratiopn
    def integrate_context(self):
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": prompt}
        )
        return chain
    
    # STEP 6: Response Generation
    def generate_response(self, query, status_placeholder):
        status_placeholder.write("🔄 Generating response...")
        chain = self.integrate_context()
        response = chain.invoke({"query": query})
        time.sleep(0.5)  # Simulate processing time
        status_placeholder.write("✅ Response generated")
        return response["result"]

    def full_process(self, pdf_paths, query, status_placeholder):
        chunks = self.process_text(pdf_paths, status_placeholder)
        self.generate_embeddings(chunks, status_placeholder)
        processed_query = self.process_query(query, status_placeholder)
        similar_chunks = self.similarity_matching(processed_query, status_placeholder)
        response = self.generate_response(query, status_placeholder)
        return response, len(chunks)

# STREAMLIT APP USER-INTERFACE
def create_streamlit_ui():
    st.set_page_config(layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.title("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True
        )
       
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} documents uploaded")
            
            st.write("Uploaded files:")
            for file in uploaded_files:
                st.write(f"- {file.name}")
    
        # Adding Footer
        st.markdown("""
                <style>
                    [data-testid=stSidebar] [data-testid=stVerticalBlock] {
                        gap: 0rem;
                        padding-bottom: 1rem;
                    }
                    .footer {
                        position: fixed;
                        bottom: 0;
                        left: 2;
                        width: 100%;
                        padding: 1rem;
                        text-align: left;
                        color: #00FF00;
                    }
                </style>
                """, unsafe_allow_html=True)
            
        # Create a container at the bottom of the sidebar
        footer_container = st.container()
        
        # Fill the space between content and footer
        st.markdown("<div style='flex: 1'></div>", unsafe_allow_html=True)
        
        # Add the footer with green text
        footer_container.markdown(
            "<p class='footer'>Designed and Developed by Amir Aijaz</p>",
            unsafe_allow_html=True
        )

    # Main page
    st.title("🤖 RAG System")
    st.markdown("---")
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Main content area
    if uploaded_files:
        # Save uploaded files temporarily
        temp_paths = []
        for file in uploaded_files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            temp_paths.append(temp_path)
        
        # Create columns for processing information
        col1, col2 = st.columns(2)
        
        with col1:
            status_placeholder = st.empty()
        
        # Query interface
        st.markdown("---")
        st.subheader("🔍 Query Interface")
        query = st.text_input("Enter your question:")
        
        if query:
            # Create a new placeholder for processing status
            status_area = st.empty()
            with status_area:
                with st.spinner("Processing..."):
                    response, num_chunks = rag.full_process(temp_paths, query, status_placeholder)
                    
                    # Display statistics
                    with col2:
                        st.metric("📊 Documents Processed", len(uploaded_files))
                        st.metric("📄 Total Chunks Created", num_chunks)
                        st.metric("🔢 Embeddings Generated", num_chunks)
                
                # Display response
                st.markdown("---")
                st.subheader("🤖 Response")
                st.write(response)
        
        # Clean up temporary files
        for path in temp_paths:
            os.remove(path)
    
    else:
        st.info("👈 Please upload PDF documents using the sidebar to begin.")
        st.markdown("""
        ### How to use:
        1. Upload PDF documents using the sidebar
        2. Watch real-time processing status
        3. Enter your question in the query interface
        4. View the generated response
        """)

if __name__ == "__main__":
    create_streamlit_ui()
