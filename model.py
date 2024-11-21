import streamlit as st 
from PyPDF2 import PdfReader  
import os   
import tempfile
import json  
from datetime import datetime  

from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings  
from langchain.prompts import ChatPromptTemplate

def read_data(files):
    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        try:
            pdf_reader = PdfReader(tmp_file_path)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                documents.append(Document(page_content=text, metadata={"source": file.name, "page_number": page_num + 1}))
        finally:
            os.remove(tmp_file_path)
    return documents

def get_chunks(texts, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for text in texts:
        split_texts = text_splitter.split_text(text.page_content)
        for split_text in split_texts:
            chunks.append(Document(page_content=split_text, metadata=text.metadata))
    return chunks

def vector_store(text_chunks, embedding_model_name, vector_store_path):
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_name)
    vector_store = FAISS.from_texts(texts=[doc.page_content for doc in text_chunks], embedding=embeddings, metadatas=[doc.metadata for doc in text_chunks])
    vector_store.save_local(vector_store_path)

def load_vector_store(embedding_model_name, vector_store_path):
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_name)
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def save_conversation(conversation, vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    with open(conversation_path, "w") as f:
        json.dump(conversation, f, indent=4)

def load_conversation(vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    if os.path.exists(conversation_path):
        with open(conversation_path, "r") as f:
            conversation = json.load(f)
    else:
        conversation = []
    return conversation

def document_to_dict(doc):
    return {
        "metadata": doc.metadata
    }

def get_conversational_chain(retriever, ques, llm_model, system_prompt):
    llm = Ollama(model=llm_model, verbose=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  
    )
    response = qa_chain.invoke({"query": ques})
    return response

def user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt):
    vector_store = load_vector_store(embedding_model_name, vector_store_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": num_docs})
    response = get_conversational_chain(retriever, user_question, llm_model, system_prompt)
    
    conversation = load_conversation(vector_store_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if 'result' in response:
        result = response['result']
        source_documents = response['source_documents'] if 'source_documents' in response else []
        conversation.append({
            "question": user_question, 
            "answer": result, 
            "timestamp": timestamp, 
            "llm_model": llm_model,
            "source_documents": [document_to_dict(doc) for doc in source_documents]
        })
        
        st.success("AI Assistant's Response:")
        st.write(result)
        st.info(f"🤖 Model: {llm_model}")
        
        with st.expander("📚 Source Documents", expanded=False):
            for doc in source_documents:
                metadata = doc.metadata
                st.write(f"📄 **Source:** {metadata.get('source', 'Unknown')}")
                st.write(f"📃 **Page:** {metadata.get('page_number', 'N/A')}")
                st.write(f"ℹ️ **Additional Info:** {metadata}")
                st.markdown("---")
    else:
        conversation.append({"question": user_question, "answer": response, "timestamp": timestamp, "llm_model": llm_model})
        st.error("An error occurred. Please try again.")
    
    save_conversation(conversation, vector_store_path)
    
    with st.expander("💬 Conversation History", expanded=False):
        for entry in sorted(conversation, key=lambda x: x['timestamp'], reverse=True):
            st.write(f"🙋 **Question ({entry['timestamp']}):** {entry['question']}")
            st.write(f"🤖 **Answer:** {entry['answer']}")
            st.write(f"🧠 **Model:** {entry['llm_model']}")
            if 'source_documents' in entry:
                for doc in entry['source_documents']:
                    st.write(f"📚 **Source:** {doc['metadata'].get('source', 'Unknown')}, **Page:** {doc['metadata'].get('page_number', 'N/A')}")
            st.markdown("---")

def main():
    st.set_page_config(page_title="AI Document Assistant", page_icon="🤖", layout="wide")
    st.title("🤖 Chat with PDFs using Llama3")
    
    st.sidebar.image("https://th.bing.com/th/id/OIP.tZ0EH_Yi857WlxDiKDr6nAHaE7?w=255&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7", use_column_width=True)
    
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        value="You are an intelligent AI assistant designed to help users understand and analyze documents. Provide clear, concise, and accurate information based on the context given. If you're unsure about something, say so and suggest ways to find more information."
    )

    user_question = st.text_input("🔍 Ask a question about your documents:", placeholder="E.g., What are the main topics discussed in the uploaded PDFs?")
   
    embedding_model_name = "llama3:instruct"
    llm_model = "llama3:instruct" 
    vector_store_path = st.sidebar.text_input("📁 Vector Store Path:", "../data/vectorstore/my_store")

    chunk_text = True
    chunk_size = 1000 
    chunk_overlap = 200 
    num_docs = 3 

    if user_question:
        with st.spinner("🧠 Thinking..."):
            user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt)

    with st.sidebar:
        st.header("📄 Document Upload")
        data_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type=['pdf'])
        if st.button("Process Documents", type="primary"):
            with st.spinner("🔄 Processing documents..."):
                raw_documents = read_data(data_files)
                if chunk_text:
                    text_chunks = get_chunks(raw_documents, chunk_size, chunk_overlap)
                else:
                    text_chunks = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_documents]
                vector_store(text_chunks, embedding_model_name, vector_store_path)
                st.success("✅ Documents processed successfully!")
    
    st.markdown("---")
    st.markdown("👨‍💻 Developed with ❤️ using Streamlit and LangChain")

if __name__ == "__main__":
    main()