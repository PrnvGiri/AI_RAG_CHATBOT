import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# Load Gemini API key from .env file
load_dotenv()

# PDF document path
PDF_PATH = "RayOptics.pdf"


# Step 1: Load and Split PDF Document
def load_and_split(filepath):
    print(f"Loading PDF from: {filepath}")
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    print("Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    splits = splitter.split_documents(docs)
    print(f"Total Chunks: {len(splits)}")
    return splits


# Step 2: Create RAG Chain
def create_rag_chain(splits):
    print("Embedding document chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        task_type="RETRIEVAL_DOCUMENT"
    )

    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vector_store.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    template = """Answer the question based only on the following Context: {context}
Question: {question}

Helpful Answer: """

    sys_prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | sys_prompt
        | llm
        | StrOutputParser()
    )

    return chain


# Step 3: Run RAG Pipeline and Launch Gradio App
if __name__ == "__main__":
    # Load document and build RAG chain
    splits = load_and_split(PDF_PATH)
    rag_chain = create_rag_chain(splits)

    # Launch Gradio Chat Interface
    print("\nLaunching Chatbot Web UI...")
    
    def response(message, history):
        return rag_chain.invoke(message)

    demo = gr.ChatInterface(
        fn=response,
        textbox=gr.Textbox(placeholder="Ask a question related to Ray Optics...")
    )
    demo.launch(share=True)
