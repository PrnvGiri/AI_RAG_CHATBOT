import os
import sys

# Try importing dotenv for automatic .env loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# ==========================================
# 1. CONFIGURATION & API KEY SETUP
# ==========================================
DEFAULT_PDF_PATH = "RayOptics.pdf"

def setup_api_key():
    """Ensures a valid Google Gemini API Key is set in environment or loaded from .env."""
    # Check .env file directly if not loaded yet
    if not os.getenv("GOOGLE_API_KEY") and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("GOOGLE_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    if key:
                        os.environ["GOOGLE_API_KEY"] = key
                        break

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    placeholder_keys = ["YOUR_GEMINI_API_KEY_HERE", "YOUR_GOOGLE_API_KEY", ""]
    
    if api_key in placeholder_keys:
        print("\n=======================================================")
        print(" [!] GOOGLE_API_KEY environment variable is missing.")
        print(" Get a free API key at: https://aistudio.google.com/")
        print("=======================================================")
        api_key = input("Enter your Google Gemini API Key: ").strip()
        if not api_key:
            print("Error: A valid API Key is required to run the chatbot.")
            sys.exit(1)
    
    os.environ["GOOGLE_API_KEY"] = api_key


# ==========================================
# 2. LOAD AND SPLIT PDF DOCUMENT
# ==========================================
def load_and_split(filepath=DEFAULT_PDF_PATH):
    """Loads PDF file and splits text into chunks."""
    print(f"Loading PDF document from: {filepath}")
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    print("Splitting data into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    splits = splitter.split_documents(docs)
    print(f"Total Chunks Created: {len(splits)}")
    return splits


# ==========================================
# 3. CREATE RAG CHAIN
# ==========================================
def create_rag_chain(splits):
    """Embeds document chunks into Chroma Vector Store and builds the RAG chain."""
    print("Generating embeddings and storing vectors...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        task_type="RETRIEVAL_DOCUMENT"
    )

    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vector_store.as_retriever()

    # Initialize Gemini Chat Model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    # Prompt Template
    template = """Answer the question based only on the following Context: {context}
Question: {question}

Helpful Answer: """

    sys_prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build RAG Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | sys_prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    setup_api_key()

    prompt_msg = f"Enter PDF File Path (press Enter for default '{DEFAULT_PDF_PATH}'): "
    pdf_path = input(prompt_msg).strip()

    if not pdf_path:
        pdf_path = DEFAULT_PDF_PATH

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
    else:
        # Load PDF and build RAG chain
        splits = load_and_split(pdf_path)
        rag_chain = create_rag_chain(splits)

        # Ask user if they want Web UI or CLI
        choice = input("\nDo you want to open Web UI? (y/n): ").strip().lower()

        if choice == "y":
            print("\nLaunching Gradio Web UI...")
            
            def chat_fn(message, history):
                return rag_chain.invoke(message)

            demo = gr.ChatInterface(
                fn=chat_fn,
                textbox=gr.Textbox(placeholder="Ask a question about Ray Optics or your PDF...")
            )
            demo.launch(share=True)
        else:
            print("\n--- RAG Chatbot CLI ---")
            while True:
                question = input("\nAsk Question (or type 'exit'): ").strip()
                if question.lower() == "exit":
                    break
                answer = rag_chain.invoke(question)
                print(f"\nAnswer:\n{answer}")
