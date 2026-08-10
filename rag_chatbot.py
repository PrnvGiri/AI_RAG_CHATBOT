import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# ==========================================
# 1. SET YOUR API KEY
# ==========================================
# Set your Google Gemini API key here or via environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ==========================================
# 2. LOAD AND SPLIT PDF DOCUMENT
# ==========================================
def load_and_split(filepath):
    """Loads PDF file and splits text into chunks."""
    print("Loading PDF document...")
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
    pdf_path = input("Enter PDF File Path: ").strip()

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
                textbox=gr.Textbox(placeholder="Ask a question about your PDF...")
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
