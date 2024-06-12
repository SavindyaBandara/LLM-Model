import os
import fitz  # PyMuPDF
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.docstore.document import Document
import faiss 

# Step 1: Read PDF Files
def read_pdf_files(directory):
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            pdf_texts.append(extract_text_from_pdf(pdf_path))
    return pdf_texts

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Process Content
def preprocess_text(text):
    # Add any text preprocessing steps if necessary
    return text

# Step 3: Implement RAG
def create_knowledge_base(texts):
    # Convert texts to embeddings and store in a vector database
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Get the embedding dimension
    sample_embedding = embeddings.embed_query("sample text")
    embed_dimension = len(sample_embedding)
    
    # Initialize the FAISS index and index_to_docstore_id
    index = faiss.IndexFlatL2(embed_dimension)  # Create a flat (CPU) index
    index_to_docstore_id = {}
    
    # Create the FAISS instance
    vector_db = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id=index_to_docstore_id)
    
    # Add texts to the vector database
    documents = []
    for i, text in enumerate(texts):
        doc = Document(page_content=text)
        doc_id = str(i)
        documents.append(doc)
        index_to_docstore_id[i] = doc_id

    vector_db.add_documents(documents)
    
    return vector_db

def retrieve_and_generate_answer(question, knowledge_base):
    retriever = knowledge_base.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)
    # Use a language model to generate the answer
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    answers = [qa_pipeline(question=question, context=doc.page_content) for doc in relevant_docs]
    return answers

# Step 4: Chatbot Interface
def chatbot_interface(knowledge_base):
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit"]:
            break
        answers = retrieve_and_generate_answer(question, knowledge_base)
        for answer in answers:
            print(f"Answer: {answer['answer']}")

# Main Execution
if __name__ == "__main__":
    directory = r"C:\Users\Asus\Documents\CV"
    pdf_texts = read_pdf_files(directory)
    processed_texts = [preprocess_text(text) for text in pdf_texts]
    knowledge_base = create_knowledge_base(processed_texts)
    chatbot_interface(knowledge_base)
