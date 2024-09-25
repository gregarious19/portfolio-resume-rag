from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def pipeline(question):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    pdf_text = extract_text_from_pdf(pdf_path="./static/resume.pdf")
    text_splits = text_splitter.split_text(pdf_text)

    doc_splits = []

    for i, text in enumerate(text_splits):
        doc_splits.append(Document(page_content=text, metadata={"source": "pdf"}, id=i))

    vectorstore = Chroma.from_documents(
        documents=doc_splits, embedding=OllamaEmbeddings(model="llama3.1")
    )
    pem_file_path = "./static/resume.pem"  # Replace with your actual .pem file path

    llm = ChatOllama(
        host="http://13.71.23.14",
        server_url="http://13.71.23.14:11434",  # Replace with your Azure VM public IP or domain
        server_type="https",  # Depending on how you expose the API, you may want to use HTTPS for secure access
        num_predict=2098,  # Prediction limit (adjust based on model needs)
        model="llama3.1",  # Model being served by Ollama
        cert=pem_file_path,
    )

    rag_prompt = hub.pull("rlm/rag-prompt-llama")

    retriever = vectorstore.as_retriever()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    output = chain.invoke({"context": doc_splits, "question": question})

    return output
