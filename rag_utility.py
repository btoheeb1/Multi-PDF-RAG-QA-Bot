import os
import uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# RetrievalQA import differs by langchain version
try:
    from langchain.chains import RetrievalQA
except Exception:
    from langchain_classic.chains import RetrievalQA

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

# Models
embedding = HuggingFaceEmbeddings()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

PERSIST_DIR = os.path.join(working_dir, "doc_vectorstore")


def build_multi_pdf_chroma_db(pdf_paths):
    """
    Builds a Chroma vector DB from multiple PDFs.
    Returns: (collection_name, persist_directory)
    """
    if not pdf_paths:
        raise ValueError("No PDFs provided.")

    collection_name = f"multi_pdf_{uuid.uuid4().hex[:8]}"

    # Load + attach source filename metadata
    all_docs = []
    for path in pdf_paths:
        filename = os.path.basename(path)
        loader = UnstructuredFileLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = filename
        all_docs.extend(docs)

    # Chunk
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Store in Chroma (persistent). Use the new collection name each time.
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )

    return collection_name, PERSIST_DIR


def answer_question(user_question, collection_name, persist_directory=PERSIST_DIR, k=4):
    """
    Returns: (answer_text, list_of_source_filenames)
    """
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name=collection_name,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response.get("result", "")
    docs = response.get("source_documents", [])

    # Collect unique sources
    sources = []
    seen = set()
    for d in docs:
        src = (d.metadata or {}).get("source", "Unknown")
        if src not in seen:
            sources.append(src)
            seen.add(src)

    return answer, sources
