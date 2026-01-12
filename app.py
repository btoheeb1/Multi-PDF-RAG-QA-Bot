import os
import streamlit as st
import shutil

from rag_utility import build_multi_pdf_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(working_dir, "uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("📄 Multi-PDF RAG Q&A")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs"):
    if not uploaded_files:
        st.warning("Upload at least one PDF.")
    else:
        pdf_paths = []
        for f in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            pdf_paths.append(save_path)

        collection_name, persist_dir = build_multi_pdf_chroma_db(pdf_paths)
        st.session_state["collection_name"] = collection_name
        st.session_state["persist_dir"] = persist_dir

        st.success("PDFs processed!")

st.write("---")

user_question = st.text_area("Ask a question across all uploaded PDFs")

if st.button("Answer"):
    if "collection_name" not in st.session_state:
        st.warning("Please upload and process PDFs first.")
    elif not user_question.strip():
        st.warning("Please type a question.")
    else:
        answer, sources = answer_question(
            user_question,
            collection_name=st.session_state["collection_name"],
            persist_directory=st.session_state["persist_dir"],
        )

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        if sources:
            for s in sources:
                st.write(f"- {s}")
        else:
            st.write("No sources returned.")
