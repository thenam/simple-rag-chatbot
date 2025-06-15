import streamlit as st
import os
import tempfile
import torch
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer, pipeline
from huggingface_hub import snapshot_download

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
                model_name="bkai-foundation-models/vietnamese-bi-encoder"
            )

@st.cache_resource
def load_llm():
    #MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    #MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

    local_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "Qwen2-1.5B-Instruct")

    #Check da ton tai model trong cache chua?
    if not os.path.exists(local_dir):
        local_dir = snapshot_download(
            repo_id=MODEL_NAME, 
            local_dir=local_dir, 
            local_dir_use_symlinks=False
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        quantization_config=quantization_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    return HuggingFacePipeline(pipeline=model_pipeline)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
        min_chunk_size=300,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()
        
    prompt = PromptTemplate.from_template(
        "You are an intelligent assistant. Your task is to answer questions based on the **provided context**."
        "---"
        "**Rules:**"
        "* Only answer based on the information available in the context."
        "* If the information is not found in the context, respond with: Không tìm thấy thông tin trong ngữ cảnh được cung cấp."
        "* Answer **concisely, directly**, and **focus solely** on the question."
        "* Do not add any explanations, inferences, or information outside the context."
        "* Ensure the answer has a clear structure and is relevant to the core of the question."
        "* Use Vietnamese."
        "---"
        "**Context:**"
        "{context}"
        "---"
        "**Question:**"
        "{question}"
        "---"
        "**Answer:**"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    os.unlink(tmp_file_path)

    return rag_chain, len(docs)


st.set_page_config(page_title="Simple RAG Chatbot", layout="wide")
st.markdown("""
**Ứng dụng Chatbot hỏi đáp với PDF**
**Upload PDF và đặt câu hỏi**
""")

if not st.session_state.model_loaded:
    st.info("Đang tải models...")
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm()
    st.session_state.model_loaded = True
    st.success("Models đã sẵn sàng!!!")
    st.rerun()


uploaded_file = st.file_uploader("Upload pdf file", type="pdf")
if uploaded_file and st.button("Xử lý tệp tin"):
    with st.spinner("Đang xử lý..."):
        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
        st.success(f"Hoàn thành xử lý file, có {num_chunks} chunks")

if st.session_state.rag_chain:
    question = st.text_input("Đặt câu hỏi!")
    if question:
        with st.spinner("Đang tìm câu trả lời..."):
            output = st.session_state.rag_chain.invoke(question)
            print(output)
            answer = output.split("Trả lời:")[1].strip() if "Trả lời:" in output else output.strip()

            st.write("**Trả lời:**")
            st.write(answer)
