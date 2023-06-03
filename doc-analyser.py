import os
import openai
import streamlit as st
import pypdf
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAIChat
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def split_file(fpath, chunk_chars=4000, overlap=50):
    """
    Pre-process PDF into chunks
    Some code from: https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py
    """
    st.info("`Reading and splitting doc ...`")
    filename = uploaded_file.name
    temp_path = os.path.join("./Temp", filename)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if fpath.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    if fpath.name.endswith(".doc") or fpath.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)
    if fpath.name.endswith(".txt"):
        loader = TextLoader(temp_path)
    if fpath.name.endswith(".ppt") or fpath.name.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(temp_path)

    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    splits = text_splitter.split_documents(document)
    return splits

@st.cache_resource
def create_ix(_splits):
    """ 
    Create vector DB index of file
    """
    st.info("`Building index ...`")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(_splits, embeddings)

# Auth
st.sidebar.image("Img/reading.jpg")
st.sidebar.write("'By: Claude Feldges (modified version of Lance Martin doc-gpt)'")
chunk_chars = st.sidebar.radio("`Choose chunk size for splitting`", (2000, 3000, 4000), index=1)
st.sidebar.info("`Larger chunk size can produce better answers, but may hit ChatGPT context limit (4096 tokens)`")

# App 
st.header("`Document Analyser`")
st.info("`Hello! I am a Chat GPT connected to whatever document (pdf, word, power point, txt) you upload.`")
uploaded_file = st.file_uploader("`Upload File:` ", type = ['pdf','doc','docx','ppt','pptx','txt'], accept_multiple_files=False)
if uploaded_file:
    # Split and create index
    d = split_file(uploaded_file, chunk_chars)
    if d:
        ix = create_ix(d)
        # Use ChatGPT with index QA chain
        llm = OpenAIChat(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0)
        # chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=ix.as_retriever())
        chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=ix.as_retriever())
        query = st.text_input("`Please ask a question:` ", "What is this document about?")
        try:
            st.info(f"`{chain.run(query)}`")
            #st.header("Answer")
            #st.info(f"'{chain({'question':query}, return_only_outputs=True)['answer']}`")
            #st.header("Source")
            #st.info(f"'{chain({'question': query}, return_only_outputs=True)['sources']}`")
        except openai.error.InvalidRequestError:
            # Limitation w/ ChatGPT: 4096 token context length
            # https://github.com/acheong08/ChatGPT/discussions/649
            st.warning('Error with model request, often due to context length. Try reducing chunk size.', icon="⚠️")
    else:
        st.warning('Error with reading pdf, often b/c it is a scanned image of text. Try another file.', icon="⚠️")

else:
    st.info("`There is an issue with the OpenAI key.`")
