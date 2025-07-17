# 📚 Project 3: Custom PDF Question Answering Bot
# Goal: Ask questions about your own PDF using embeddings.

# ✅ Learn:

# DocumentLoader, TextSplitter

# FAISS VectorStore

# RetrievalQA


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
)


def create_vector_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db


retriever = create_vector_db("pdf/God.pdf").as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

res = qa.run("is there is a god?")
print(res)
