import io
import os
from dotenv import dotenv_values, load_dotenv 
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override = True)
    config = dotenv_values(".env")

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loader = PyPDFLoader("../../assets/GDPR_CELEX_32016R0679_EN_TXT.pdf")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
pages = loader.load_and_split(text_splitter)
db = Chroma.from_documents(pages, OpenAIEmbeddings())

query = "What constitutes a personal data breach under the GDPR?"
docs = db.similarity_search(query)
print(docs[0].page_content)
