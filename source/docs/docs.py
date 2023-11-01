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


if os.path.exists("../../assets/data/chroma"):
    # Load the vector store from disk
    print('Creating vector store from existing data...')
    db = Chroma(persist_directory='../../assets/data/chroma', embedding_function=OpenAIEmbeddings())
else:    
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    print('Creating vector store...')
    loader = PyPDFLoader("../../assets/GDPR_CELEX_32016R0679_EN_TXT.pdf")
    print('Loaded PDF file...')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    pages = loader.load_and_split(text_splitter)
    print('Finished spliting text...')
    db = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory='../../assets/data/chroma')
print('Finished creating vector store!')

query = "What constitutes a personal data breach under the GDPR?"
docs = db.similarity_search(query)
print(docs[0].page_content + ' METADATA: ' + str(docs[0].metadata))

retriever = db.as_retriever()
