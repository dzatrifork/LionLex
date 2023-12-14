import asyncio
import os

import chainlit as cl
from dotenv import dotenv_values, load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseLLMOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

vectorStoreReady = False

def get_document_name(document):
    if document == 'doc1.pdf':
        return 'Technical Description: MULTICAL® 403'
    if document == 'doc2.pdf':
        return 'READy Solution description:  Heat/Cooling'
    if document == 'doc3.pdf':
        return 'Installation guide - Kamstrup Wireless M-Bus radio network'

async def prepareVectorStore():
    ai_embeddings = AzureOpenAIEmbeddings(
        deployment="embedding",
        model="text-embedding-ada-002")
    if os.path.exists("assets/data/chroma"):
        # Load the vector store from disk
        print('Creating vector store from existing data...')
        db = Chroma(persist_directory='assets/data/chroma',
                    embedding_function=ai_embeddings)
    else:
        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        print('Creating vector store...')
        pages = []
        for file in os.listdir("assets/documents"):
            loader = PyMuPDFLoader('assets/documents/' + file)
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
            split_texts = loader.load_and_split(text_splitter)
            for page in split_texts:
                page.metadata['document'] = get_document_name(file)

            pages += split_texts
            print('Finished splitting text...')
        db = Chroma.from_documents(
            pages,
            ai_embeddings,
            persist_directory='assets/data/chroma')
    print('Finished creating vector store!')
    return [db, True]


[db, vectorStoreReady] = asyncio.run(prepareVectorStore())


def updateMessage(msg, content):
    msg.content = content
    msg.update()


def createQuestionChain():
    # Configure system prompt
    system_template = """You're a helpful assistant who answers questions about GDPR
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    The "SOURCES" part should be a reference to the source of the document from which you got your answer.

    ----------------
    {summaries}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", temperature=0.9, model="gpt-4-32k")

    # TODO: Play with retriever parameters (https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa#vectorstore-retriever-options)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    return chain

@cl.on_chat_start
async def start():
    document_titles = [get_document_name(doc) for doc in os.listdir("assets/documents")]
    msg = cl.Message(content="Welcome to the GDPR chatbot!\n\nYou can ask me questions about the following documents:\n" + "\n".join(document_titles) + "\n\nPlease ask me a question about one of these documents.")
    await msg.send()
    if (vectorStoreReady) == False:
        updateMessage(msg, "Loading vector store...")
        await prepareVectorStore()
        updateMessage(msg, "Vector store loaded!")

    chain = createQuestionChain()

    # Store the chain in the user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def message(clMessage: cl.Message):
    message = clMessage.content
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")
    print("Question asked: " + message)
    response = await chain.acall(message)
    answer = response["answer"]
    source_documents = response["source_documents"]

    message = f"Answer: {answer}\n\nSources:"
    source_elements = []
    print("Answer: " + answer)
    if source_documents:
        sorted_source_documents = [x for _, x in sorted(
            zip([doc.metadata["page"] for doc in source_documents], source_documents))]

        # Add the sources to the message
        for source in sorted_source_documents:
            name = f"{str(source.metadata['document'])}, Page {str(source.metadata['page'])}"
            source_elements.append(
                cl.Text(content=source.page_content, name=name))
            message += (f"\n Document: {name}, ")
    else:
        answer += "\nNo sources found"
    await cl.Message(content=message, elements=source_elements).send()
