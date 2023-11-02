import asyncio
import os
from dotenv import dotenv_values, load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import chainlit as cl

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

vectorStoreReady = False
async def prepareVectorStore():
    if os.path.exists("../../assets/data/chroma"):
        # Load the vector store from disk
        print('Creating vector store from existing data...')
        db = Chroma(persist_directory='../../assets/data/chroma',
                    embedding_function=OpenAIEmbeddings())
    else:
        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        print('Creating vector store...')
        loader = PyMuPDFLoader("../../assets/GDPR_CELEX_32016R0679_EN_TXT.pdf")
        print('Loaded PDF file...')
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        pages = loader.load_and_split(text_splitter)
        print('Finished splitting text...')
        db = Chroma.from_documents(pages, OpenAIEmbeddings(
        ), persist_directory='../../assets/data/chroma')
    print('Finished creating vector store!')
    return [db, True]

[db, vectorStoreReady] = asyncio.run(prepareVectorStore())


def updateMessage(msg, content):
    msg.content = content
    msg.update()

async def askUserSelectAction():
    res = await cl.AskActionMessage(
        content="Pick what you want to do!",
        actions=[
            cl.Action(name="question", value="question",
                      label="❓ Ask a question about GDPR law"),
            cl.Action(name="difference", value="difference",
                      label="🤔 What's the difference between the 2020 and 2023 versions of Finansforbundet Standardoverenskomst?"),
        ]
    ).send()

    if (res == None):
        await askUserSelectAction()
    cl.user_session.set("action", res.get("value"))


@cl.on_chat_start
async def start():
    msg = cl.Message(content="")
    await msg.send()
    if (vectorStoreReady) == False:
        updateMessage(msg, "Loading vector store...")
        await prepareVectorStore()
        updateMessage(msg, "Vector store loaded!")

    # Configure system prompt
    system_template = """Use the following pieces of context to answer the users question.
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

    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-16k")
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    await askUserSelectAction()

    # Store the chain in the user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def message(clMessage: cl.Message):
    message = clMessage.content
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")
    action = cl.user_session.get("action")

    if action == "difference":
        await cl.Message(content="Not implemented yet!").send()
        await askUserSelectAction()
        return

    if action == "question":
        print("Question asked: " + message)
        response = await chain.acall(message)
        answer = response["answer"]
        source_documents = response["source_documents"]
        pages = 'Pages: '

        message = f"Answer: {answer}\n\nSources:"
        source_elements = []
        print("Answer: " + answer)
        if source_documents:
            sorted_source_documents = [x for _, x in sorted(
                zip([doc.metadata["page"] for doc in source_documents], source_documents))]

            # Add the sources to the message
            for source in sorted_source_documents:
                source_elements.append(
                    cl.Text(content=source.page_content, name=str(source.metadata["page"])))
                message += (f"\n Page {str(source.metadata['page'])}, ")
        else:
            answer += "\nNo sources found"
        await cl.Message(content=message, elements=source_elements).send()
