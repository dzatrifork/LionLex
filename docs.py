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
        loader = PyMuPDFLoader("assets/GDPR_CELEX_32016R0679_EN_TXT.pdf")
        print('Loaded PDF file...')
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        pages = loader.load_and_split(text_splitter)
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


def createDifferenceChain():
    # Configure system prompt
    # TODO: Convert overenskomst documents to markdown and embed in vector store with metadata regarding the chapter/section title.
    # TODO: Create a chain that can find the difference between two documents. Using SelfQueryRetriever to fetch correct documents for each version.
    # TODO: Make user able to ask questions about specific topics they want to know the differences about. 
    difference_template = """Summarize the differences using the following list of differences in the two texts.
    You should focus 
    ----------------
    {differences}"""

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", temperature=0.9, model="gpt-4-32k")
    chain = (difference_template | llm | BaseLLMOutputParser())

    # metadata_field_info = [
    #     AttributeInfo(
    #         name="page",
    #         description="The number of the page",
    #         type="integer",
    #     ),
    #     AttributeInfo(
    #         name="source",
    #         description="The document source",
    #         type="integer",
    #     ),
    # ]
    # retriever = SelfQueryRetriever.from_llm(
    #     llm,
    #     db,
    #     'Snippets from the EU GDPR Regulations',
    #     metadata_field_info=metadata_field_info,
    # )

    return chain


def createChain(action):
    match action:
        case "difference":
            return createDifferenceChain()
        case "question":
            return createQuestionChain()
        case _:
            return None


@cl.on_chat_start
async def start():
    msg = cl.Message(content="")
    await msg.send()
    if (vectorStoreReady) == False:
        updateMessage(msg, "Loading vector store...")
        await prepareVectorStore()
        updateMessage(msg, "Vector store loaded!")

    chain = createQuestionChain()
    await askUserSelectAction()
    action = cl.user_session.get("action")

    chain = createChain(action)

    # Store the chain in the user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def message(clMessage: cl.Message):
    message = clMessage.content
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")
    action = cl.user_session.get("action")

    if action == "difference":
        await askUserSelectAction()
        return

    if action == "question":
        # TODO Enable the user to ask more questions and retain the context from the previous questions
        # TODO Handle cases where the user asks a question that should not look at sources. E.g. "Test" or "What is a Banana pancake?". Maybe use higher threshold for retrieval?
        # TODO Play around with different parameters (temp, chunk_size, chunk_overlap, k, etc.) to get better results
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
                source_elements.append(
                    cl.Text(content=source.page_content, name=str(source.metadata["page"])))
                message += (f"\n Page {str(source.metadata['page'])}, ")
        else:
            answer += "\nNo sources found"
        await cl.Message(content=message, elements=source_elements).send()
