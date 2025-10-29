from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

import bs4
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

import os

DEBUG = True

model = ChatOllama(
    model="qwen3:1.7b",
    validate_model_on_init=True,
    temperature=0.6,
    num_predict=4096,
    # other params ...
)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = Chroma(
    collection_name="webpage_collection",
    embedding_function=embeddings,
    # persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

web_path = input(
    "Enter the URL of the Wikipedia page that you want to ask questions about: \n"
)

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=(web_path,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(name="div", attrs={"id": "mw-content-text"})
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    add_start_index=True,
    separators=[
        "\n\n",
        "\n",
        " ",
        "",
    ],  # Standard separators    is_separator_regex=True,
)
all_splits = text_splitter.split_documents(docs)

if DEBUG:
    # if exist, remove docsraw.txt
    if os.path.exists("docsraw.txt"):
        os.remove("docsraw.txt")

    with open("docsraw.txt", "w") as f:
        for doc in docs:
            f.write(doc.page_content + f"\n{100*'*'}\n")

    # if exist, remove docs.txt
    if os.path.exists("docs.txt"):
        os.remove("docs.txt")

    with open("docs.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_splits):
            start = chunk.metadata.get("start_index", "unknown")
            length = len(chunk.page_content)
            f.write(f"CHUNK {i} | START: {start} | LEN: {length}\n")
            f.write(chunk.page_content.strip())
            f.write(f"\n{'*'*100}\n\n")

# Index chunks
_ = vector_store.add_documents(documents=all_splits)


# Construct a tool for retrieving context
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=10)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You are Webby, an expert in answering question given a Wikipedia page URL. "
    "You have access to a tool that retrieves context from the Wikipedia page. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

query = ""

while query != "exit":
    query = input("Your prompt: ")

    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        # print(step)
        step["messages"][-1].pretty_print()
