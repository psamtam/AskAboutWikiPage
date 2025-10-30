# Introduction
This is a personal project aimed at learning and showcasing the use of **Retrieval-Augmented Generation (RAG)** and **webpage scraping** — all running **locally**.

The project runs on an Ubuntu (WSL2) environment with CUDA enabled. Both the **LLM** and the **vector database** operate completely offline, ensuring a privacy-friendly and self-contained workflow.

# How It Works
1. Scrape a Wikipedia page using **BeautifulSoup** and **LangChain loaders**
2. Split and embed text locally using **`nomic-embed-text:latest`**
3. Store embeddings in a **Chroma** vector database
4. Use **Gwen3** (via **Ollama**) as the local LLM for generation
5. Ask natural language questions about the scraped content

# Environment
- Python: 3.10.19  
- For all other dependencies, please see [`requirements.txt`](./requirements.txt).

# Stack Overview
- **LangChain** (for RAG pipeline)  
- **LangChain Ollama** & **LangChain Chroma**  
- **Ollama** for running local models (**Gwen3**)  
- **BeautifulSoup** for web scraping  
- **Chroma** for vector storage  
- **nomic-embed-text:latest** for text embeddings  

# Demo
![](https://github.com/psamtam/AskAboutWikiPage/blob/main/Demo.gif)

# Repository
[https://github.com/psamtam/AskAboutWikiPage](https://github.com/psamtam/AskAboutWikiPage)
