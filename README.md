# LionLex
App to help describe EU law documentation with Langchain, Azure, Chainlit and ChromaDB 

# Needed software

- Python 3.11.6

# Install deps

```bash
pip install -r .\requirements.txt
```

# Run

Setup `.env` file with the following variables:

````agsl
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_ENDPOINT=https:<your-endpoint>
AZURE_OPENAI_API_KEY=<your-key>
````


```bash 
chainlit run docs.py
```