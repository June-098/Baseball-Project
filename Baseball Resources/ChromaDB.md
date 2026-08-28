ChromaDB is ==an open-source vector database designed to build AI applications with large language models==. It stores text and vector embeddings to perform fast semantic similarity searches, making it popular for Retrieval-Augmented Generation (RAG) and local AI prototyping. 

### Core Features

- **Embeddings Management**: Automatically handles text embedding generation or accepts custom vectors.
- **Collections**: Groups data into logical tables similar to relational database tables.
- **Metadata Filtering**: Allows filtering search results using associated key-value metadata.
- **Flexible Storage**: Supports running in-memory for quick testing, persisting to local disk, or scaling via serverless cloud.

Quick Start

- **Installation**: Run `pip install chromadb` for Python or `npm install chromadb` for JavaScript.

- **Basic Python Usage**:
    
    python
    
    ```
    import chromadb
    
    client = chromadb.PersistentClient(path="./my_chroma")
    collection = client.get_or_create_collection("my_collection")
    
    collection.add(
        documents=["This is a test document", "Another document about AI"],
        ids=["doc1", "doc2"],
    )
    
    results = collection.query(
        query_texts=["What is this about?"],
        n_results=1,
    )
    ```
    
    Use code with caution.
    
    [[1](https://www.youtube.com/watch?v=qHFcqL11LeQ), [2](https://github.com/chroma-core/chroma)]