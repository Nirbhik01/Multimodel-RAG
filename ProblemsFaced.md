
## Ingestion and Retrieval
    1. During Data Ingestion, the weighted average of the text embedding and image embedding was stored in the vector database.
    2. This caused data loss of the embeddings which resulted in poor retrieval performance.
    3. To overcome this, we stored both the text embedding and image embedding separately in the vector database. 