## Ingestion and Retrieval

    1. During Data Ingestion, the weighted average of the text embedding and image embedding was stored in the vector database.
    2. This caused data loss of the embeddings which resulted in poor retrieval performance.
    3. Storing of both the text embedding and image embedding separately in the vector database seemed to overcome the issue to some extent.

## Score and Ranking

    1. Previously Weighted Average was used on socres of the initially retrieved results form the database and generate an answer`.
    2. Weighted Average caused data loss of the embeddings which resulted in poor retrieval performance.
    3. Now Reciprocal Rank Fusion (RRF) is used to rank the text and corresponding image.

## LLM prompts

    1.
