## Ingestion and Retrieval

    1. During Data Ingestion, the weighted average of the text embedding and image embedding was stored in the vector database.
    2. This caused data loss of the embeddings which resulted in poor retrieval performance.
    3. Storing of both the text embedding and image embedding separately in the vector database seemed to overcome the issue to some extent.

## Score and Ranking

    1. Previously Weighted Average was used on socres of the initially retrieved results form the database and generate an answer`.
    2. Weighted Average caused data loss of the embeddings which resulted in poor retrieval performance.
    3. Now Reciprocal Rank Fusion (RRF) is used to rank the text and corresponding image.

## LLM prompts

    1. Initial prompts were too generic and did not guide the model to reason in a radiological context, resulting in vague or hallucinated responses.
    2. Multiple rounds of prompt engineering were required to improve response quality and relevance.
    3. The system prompt was refined iteratively to enforce structured radiological reasoning, include reference case context, and anonymize case labels to avoid bias during comparison.

## LLM Model Selection

    1. The initial implementation used Gemini as the vision-capable LLM for comparing X-ray images.
    2. Gemini imposed restrictions on medical image uploads, making it unsuitable for chest X-ray inference in this use case.
    3. Switched to a locally hosted Ollama instance running qwen2, which has no upload restrictions and keeps patient image data on-premise.

## Sparse Retrieval

    1. Dense vector retrieval alone (PubMedBERT + Rad-DINO) failed to surface cases that matched on exact clinical terminology, since embeddings can miss rare or specific medical terms.
    2. This led to poor recall for queries containing specific diagnostic terms that were underrepresented in the embedding space.
    3. BM25 sparse retrieval was added as a third retrieval source alongside text and image vector search, with all three result sets fused via RRF.

## Cross-Encoder Reranking

    1. After RRF fusion, the top candidates were ranked by vector similarity scores alone, which did not account for fine-grained semantic relevance between the query and retrieved case text.
    2. This caused cases with superficially similar embeddings but weak clinical relevance to rank too highly.
    3. A cross-encoder reranker was added as a second-stage ranker over the RRF top-10 pool, scoring each (query, cleaned_text) pair before selecting the final top-3 cases.
    4. The initial cross-encoder was a general-purpose model that lacked domain-specific understanding of clinical language, leading to suboptimal reranking on medical text.
    5. Switched to MedCPT-Cross-Encoder (ncbi/MedCPT-Cross-Encoder), a biomedical cross-encoder trained on PubMed data, which improved relevance scoring for radiology reports and clinical findings.

## Query Enhancement

    1. Short or underspecified clinical queries produced weak embedding representations, reducing the quality of vector search results.
    2. This was particularly noticeable when users submitted brief queries without detailed symptom or finding descriptions.
    3. HyDE (Hypothetical Document Embeddings) was integrated to generate a synthetic radiology report from the query before embedding, improving semantic alignment with the indexed case texts.
