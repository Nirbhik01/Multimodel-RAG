def query_vector_db(query_vector, index, vector_type):
    return index.query(
        vector = query_vector,
        top_k = 20,
        include_vectors = False,
        include_metadata = True,
        filter=f"type = '{vector_type}'"  # Upstash metadata filter syntax
    )