def query_vector_db(query_vector, index):
    return index.query(
        vector = query_vector,
        top_k = 3,
        include_vectors = False,
        include_metadata = False
    )