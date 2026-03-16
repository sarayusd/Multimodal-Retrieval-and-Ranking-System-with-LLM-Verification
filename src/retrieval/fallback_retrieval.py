
def retrieve_with_fallback(query, dense_fn, hybrid_fn, rerank_fn, bm25_fn):

    try:
        results = rerank_fn(query)

        if len(results) > 0:
            return results

    except Exception:
        pass

    try:
        results = hybrid_fn(query)

        if len(results) > 0:
            return results

    except Exception:
        pass

    try:
        results = dense_fn(query)

        if len(results) > 0:
            return results

    except Exception:
        pass

    return bm25_fn(query)