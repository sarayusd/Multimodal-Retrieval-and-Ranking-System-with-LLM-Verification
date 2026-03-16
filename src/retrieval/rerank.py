
from sentence_transformers import CrossEncoder


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def reranked_pred_ids(query, candidate_results):

    pairs = [(query, r["caption"]) for r in candidate_results]

    scores = reranker.predict(pairs)

    for r, s in zip(candidate_results, scores):
        r["rerank_score"] = float(s)

    ranked = sorted(candidate_results, key=lambda x: x["rerank_score"], reverse=True)

    return ranked