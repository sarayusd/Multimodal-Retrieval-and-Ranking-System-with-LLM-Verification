import numpy as np


def cosine_topk(query_vec, matrix, k=5):

    sims = matrix @ query_vec.T

    idx = np.argsort(-sims.squeeze())[:k]

    return idx, sims[idx]


def retrieve_text_to_image_plain(query_vec, img_embeddings, image_paths, k=5):

    idx, scores = cosine_topk(query_vec, img_embeddings, k)

    results = []

    for i, s in zip(idx, scores):

        results.append({
            "image_path": image_paths[i],
            "score": float(s)
        })

    return results


def retrieve_image_to_image(query_vec, img_embeddings, image_paths, k=5):

    idx, scores = cosine_topk(query_vec, img_embeddings, k)

    return [
        {"image_path": image_paths[i], "score": float(scores[j])}
        for j, i in enumerate(idx)
    ]