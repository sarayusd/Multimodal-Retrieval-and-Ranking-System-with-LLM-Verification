
from rank_bm25 import BM25Okapi


def bm25_tokenize(text):
    return text.lower().split()


def retrieve_text_to_image_bm25_only(query, captions, image_paths, k=5):

    tokenized = [bm25_tokenize(c) for c in captions]

    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(bm25_tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [
        {
            "image_path": image_paths[i],
            "score": float(scores[i]),
            "caption": captions[i]
        }
        for i in ranked
    ]