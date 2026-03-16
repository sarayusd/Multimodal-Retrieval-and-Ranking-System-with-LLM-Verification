import numpy as np


def compute_recall_at_k(gt, preds, k):

    hits = sum([1 for g, p in zip(gt, preds) if g in p[:k]])

    return hits / len(gt)


def compute_precision_at_k(gt, preds, k):

    scores = []

    for g, p in zip(gt, preds):
        scores.append(1 if g in p[:k] else 0)

    return np.mean(scores)


def compute_mrr(gt, preds):

    rr = []

    for g, p in zip(gt, preds):

        if g in p:
            rr.append(1 / (p.index(g) + 1))
        else:
            rr.append(0)

    return np.mean(rr)