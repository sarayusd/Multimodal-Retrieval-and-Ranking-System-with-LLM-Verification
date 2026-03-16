import torch
import numpy as np


def l2_normalize_np(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def encode_images(model, preprocess, image_paths, device):

    image_embeddings = []

    for p in image_paths:
        image = preprocess(p).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image)

        emb = emb.cpu().numpy()
        image_embeddings.append(emb)

    image_embeddings = np.vstack(image_embeddings)

    return l2_normalize_np(image_embeddings)


def encode_texts(model, tokenizer, texts, device):

    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(tokens)

    text_embeddings = text_embeddings.cpu().numpy()

    return l2_normalize_np(text_embeddings)