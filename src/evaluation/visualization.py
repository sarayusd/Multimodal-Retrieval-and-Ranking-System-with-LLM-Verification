
import matplotlib.pyplot as plt
from PIL import Image


def safe_open_image(path):

    return Image.open(path).convert("RGB")


def show_image_results(results):

    n = len(results)

    plt.figure(figsize=(14, 4 * n))

    for i, r in enumerate(results):

        img = safe_open_image(r["image_path"])

        plt.subplot(n, 1, i + 1)

        plt.imshow(img)

        plt.axis("off")

        plt.title(r["caption"])

    plt.show()