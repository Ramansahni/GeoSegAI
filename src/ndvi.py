import numpy as np
import matplotlib.pyplot as plt

def compute_ndvi(img):

    red = img[2].astype(np.float32)
    nir = img[3].astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-8)

    ndvi = np.clip(ndvi, -1, 1)

    return ndvi


if __name__ == "__main__":

    img = np.load("../data/sample/img_0010.npy")

    ndvi = compute_ndvi(img)

    print("NDVI Shape:", ndvi.shape)
    print("NDVI Min:", ndvi.min())
    print("NDVI Max:", ndvi.max())

    plt.figure(figsize=(6,6))

    plt.imshow(ndvi, cmap="RdYlGn")
    plt.colorbar(label="NDVI")

    plt.title("NDVI Map")
    plt.axis("off")

    plt.show()