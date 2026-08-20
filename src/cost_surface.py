import numpy as np
import matplotlib.pyplot as plt

from ndvi import compute_ndvi

# LOAD INPUT IMAGE
img = np.load("../data/sample/img_0001.npy")

# LOAD SEGMENTATION OUTPUT
segmentation = np.load("../outputs/probability_map.npy")

# COMPUTE NDVI
ndvi = compute_ndvi(img)

# NORMALIZE NDVI TO 0-1
ndvi_norm = (ndvi + 1) / 2

# WEIGHTS
w1 = 0.5
w2 = 0.5

# COST SURFACE
cost_surface = (
    w1 * ndvi_norm +
    w2 * segmentation
)

# NORMALIZE FINAL COST
cost_surface = (
    cost_surface - cost_surface.min()
) / (
    cost_surface.max() - cost_surface.min()
)

print("Cost Surface Shape:", cost_surface.shape)

# SAVE
np.save("../outputs/cost_surface.npy", cost_surface)

print("Cost surface saved!")

# VISUALIZE
plt.figure(figsize=(7,7))

plt.imshow(cost_surface, cmap="inferno")
plt.colorbar(label="Traversal Cost")

plt.title("Environmental Cost Surface")
plt.axis("off")

plt.show()