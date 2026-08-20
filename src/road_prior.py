import numpy as np
import matplotlib.pyplot as plt

# LOAD IMAGE
img = np.load("../data/sample/img_0010.npy")

# RGB CHANNELS
blue = img[0]
green = img[1]
red = img[2]

# CREATE RGB
rgb = np.stack(
    [red, green, blue],
    axis=-1
)

rgb = np.clip(rgb, 0, 1)

# BRIGHTNESS MAP
brightness = np.mean(rgb, axis=-1)

# LOAD NDVI
from ndvi import compute_ndvi

ndvi = compute_ndvi(img)

# ROAD PRIOR
# higher for:
# low NDVI
# brighter terrain

road_prior = (
    0.7 * brightness
    +
    0.3 * (1 - ((ndvi + 1) / 2))
)

# NORMALIZE
road_prior = (
    road_prior - road_prior.min()
) / (
    road_prior.max() - road_prior.min()
)

# SAVE
np.save(
    "../outputs/road_prior.npy",
    road_prior
)

print("Road prior saved!")

# VISUALIZE
plt.figure(figsize=(8,8))

plt.imshow(
    road_prior,
    cmap="gray"
)

plt.title("Road / Accessibility Prior")

plt.colorbar()

plt.axis("off")

plt.show()