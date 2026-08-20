import numpy as np
import matplotlib.pyplot as plt

img = np.load("../data/sample/img_0010.npy")
mask = np.load("../data/sample/mask_0010.npy")

print("=" * 50)
print("IMAGE INFO")
print("=" * 50)
print("Shape:", img.shape)
print("Datatype:", img.dtype)
print("Min:", img.min())
print("Max:", img.max())

print()

print("=" * 50)
print("MASK INFO")
print("=" * 50)
print("Shape:", mask.shape)
print("Datatype:", mask.dtype)
print("Min:", mask.min())
print("Max:", mask.max())

print()

# CHECK CHANNEL FORMAT
if img.shape[0] == 4:
    print("Detected format: (C, H, W)")

    red = img[2]
    green = img[1]
    blue = img[0]

    rgb = np.stack([red, green, blue], axis=-1)

else:
    print("Detected format: (H, W, C)")

    rgb = img[:, :, :3]

# normalize
rgb = rgb.astype(np.float32)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.show()