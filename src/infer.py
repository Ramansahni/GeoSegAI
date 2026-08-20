import numpy as np
import torch
import matplotlib.pyplot as plt

from model import UNet

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
model = UNet(in_channels=3, out_channels=1)

model.load_state_dict(
    torch.load("../models/eco_rgb_model.pth", map_location=device)
)

model.to(device)
model.eval()

print("Model loaded successfully!")

# LOAD SAMPLE IMAGE
img = np.load("../data/sample/img_0010.npy")

# EXTRACT RGB CHANNELS
blue = img[0]
green = img[1]
red = img[2]

rgb = np.stack([red, green, blue], axis=0)

print("Before normalization:")
print("Min:", rgb.min())
print("Max:", rgb.max())
print("Mean:", rgb.mean())

# NORMALIZE
rgb = rgb.astype(np.float32)
rgb = np.clip(rgb, 0, 1)

# CONVERT TO TENSOR
tensor = torch.tensor(rgb).unsqueeze(0).to(device)

print("Input tensor shape:", tensor.shape)

# INFERENCE
with torch.no_grad():
    pred = model(tensor)

pred = pred.squeeze().cpu().numpy()

print("Prediction shape:", pred.shape)

# THRESHOLD
binary_pred = (pred > 0.5).astype(np.uint8)

np.save("../outputs/probability_map.npy", pred)
np.save("../outputs/binary_mask.npy", binary_pred)

print("Saved outputs to outputs/")

# VISUALIZATION
rgb_vis = np.transpose(rgb, (1,2,0))

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(rgb_vis)
plt.title("RGB Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(pred, cmap="viridis")
plt.title("Probability Map")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(binary_pred, cmap="gray")
plt.title("Binary Prediction")
plt.axis("off")

plt.tight_layout()
plt.show()