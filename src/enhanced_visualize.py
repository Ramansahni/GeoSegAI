import numpy as np
import matplotlib.pyplot as plt

# LOAD IMAGE
img = np.load(
    "../data/sample/img_0001.npy"
)

# RGB
blue = img[0]
green = img[1]
red = img[2]

rgb = np.stack(
    [red, green, blue],
    axis=-1
)

rgb = np.clip(rgb, 0, 1)

# LOAD ENHANCED PATH
path = np.load(
    "../outputs/enhanced_path.npy"
)

# COORDS
path_x = [p[1] for p in path]
path_y = [p[0] for p in path]

# START/GOAL
start = path[0]
goal = path[-1]

# VISUALIZE
plt.figure(figsize=(10,10))

plt.imshow(rgb)

# PATH
plt.plot(
    path_x,
    path_y,
    color="deepskyblue",
    linewidth=4,
    label="Enhanced Eco Corridor"
)

# START
plt.scatter(
    start[1],
    start[0],
    color="lime",
    s=180,
    edgecolors="black"
)

# GOAL
plt.scatter(
    goal[1],
    goal[0],
    color="red",
    s=180,
    edgecolors="black"
)

plt.title(
    "Infrastructure-Aware Corridor"
)

plt.legend()

plt.axis("off")

plt.tight_layout()

plt.show()