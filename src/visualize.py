import numpy as np
import matplotlib.pyplot as plt

# LOAD ORIGINAL IMAGE
img = np.load("../data/sample/img_0010.npy")

# EXTRACT RGB
blue = img[0]
green = img[1]
red = img[2]

rgb = np.stack([red, green, blue], axis=-1)

rgb = rgb.astype(np.float32)
rgb = np.clip(rgb, 0, 1)

# LOAD PATH
path = np.load("../outputs/path.npy")

# LOAD COST SURFACE
cost_surface = np.load("../outputs/cost_surface.npy")

# START AND GOAL
start = tuple(path[0])
goal = tuple(path[-1])

# PLOT
plt.figure(figsize=(10,10))

# SHOW RGB
plt.imshow(rgb)


# PATH
path_x = [p[1] for p in path]
path_y = [p[0] for p in path]

plt.plot(
    path_x,
    path_y,
    color="cyan",
    linewidth=3,
    label="Eco Corridor"
)

# START POINT
plt.scatter(
    start[1],
    start[0],
    color="lime",
    s=150,
    edgecolors="black",
    label="Start"
)

# GOAL POINT
plt.scatter(
    goal[1],
    goal[0],
    color="red",
    s=150,
    edgecolors="black",
    label="Goal"
)

plt.title("Eco-Aware Corridor Overlay")

plt.legend()

plt.axis("off")

plt.tight_layout()

plt.show()