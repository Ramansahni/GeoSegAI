import heapq
import numpy as np
import matplotlib.pyplot as plt

# LOAD COST SURFACE
cost_surface = np.load("../outputs/cost_surface.npy")

HEIGHT, WIDTH = cost_surface.shape

# START AND END POINTS
clicked_points = []

def onclick(event):

    x = int(event.xdata)
    y = int(event.ydata)

    clicked_points.append((y, x))

    plt.scatter(x, y, color="cyan", s=100)

    plt.draw()

    if len(clicked_points) == 2:
        plt.close()

# DISPLAY COST SURFACE
fig = plt.figure(figsize=(8,8))

plt.imshow(cost_surface, cmap="inferno")

plt.title("Click START then GOAL")

cid = fig.canvas.mpl_connect(
    'button_press_event',
    onclick
)

plt.show()

start = clicked_points[0]
goal = clicked_points[1]

print("Start:", start)
print("Goal:", goal)

# MOVEMENT DIRECTIONS
directions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1)
]

# HEURISTIC FUNCTION
def heuristic(a, b):

    return np.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2
    )

# A* FUNCTION
def astar(cost_map, start, goal):

    open_set = []

    heapq.heappush(open_set, (0, start))

    came_from = {}

    g_score = {
        start: 0
    }

    f_score = {
        start: heuristic(start, goal)
    }

    while open_set:

        current = heapq.heappop(open_set)[1]

        if current == goal:

            path = []

            while current in came_from:
                path.append(current)
                current = came_from[current]

            path.append(start)

            return path[::-1]

        for dx, dy in directions:

            nx = current[0] + dx
            ny = current[1] + dy

            neighbor = (nx, ny)

            # BOUNDS CHECK
            if (
                nx < 0 or
                ny < 0 or
                nx >= HEIGHT or
                ny >= WIDTH
            ):
                continue

            traversal_cost = cost_map[nx, ny] * 5

            # DIAGONAL DISTANCE
            movement_cost = 1.4 if dx != 0 and dy != 0 else 1.0

            tentative_g = (
                g_score[current]
                + traversal_cost
                + movement_cost
            )

            if (
                neighbor not in g_score or
                tentative_g < g_score[neighbor]
            ):

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g

                f_score[neighbor] = (
                    tentative_g
                    + heuristic(neighbor, goal)
                )

                heapq.heappush(
                    open_set,
                    (
                        f_score[neighbor],
                        neighbor
                    )
                )

    return None

# RUN A*
path = astar(
    cost_surface,
    start,
    goal
)

if path is None:

    print("No valid path found!")

else:

    print("Path length:", len(path))

    np.save(
        "../outputs/path.npy",
        np.array(path)
    )

    print("Path saved successfully!")

# VISUALIZE
plt.figure(figsize=(8,8))

plt.imshow(cost_surface, cmap="inferno")

if path is not None:

    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]

    plt.plot(
        path_x,
        path_y,
        color="cyan",
        linewidth=2
    )

plt.scatter(
    start[1],
    start[0],
    color="green",
    s=100,
    label="Start"
)

plt.scatter(
    goal[1],
    goal[0],
    color="red",
    s=100,
    label="Goal"
)

plt.legend()

plt.title("Eco-Aware A* Path")

plt.axis("off")

plt.show()