import matplotlib.pyplot as plt
import numpy as np
import json

with open("../data/voronoi_loss_2.json", "r") as file:
    data = json.load(file)

exact_voronoi_loss = data["exact"]
over_voronoi_loss = data["over"]
merge_voronoi_loss = data["merge"]

exact_voronoi_loss = np.array(exact_voronoi_loss)
over_voronoi_loss = np.array(over_voronoi_loss)
merge_voronoi_loss = np.array(merge_voronoi_loss)

plt.plot(exact_voronoi_loss[:, 0], exact_voronoi_loss[:, 1], label="Exact")
plt.plot(over_voronoi_loss[:, 0], over_voronoi_loss[:, 1], label="Over")
plt.plot(merge_voronoi_loss[:, 0], merge_voronoi_loss[:, 1], label="Merge")
plt.legend()
plt.show()