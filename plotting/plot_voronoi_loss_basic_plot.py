
import json
import numpy as np
import matplotlib.pyplot as plt
import time

n_min = 1000
n_max = 100000
n_iter = 80
iter = (n_max - n_min) // 40

name = "../data/voronoi_loss_K3_" + str(n_min) + "_" + str(n_max) + "_" + str(n_iter) + ".json"
with open(name) as f:
    data = json.load(f)

exact_voronoi_loss = np.array(data['exact'])
over_voronoi_loss = np.array(data['over'])
merge_voronoi_loss = np.array(data['merge'])



plt.plot(exact_voronoi_loss[:, 0], exact_voronoi_loss[:, 1], label="Exact")
plt.plot(over_voronoi_loss[:, 0], over_voronoi_loss[:, 1], label="Over")
plt.plot(merge_voronoi_loss[:, 0], merge_voronoi_loss[:, 1], label="Merge")
plt.xlabel("Log number of samples")
plt.ylabel("Log Voronoi loss")
plt.legend()
#plt.show()

savename = "../figures/voronoi_loss_K3_" + str(n_min) + "_" + str(n_max) + "_" + str(n_iter) + ".png"
plt.savefig(savename)