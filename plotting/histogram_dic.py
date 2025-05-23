import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


data = []
for i in range (1, 4):
    print(i)
    temp = []
    with open(f"../data/experiment 1/voronoi_loss_K3_100000_100000_40_{i}.json", "r") as file:
        temp = (json.load(file))

    for j in temp["argmin_dic"]:
        data.append(j[1]+1)

plt.hist(data, weights=np.ones(len(data)) / len(data))

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
