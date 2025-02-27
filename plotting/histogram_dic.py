import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


data = []
for i in range (7, 19):
    print(i)
    temp = []
    with open(f"../data/experiment 3/voronoi_loss_K3_1000_100000_40_{i}.json", "r") as file:
        temp = (json.load(file))

    for j in temp["argmin_dic"]:
        data.append(j[1])

plt.hist(data, weights=np.ones(len(data)) / len(data))

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
