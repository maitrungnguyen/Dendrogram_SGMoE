import json
import numpy as np
import matplotlib.pyplot as plt

data = []
for i in range (1, 4):
    print(i)
    temp = []
    with open(f"../data/voronoi_loss_K3_35000_35000_40_{i}.json", "r") as file:
        temp = (json.load(file))

    for j in temp["argmin_dic"]:
        data.append(j[1])

#histogram
plt.hist(data, bins = 100)
plt.show()
