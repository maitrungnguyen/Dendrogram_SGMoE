import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

data = []
directory = "../data/experiment 3/"  # Ensure the path is correct

for i in range(50, 99):
    file_path = os.path.join(directory, f"voronoi_loss_K3_1000_100000_40_{i}.json")

    try:
        with open(file_path, "r") as file:
            temp = json.load(file)

        # Ensure "argmin_dic" exists and is iterable
        if "argmin_dic" in temp and isinstance(temp["argmin_dic"], list):
            for j in temp["argmin_dic"]:
                if isinstance(j, list) and len(j) > 1:  # Ensure j[1] exists
                    data.append(j[1])
                else:
                    print(f"Skipping invalid entry in {file_path}: {j}")
        else:
            print(f"Warning: 'argmin_dic' missing or not a list in {file_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {file_path}")

# Plot the histogram
plt.hist(data, bins=20, weights=np.ones(len(data)) / len(data), edgecolor='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel("Values from argmin_dic")
plt.ylabel("Percentage")
plt.title("Distribution of argmin_dic Values")
plt.show()
