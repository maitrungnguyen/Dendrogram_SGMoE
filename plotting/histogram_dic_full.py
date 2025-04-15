import json
import numpy as np
import matplotlib.pyplot as plt

# Path to data folder
directory = "../data/experiment 5"
setting = "voronoi_loss_K4-2_100_100000_80"
true_argmin_dic = 2 # True K_0

# Change this if you have 2 ranges of data:
# 0: only first range
# 1: only second range
# 2: both ranges
dual = 2
if dual == 0:
    indices = range(1000, 1024)
elif dual == 1:
    indices = range(1100, 1111)
elif dual == 2:
    indices = list(range(1000, 1024)) + list(range(1100, 1111))
else:
    raise ValueError("Invalid value for 'dual'. Use 0, 1, or 2.")


n_counts = {}
correct_counts = {}
for i in indices:
    file_path = f"{directory}/{setting}_{i}.json"
    try:
        with open(file_path, "r") as file:
            temp = json.load(file)

        if "argmin_dic" in temp and isinstance(temp["argmin_dic"], list):
            for entry in temp["argmin_dic"]:
                if isinstance(entry, list) and len(entry) == 2:
                    n, argmin_val = entry

                    # Initialize count tracking for n
                    if n not in n_counts:
                        n_counts[n] = 0
                        correct_counts[n] = 0

                    # Update counts
                    n_counts[n] += 1
                    if argmin_val == true_argmin_dic:
                        correct_counts[n] += 1
                else:
                    print(f"Skipping invalid entry: {entry}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {file_path}")

# Compute percentages for each n
n_values = sorted(n_counts.keys())  # Ensure n values are in order
percentages = [(correct_counts[n] / n_counts[n]) * 100 if n_counts[n] > 0 else 0 for n in n_values]

# Convert n values to log scale
log_n_values = np.log(n_values)

# Plot the line graph
plt.figure(figsize=(10, 5))
plt.plot(log_n_values, percentages, marker='o', linestyle='-', color='blue', label="Correct argmin %")

plt.xlabel("$\log(n)$ Values")
plt.ylabel("Percentage of correct true $K_{0}$")
plt.title("Percentage of correct true $K_{0}$ per $\log(n)$")
plt.ylim(0, 100)  # y-axis range from 0% to 100%
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.show()
