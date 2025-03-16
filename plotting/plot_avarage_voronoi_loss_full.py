import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Parameters for selecting the range of data points
n_begin = 0 # Start index (0-based)
n_end = 80  # End index (exclusive)

# Initialize lists
sample_list = []
exact_data = []
exact_data_d2 = []
over_data = []
merge_data = []
merge_data_d2 = []

# Load data
for i in range(1138, 1195):
    print(i)
    with open(f"../data/experiment 3/voronoi_loss_K3_100_100000_80_{i}.json", "r") as file:
        temp = json.load(file)

    temp_exact = np.array(temp["exact_d1"])
    exact_data.append(temp_exact[:, 1])

    temp_exact_d2 = np.array(temp["exact_d2"])
    exact_data_d2.append(temp_exact_d2[:, 1])

    temp_over = np.array(temp["over"])
    over_data.append(temp_over[:, 1])

    temp_merge = np.array(temp["merge_d1"])
    merge_data.append(temp_merge[:, 1])

    temp_merge_d2 = np.array(temp["merge_d2"])
    merge_data_d2.append(temp_merge_d2[:, 1])

    sample_list = temp_merge[:, 0]

print(len(merge_data_d2[0]))
# Convert to numpy arrays and trim based on the selected range
exact_data = np.array(exact_data)[:, n_begin:n_end]
exact_data_d2 = np.array(exact_data_d2)[:, n_begin:n_end]
over_data = np.array(over_data)[:, n_begin:n_end]
merge_data = np.array(merge_data)[:, n_begin:n_end]
merge_data_d2 = np.array(merge_data_d2)[:, n_begin:n_end]
sample_list = sample_list[n_begin:n_end]

# Compute averages
average_exact = np.mean(exact_data, axis=0)
average_exact_d2 = np.mean(exact_data_d2, axis=0)
average_over = np.mean(over_data, axis=0)
average_merge = np.mean(merge_data, axis=0)
average_merge_d2 = np.mean(merge_data_d2, axis=0)

# Perform linear regression for all cases
x = np.array(sample_list).reshape(-1, 1)
y_list = [average_exact, average_exact_d2, average_over, average_merge, average_merge_d2]
labels = [
    "$\mathcal{D}_1(\widehat G_n, G_{*})$ (Exact)",
    "$\mathcal{D}_2(\widehat G_n, G_{*})$ (Exact)",
    "$\mathcal{D}_2(\widehat G_n, G_{*})$ (Over)",
    "$\mathcal{D}_1(\widehat G_n, G_{*})$ (Merge)",
    "$\mathcal{D}_2(\widehat G_n, G_{*})$ (Merge)"
]
colors = ['red', 'blue', 'green', 'purple', 'orange']
linestyles = ['--', '--', '--', '--', '--']


# Function to plot individual graphs
def plot_individual(x, y, label, color, linestyle):
    reg = LinearRegression().fit(x, y.reshape(-1, 1))
    coef = reg.coef_[0][0]
    intercept = reg.intercept_[0]

    plt.figure(figsize=(6, 4))
    plt.plot(sample_list, y, color=color, label=label)
    plt.plot(x, reg.predict(x), color='black', linestyle='--',
             label=f"{np.round(intercept, 1)}$n^{{{np.round(coef, 5)}}}$")

    plt.xlabel("log(sample size)", fontsize=14)
    plt.ylabel("log(loss)", fontsize=14)
    plt.title(label, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# # Plot individual figures
for i in range(1):
    plot_individual(x, y_list[i], labels[i], colors[i], linestyles[i])

# Combined figure with regression lines and coefficients in the legend
plt.figure(figsize=(8, 6))
for i in range(5):
    reg = LinearRegression().fit(x, y_list[i].reshape(-1, 1))
    coef = np.round(reg.coef_[0][0], 5)
    intercept = np.round(reg.intercept_[0], 1)

    plt.plot(sample_list, y_list[i], color=colors[i], linestyle=linestyles[i],
             label=f"{labels[i]}: {intercept}$n^{{{coef}}}$")
    plt.plot(x, reg.predict(x), color=colors[i], linestyle='dotted')

plt.xlabel("log(sample size)", fontsize=14)
plt.ylabel("log(loss)", fontsize=14)
plt.title(f"Comparison of Different Settings\n(Data Points: {n_begin} to {n_end})", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
