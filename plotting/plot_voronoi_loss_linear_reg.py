import numpy as np
import matplotlib.pyplot as plt
import json

with open('../data/experiment 1/voronoi_loss_K3_1000_100000_40_30.json') as f:
    data = json.load(f)

# Extract Data
x = np.array(data['exact'])[:, 0]
y = np.array(data['exact'])[:, 1]

#Eliminate the first element
x = x[1:]
y = y[1:]

# Perform Linear Regression: y = β0 + β1 * x
X = np.vstack([np.ones(len(x)), x]).T  # Add intercept (column of ones)
beta = np.linalg.lstsq(X, y, rcond=None)[0]  # Solve for β0 and β1

# Generate fitted line
y_pred = X @ beta

# Plot Data & Regression Line
plt.figure(figsize=(8, 6))

label = "$\mathcal{D}_2(\widehat G_n, G_{*})$"
plt.plot(x, y, color='red', label = label)



plt.plot(x, y_pred, color='black', linestyle='-.', label=str(np.round(beta[0],1)) + \
                                                                        "$n^{" + str(np.round(beta[1],5)) + "}$")

# Labels & Legend
plt.xlabel("log(sample size)", fontsize=14)
plt.ylabel("log(loss)", fontsize=14)
plt.title("Log-Log Scale Linear Regression", fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Show Plot
plt.show()

# Print Regression Coefficients
print(f"Regression Equation: log(loss) = {beta[0]:.5f} + {beta[1]:.5f} * log(sample size)")
