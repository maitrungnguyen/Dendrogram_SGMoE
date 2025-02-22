import json
import numpy as np

with open("../data/output_data.json", "r") as file:
    data = json.load(file)

x = np.array([item["x"] for item in data])
y = np.array([item["y"] for item in data])

print(f"x shape: {x.shape}, y shape: {y.shape}")


from emNMoE import emNMoE

K = 3  # Number of experts
p = 1  # Order of polynomial regression
q = 1  # Order of logistic regression

fitted_model = emNMoE(X=x, Y=y, K=K, p=p, q=q, verbose=True)
fitted_model.summary()
#fitted_model.plot(["meancurve", "confregions", "clusters", "loglikelihood"])
fitted_model.output_gating_coefficients()