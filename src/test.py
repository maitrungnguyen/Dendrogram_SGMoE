import pandas as pd

# Load the tempanomalies dataset
data = pd.read_csv("../data/tempanomalies.csv")

# Extract the predictor (X) and response (Y)
x = data["Year"].values
y = data["AnnualAnomaly"].values
K = 2  # Number of experts
p = 1  # Order of the polynomial regression for experts
q = 1  # Order of the logistic regression for the gating network

from paramNMoE import ParamNMoE
from statNMoE import StatNMoE
from emNMoE import emNMoE  # Assuming these are implemented as per earlier discussions

# Initialize and fit the model
nmoe = emNMoE(X=x, Y=y, K=K, p=p, q=q, max_iter=1500, threshold=1e-6, verbose=True)

# Output results
nmoe.summary()
nmoe.plot(["meancurve", "confregions", "clusters", "loglikelihood"])