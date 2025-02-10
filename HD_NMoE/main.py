# import json
# import numpy as np
#
# with open("../data/output_data_2D.json", "r") as file:
#     data = json.load(file)
#
# x = np.array(data["X"])
# #x_d = [[ xi, xi-1] for xi in x]
# #x = np.array(x_d)
# y = np.array(data["y"])
#
# print(f"x shape: {x.shape}, y shape: {y.shape}")
# #print(x)
# #print(y)
#
# from emNMoE import emNMoE
#
# K = 12 # Number of experts
# p = 1  # Order of polynomial regression
# q = 1  # Order of logistic regression
#
# fitted_model = emNMoE(X=x, Y=y, K=K, p=p, q=q, verbose=True)
# fitted_model.summary()
# fitted_model.output_to_json("../data/median_file.json")

from voronoi_experiment_w_nmin import voronoi_experiment_w_nmin

voronoi_experiment_w_nmin(1500, 100000, 80)