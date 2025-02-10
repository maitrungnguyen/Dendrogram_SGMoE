import json
import numpy as np
from modelNMoE import ModelNMoE
import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import json
from paramNMoE import ParamNMoE
from statNMoE import StatNMoE
from emNMoE import emNMoE
from scipy.cluster.hierarchy import dendrogram, linkage
from dendogram import Dendrogram, ComponentNode
from sampleUniv import sample_univ_nmoe
from voronoi_loss_function import construct_voronoi_cells, voronoi_loss_function, MixingMeasure

np.random.seed(42)
n_samples_list = [100, 1000, 8000, 10000, 20000, 27000, 58000, 63000, 81000, 100000, 200000, 500000, 800000, 1000000]
#n_samples_list = [100, 1000]

n_features = 2
n_components = 3
alphak = np.array([[0.3, 0.5],
                   [1, 1],
                   [-1, 0.5]
                   ])
betak = np.array([[0.5, 0.5, 0.5],
                  [1.5, 0.6, 1.1],
                  [0.9, 0.5, -0.3]
                  ])
sigmak = np.array([0.9, 0.5, 1.2])

true_components = []

for i in range(alphak.shape[1]):
    component = ComponentNode(alphak[:,i], betak[:,i], sigmak[i])
    true_components.append(component)

alpha = np.zeros(betak.shape[0])
component = ComponentNode(alpha, betak[:,-1], sigmak[-1])
true_components.append(component)

exact_voronoi_loss = []
over_voronoi_loss = []
merge_voronoi_loss = []

for n_samples in n_samples_list:
    print(n_samples)
    X = np.random.normal(0, 1, size=(n_samples, n_features))


    # Sample data
    data = sample_univ_nmoe(alphak, betak, sigmak, X)

    data = {
        "X": X.tolist(),
        "y": data["y"].tolist(),
    }

    x = np.array(data["X"])
    y = np.array(data["y"])

    K = 3
    p = 1
    q = 1
    exact_fitted_model = emNMoE(X=x, Y=y, K=K, p=p, q=q, verbose=False)
    exact_ddg = Dendrogram(exact_fitted_model, x, y)
    exact_ddg.create_dendrogram_tree()
    true_voronoi_cells_0 = construct_voronoi_cells(true_components, exact_ddg, 0)
    # exact_voronoi_loss.append({
    #     "log_n_samples": log(n_samples),
    #     "log_voronoi_loss": log(voronoi_loss_function(MixingMeasure(exact_ddg.dendrogram_tree[0]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
    # })
    exact_voronoi_loss.append([
        log(n_samples),
        log(voronoi_loss_function(MixingMeasure(exact_ddg.dendrogram_tree[0]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
    ])

    K = 5
    p = 1
    q = 1
    over_fitted_model = emNMoE(X=x, Y=y, K=K, p=p, q=q, verbose=False)
    over_ddg = Dendrogram(over_fitted_model, x, y)
    over_ddg.create_dendrogram_tree()
    true_voronoi_cells_0 = construct_voronoi_cells(true_components, over_ddg, 0)
    # over_voronoi_loss.append({
    #     "log_n_samples": log(n_samples),
    #     "log_voronoi_loss": log(voronoi_loss_function(MixingMeasure(over_ddg.dendrogram_tree[0]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
    # })
    over_voronoi_loss.append([
        log(n_samples),
        log(voronoi_loss_function(MixingMeasure(over_ddg.dendrogram_tree[0]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
    ])

    K = 5
    p = 1
    q = 1
    merge_fitted_model = emNMoE(X=x, Y=y, K=K, p=p, q=q, verbose=False)
    merge_ddg = Dendrogram(merge_fitted_model, x, y)
    merge_ddg.create_dendrogram_tree()
    true_voronoi_cells_0 = construct_voronoi_cells(true_components, merge_ddg, 3)
    # merge_voronoi_loss.append({
    #     "log_n_samples": log(n_samples),
    #     "log_voronoi_loss": log(voronoi_loss_function(MixingMeasure(merge_ddg.dendrogram_tree[3]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
    # })
    merge_voronoi_loss.append([
        log(n_samples),
        log(voronoi_loss_function(MixingMeasure(merge_ddg.dendrogram_tree[3]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
    ])

exact_voronoi_loss = np.array(exact_voronoi_loss)
over_voronoi_loss = np.array(over_voronoi_loss)
merge_voronoi_loss = np.array(merge_voronoi_loss)

with open("../data/voronoi_loss_2.json", "w") as file:
    json.dump({
        "exact": exact_voronoi_loss.tolist(),
        "over": over_voronoi_loss.tolist(),
        "merge": merge_voronoi_loss.tolist()
    }, file)

plt.plot(exact_voronoi_loss[:, 0], exact_voronoi_loss[:, 1], label="Exact")
plt.plot(over_voronoi_loss[:, 0], over_voronoi_loss[:, 1], label="Over")
plt.plot(merge_voronoi_loss[:, 0], merge_voronoi_loss[:, 1], label="Merge")
plt.xlabel("Log number of samples")
plt.ylabel("Log Voronoi loss")
plt.legend()
plt.show()

#plt.savefig("../figures/voronoi_loss_2.png")




