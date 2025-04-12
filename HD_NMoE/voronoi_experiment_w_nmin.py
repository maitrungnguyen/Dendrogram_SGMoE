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
from voronoi_loss_function import construct_voronoi_cells, voronoi_loss_function, voronoi_loss_D1, MixingMeasure
import time

def voronoi_experiment_w_nmin(n_min, n_max, n_iter,
                              n_features=2,
    n_components = 3,
    n_over_components = 5,
            alphak = np.array([[0.3, 0.5],
                       [1, 1],
                       [-1, 0.5]
                       ]),
    betak = np.array([[0.5, 0.5, 0.5],
                      [1.5, 0.6, 1.1],
                      [0.9, 0.5, -0.3]
                      ]),
    sigmak = np.array([0.9, 0.5, 1.2]),
    n_tries = 1,
    name_exp = "experiment 1", favourable = False,
    exact_enable = True,
    seed = 2025,
    spacing_type = "log",
    ):

    start = time.time()


    alphak = np.array(alphak)
    betak = np.array(betak)
    sigmak = np.array(sigmak)


    true_components = []

    for i in range(alphak.shape[1]):
        component = ComponentNode(alphak[:,i], betak[:,i], sigmak[i])
        true_components.append(component)

    alpha = np.zeros(betak.shape[0])
    component = ComponentNode(alpha, betak[:,-1], sigmak[-1])
    true_components.append(component)

    argmin_dic = []
    exact_voronoi_loss = []
    exact_d1_voronoi_loss = []
    over_voronoi_loss = []
    merge_voronoi_loss = []
    merge_d1_voronoi_loss = []
    estimators = []
    data_records = []
    mixing_measure_records = []



    np.random.seed(seed)
    if spacing_type == "linear":
        n_values = np.linspace(n_min, n_max, num=n_iter)
    elif spacing_type == "log":
        n_values = np.logspace(np.log(n_min), np.log(n_max), num=n_iter, base=np.e)
    else:
        raise ValueError("Unsupported spacing type. Use 'linear' or 'log'.")
    n_values = np.ceil(n_values).astype(int)
    iter_no = 0

    #Main loop
    for n_samples in n_values:
        print(f"Running iteration {iter_no+1}/{n_iter} with n_samples = {n_samples}")
        iter_no += 1
        n_samples = int(n_samples)

        X = np.random.uniform(0, 1, (n_samples, n_features))

        # Sample data
        data = sample_univ_nmoe(alphak, betak, sigmak, X)

        data = {
            "X": X.tolist(),
            "y": data["y"].tolist(),
        }

        x = np.array(data["X"])
        y = np.array(data["y"])

        if exact_enable:
            K = n_components
            p = 1
            q = 1
            exact_fitted_model = emNMoE(X=x, Y=y, K=K, p=p, q=q, verbose=False, favourable = favourable, true_alpha = alphak, true_beta = betak, true_sigma2 = sigmak)
            exact_ddg = Dendrogram(exact_fitted_model, x, y)
            exact_ddg.create_dendrogram_tree()
            true_voronoi_cells_0 = construct_voronoi_cells(true_components, exact_ddg, 0)
            exact_voronoi_loss.append([
                log(n_samples),
                log(voronoi_loss_function(MixingMeasure(exact_ddg.dendrogram_tree[0]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
            ])
            exact_d1_voronoi_loss.append([
                log(n_samples),
                log(voronoi_loss_D1(MixingMeasure(exact_ddg.dendrogram_tree[0]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
            ])

        K = n_over_components
        p = 1
        q = 1
        over_fitted_model = emNMoE(X=x, Y=y, K=K, p=p, q=q, verbose=False, favourable = favourable, true_alpha = alphak, true_beta = betak, true_sigma2 = sigmak)
        merge_fitted_model = over_fitted_model
        over_ddg = Dendrogram(over_fitted_model, x, y)
        over_ddg.create_dendrogram_tree()
        true_voronoi_cells_0 = construct_voronoi_cells(true_components, over_ddg, 0)
        over_voronoi_loss.append([
            log(n_samples),
            log(voronoi_loss_function(MixingMeasure(over_ddg.dendrogram_tree[0]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
        ])

        merge_ddg = Dendrogram(merge_fitted_model, x, y)
        merge_ddg.create_dendrogram_tree()
        print("Infered true K_0 from DIC:",n_over_components - merge_ddg.argmin_dic())
        argmin_dic.append([n_samples, n_over_components - merge_ddg.argmin_dic()])
        true_voronoi_cells_0 = construct_voronoi_cells(true_components, merge_ddg, n_over_components-n_components)
        merge_voronoi_loss.append([
            log(n_samples),
            log(voronoi_loss_function(MixingMeasure(merge_ddg.dendrogram_tree[n_over_components-n_components]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
        ])
        merge_d1_voronoi_loss.append([
            log(n_samples),
            log(voronoi_loss_D1(MixingMeasure(merge_ddg.dendrogram_tree[n_over_components-n_components]), MixingMeasure(true_components, true_voronoi_cells_0), 0, 0))
        ])

        current_estimators = dict()
        current_estimators["n_samples"] = n_samples
        current_estimators["exact"] = exact_fitted_model.list_of_parameters()
        current_estimators["over"] = over_fitted_model.list_of_parameters()
        estimators.append(current_estimators)

        current_mixing_measure = dict()
        current_mixing_measure["n_samples"] = n_samples
        current_mixing_measure["exact"] = MixingMeasure(exact_ddg.dendrogram_tree[0]).list_components()
        current_mixing_measure["over"] = MixingMeasure(over_ddg.dendrogram_tree[0]).list_components()
        current_mixing_measure["merge"] = MixingMeasure(merge_ddg.dendrogram_tree[n_over_components-n_components]).list_components()
        mixing_measure_records.append(current_mixing_measure)


    exact_d1_voronoi_loss = np.array(exact_d1_voronoi_loss)
    merge_d1_voronoi_loss = np.array(merge_d1_voronoi_loss)
    exact_voronoi_loss = np.array(exact_voronoi_loss)
    over_voronoi_loss = np.array(over_voronoi_loss)
    merge_voronoi_loss = np.array(merge_voronoi_loss)
    name = "../data/" + name_exp + "/voronoi_loss_K" + str(n_over_components)+"-" + str(n_components) + "_" + str(n_min) + "_" + str(n_max) + "_" + str(n_iter) + "_" + str(n_tries) + ".json"
    time_elapsed = time.time() - start


    with open(name, "w") as file:
        json.dump({
            "random_seed": seed,
            "spacing_type": spacing_type,
            "exact_d1": exact_d1_voronoi_loss.tolist(),
            "exact_d2": exact_voronoi_loss.tolist(),
            "over": over_voronoi_loss.tolist(),
            "merge_d1": merge_d1_voronoi_loss.tolist(),
            "merge_d2": merge_voronoi_loss.tolist(),
            "argmin_dic": argmin_dic,
            "estimators": estimators,
            "mixing_measure": mixing_measure_records,
            "time_elapsed": time_elapsed
        }, file)

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj





