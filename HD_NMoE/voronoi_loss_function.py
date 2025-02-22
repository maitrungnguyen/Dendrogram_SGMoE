from dendogram import Dendrogram, ComponentNode
from modelNMoE import ModelNMoE
import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import json
from paramNMoE import ParamNMoE
from statNMoE import StatNMoE


def construct_voronoi_cells(true_components, dendrogram:Dendrogram, level):
    voronoi_cell_list = []
    for i in range(len(true_components)):
        A_i = []
        true_component_i = true_components[i]
        #print(len(dendrogram.dendrogram_tree[level]))
        for j in range(len(dendrogram.dendrogram_tree[level])):
            node_j = dendrogram.dendrogram_tree[level][j]
            for k in range(len(true_components)):
                if ComponentNode.voronoi_distance(node_j, true_component_i) <= ComponentNode.voronoi_distance(node_j, true_components[k]):
                    A_i.append(j)
        voronoi_cell_list.append(A_i)
    return voronoi_cell_list

class MixingMeasure:
    def __init__(self, components, voronoi_cells = None):
        self.K = len(components)
        self.components = components
        self.voronoi_cells = voronoi_cells


def voronoi_loss_function(G:MixingMeasure, G_0:MixingMeasure, t_0, t_1):
    loss = 0
    sum_over_1 = 0
    sum_exact_1 = 0
    for i in range(G_0.K):
        if (len(G_0.voronoi_cells[i]) > 1):
            for j in G_0.voronoi_cells[i]:
                temp_sum = 0

                summ = 0
                delta_t1_w1k = np.linalg.norm(G.components[j].w_1k - G_0.components[i].w_1k - t_1)**2
                delta_b_k = np.linalg.norm(G.components[j].b_k - G_0.components[i].b_k)**2
                summ += delta_t1_w1k + delta_b_k
                temp_sum += summ**(len(G_0.voronoi_cells[i]))

                summ2 = 0
                delta_A_k = np.linalg.norm(G.components[j].A_k - G_0.components[i].A_k)**2
                delta_sigma2 = np.linalg.norm(G.components[j].sigma2 - G_0.components[i].sigma2)**2
                summ2 += delta_A_k + delta_sigma2
                temp_sum += summ2**(len(G_0.voronoi_cells[i])/2)

                temp_sum *= np.exp(G.components[j].w_0k)

                sum_over_1 += temp_sum

        if (len(G_0.voronoi_cells[i]) == 1):
            for j in G_0.voronoi_cells[i]:
                delta_t1_w1k = np.linalg.norm(G.components[j].w_1k - G_0.components[i].w_1k - t_1)**2
                delta_b_k = np.linalg.norm(G.components[j].b_k - G_0.components[i].b_k)**2
                delta_A_k = np.linalg.norm(G.components[j].A_k - G_0.components[i].A_k)**2
                delta_sigma2 = np.linalg.norm(G.components[j].sigma2 - G_0.components[i].sigma2)**2

                sum_exact_1 += (delta_t1_w1k + delta_b_k + delta_A_k + delta_sigma2)*np.exp(G.components[j].w_0k)

    sum_gating = 0
    for i in range(G_0.K):
        temp_sum = 0
        for j in G_0.voronoi_cells[i]:
            temp_sum += (np.exp(G.components[j].w_0k) - np.exp(G_0.components[i].w_0k+ t_0))
        sum_gating += temp_sum

    loss = sum_over_1 + sum_exact_1 + sum_gating
    return loss




# with open("../data/output_data_2D.json", "r") as file:
#     data = json.load(file)
#
# x = np.array(data["X"])
# y = np.array(data["y"])
#
# model = ModelNMoE(None, None)
# model.load_from_json("../data/model.json")
# ddg = Dendrogram(model, x, y)
# ddg.create_dendrogram_tree()
#
# alphak = np.array([ [ 0.3,  0.5],
#                     [ 1 ,   1],
#                     [-1,    0.5]
#                     ])
# betak = np.array([  [ 0.5  ,0.5 , 0.5],
#                     [ 1.5,  0.6,  1.1],
#                     [ 0.9,  0.5, -0.3]
#                     ])
# sigmak = np.array([0.9, 0.5, 1.2])
# true_components = []
#
# for i in range(alphak.shape[1]):
#     component = ComponentNode(alphak[:,i], betak[:,i], sigmak[i])
#     true_components.append(component)
#
# alpha = np.zeros(betak.shape[0])
# component = ComponentNode(alpha, betak[:,-1], sigmak[-1])
# true_components.append(component)
#
#
# true_voronoi_cells_0 = construct_voronoi_cells(true_components, ddg, 0)
# true_voronoi_cells_5 = construct_voronoi_cells(true_components, ddg, 8)
# print(ddg.dendrogram_tree[5])
# print(voronoi_loss_function(MixingMeasure(ddg.dendrogram_tree[8]), MixingMeasure(true_components, true_voronoi_cells_5), 0, 0))
#
#
#
#
#


