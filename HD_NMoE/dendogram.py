from modelNMoE import ModelNMoE
import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import json
from paramNMoE import ParamNMoE
from statNMoE import StatNMoE

class ComponentNode:
    def __init__(self, alpha, beta, sigma2, position=None, leafs=None):
        self.position = position
        #self.alpha = alpha
        #self.beta = beta
        self.sigma2 = sigma2
        self.children = []
        self.parent = None
        self.children_distance = None
        self.parent_distance = None
        self.leafs = None
        self.w_0k = alpha[0]
        self.w_1k = np.delete(alpha, 0)
        self.b_k = beta[0]
        self.A_k = np.delete(beta, 0)

    @staticmethod
    def distance_of_components(component1, component2):
        #print(component1.w_0k, component2.w_0k)
        w_0k_diff = 1/(np.exp(-component1.w_0k) + np.exp(-component2.w_0k))
        #print(component1.w_1k, component2.w_1k)
        w_1k_diff = np.linalg.norm(component1.w_1k - component2.w_1k)**2
        A_k_diff = np.linalg.norm(component1.A_k - component2.A_k)**2
        b_k_diff = (component1.b_k - component2.b_k)**2
        sigma2_diff = (component1.sigma2 - component2.sigma2)**2
        sum = (A_k_diff+sigma2_diff)**(1/2)
        #print("Check")
        return w_0k_diff*(b_k_diff + sum)

    @staticmethod
    def merge(component1, component2):
        total_w0k = log(np.exp(component1.w_0k) + np.exp(component2.w_0k))
        #print("Merge:" , component1.w_0k, component2.w_0k, total_w0k)
        exp_1 = np.exp(component1.w_0k - total_w0k)
        exp_2 = np.exp(component2.w_0k - total_w0k)
        total_w1k = (exp_1 * component1.w_1k + exp_2 * component2.w_1k)
        total_Ak = (exp_1 * component1.A_k + exp_2 * component2.A_k)
        total_bk = (exp_1 * component1.b_k + exp_2 * component2.b_k)
        #print([component1.w_1k, component1.A_k[0], component1.A_k[1], component1.b_k])


        #mu_i = np.array([component1.w_1k[0], component1.w_1k[1], component1.A_k[0], component1.A_k[1], component1.b_k])
        #mu_j = np.array([component2.w_1k[0], component2.w_1k[1], component2.A_k[0], component1.A_k[1], component2.b_k])
        #mu_total = np.array([total_w1k[0], total_w1k[1], total_Ak[0], total_Ak[1], total_bk])
        # mu_i = []
        # mu_j = []
        # mu_total = []
        # for i in range(len(component1.w_1k)):
        #     mu_i.append(component1.w_1k[i])
        #     mu_j.append(component2.w_1k[i])
        #     mu_total.append(total_w1k[i])
        # for i in range(len(component1.A_k)):
        #     mu_i.append(component1.A_k[i])
        #     mu_j.append(component2.A_k[i])
        #     mu_total.append(total_Ak[i])
        # mu_i.append(component1.b_k)
        # mu_j.append(component2.b_k)
        # mu_total.append(total_bk)
        # mu_i = np.array(mu_i)
        # mu_j = np.array(mu_j)
        # mu_total = np.array(mu_total)



        total_sigma2 = (exp_1 * component1.sigma2 + exp_2 * component2.sigma2)

        #total_sigma2 = exp_1*(component1.sigma2 + (mu_i - mu_total)@(mu_i - mu_total).T) + exp_2*(component2.sigma2 + (mu_j - mu_total)@(mu_j - mu_total).T)

        #print("New Component:", total_w0k, total_w1k, total_Ak, total_bk, total_sigma2)
        new_component = ComponentNode(np.concatenate(([total_w0k], total_w1k)), np.concatenate(([total_bk], total_Ak)), total_sigma2, position=None, leafs=component1.get_leafs() + component2.get_leafs())
        new_component.children.append(component1)
        new_component.children.append(component2)
        new_component.children_distance = ComponentNode.distance_of_components(component1, component2)
        component1.parent_distance = new_component.children_distance
        component2.parent_distance = new_component.children_distance
        component1.parent = new_component
        component2.parent = new_component
        return new_component

    def is_leaf(self):
        return len(self.children) == 0

    def get_leafs(self):
        if self.leafs is not None:
            return self.leafs
        self.leafs = self.count_children_leaf()
        return self.leafs

    def count_children_leaf(self):
        if self.is_leaf():
            return 1
        return sum([child.count_children_leaf() for child in self.children])

    def display_component(self):
        print("Component:")
        print("w_0k:", self.w_0k)
        print("w_1k:", self.w_1k)
        print("A_k:", self.A_k)
        print("b_k:", self.b_k)
        print("sigma2:", self.sigma2)
        print("Position:", self.position)

    def voronoi_distance(component1, component2):
        # omega_1 = np.array([component1.w_1k, component1.A_k, component1.b_k, component1.sigma2])
        # omega_2 = np.array([component2.w_1k, component2.A_k, component2.b_k, component2.sigma2])
        w_1k_diff = np.linalg.norm(component1.w_1k - component2.w_1k)**2
        A_k_diff = np.linalg.norm(component1.A_k - component2.A_k)**2
        b_k_diff = np.linalg.norm(component1.b_k - component2.b_k)**2
        sigma2_diff = abs(component1.sigma2 - component2.sigma2)**2
        return (w_1k_diff + A_k_diff + b_k_diff + sigma2_diff)**(1/2)


class Dendrogram:
    def __init__(self, model:ModelNMoE, X, Y):
        self.X = X
        self.Y = Y
        self.model = model
        self.p = model.param.p
        self.q = model.param.q
        self.components_list = []
        self.nodes_list = []
        self.K = model.param.K
        for i in range(self.K-1):
            component = ComponentNode(
                model.param.alpha[: , i],
                model.param.beta[:,i],
                model.param.sigma2[i],
                position=i)
            self.components_list.append(component)
            self.nodes_list.append(component)
        alpha_zero = np.zeros(model.param.alpha.shape[0])
        component = ComponentNode(alpha_zero, model.param.beta[:, -1], model.param.sigma2[-1], position=self.K-1)
        self.components_list.append(component)
        self.nodes_list.append(component)
        self.mixing_measure = None
        self.distance_list = []
        self.dendrogram_tree = None
        self.merge_procedure = []
        self.dic = None
        self.aic = None
        self.bic = None
        # for component in self.components_list:
        #     component.display_component()

    @staticmethod

    def distance_of_components(component1, component2):
        return ComponentNode.distance_of_components(component1, component2)

    def merge_components(self, component1, component2):
        return ComponentNode.merge(component1, component2)

    def min_distance(self, list):
        min_distance = float('inf')
        x, y = -1, -1
        for i in range(len(list)):
            for j in range(i + 1, len(list)):
                distance = self.distance_of_components(list[i], list[j])
                if distance < min_distance:
                    min_distance = distance
                    x, y = i, j
        return x, y, min_distance

    def create_dendrogram_tree(self):
        temp_dendrogram_tree = []
        temp_dendrogram_tree.append(self.components_list)

        while len(temp_dendrogram_tree[-1]) != 1:
            x, y, min_distance = self.min_distance(temp_dendrogram_tree[-1])
            z = self.merge_components(temp_dendrogram_tree[-1][x], temp_dendrogram_tree[-1][y])
            z.position = len(self.nodes_list)
            self.nodes_list.append(z)
            procedure = [temp_dendrogram_tree[-1][x].position, temp_dendrogram_tree[-1][y].position, min_distance, z.get_leafs()]
            next_level = temp_dendrogram_tree[-1].copy()
            next_level.remove(temp_dendrogram_tree[-1][x])
            next_level.remove(temp_dendrogram_tree[-1][y])
            next_level.append(z)
            self.distance_list.append(min_distance)
            temp_dendrogram_tree.append(next_level)
            self.merge_procedure.append(procedure)
            # print("Current Comp:")
            # for j in next_level:
            #     j.display_component()
        self.dendrogram_tree = temp_dendrogram_tree
        return temp_dendrogram_tree

    def display_dendrogram_tree(self):
        if self.dendrogram_tree is None:
            self.create_dendrogram_tree()
        for level in self.dendrogram_tree:
            print("Level:", level)
            print("Number of Components:", len(level))
            for component in level:
                component.display_component()

    def plot_dendrogram(self):
        if self.dendrogram_tree is None:
            self.create_dendrogram_tree()

        levels = self.dendrogram_tree

        num_initial_components = len(levels[0])

        linkage_matrix = self.merge_procedure

        linkage_matrix = np.array(linkage_matrix)
        print(linkage_matrix)
        initial_labels = [str(i) for i in range(num_initial_components)]

        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix, labels=initial_labels)
        plt.title("Custom Dendrogram Plot")
        plt.xlabel("Component Index")
        plt.ylabel("Distance")
        plt.show()


    def calculate_dic_for_level_with_score(self, level):
        """
        Calculate the Dendrogram Information Criterion (DIC) for a specific level using sklearn's score().

        Parameters:
        - data: np.array, the dataset (n_samples, n_features).
        - components: list of ComponentNode, components in the given level of the dendrogram.
        - dendrogram: Dendrogram, the dendrogram object containing levels and distances.
        - level: int, the current level in the dendrogram.

        Returns:
        - dic_value: float, the DIC value for the given level.
        """
        current_gating_coefficients = None
        current_regression_coefficients = None
        current_sigma2 = None

        for component in self.dendrogram_tree[level]:
            component_alpha = np.concatenate(([component.w_0k], component.w_1k))
            component_beta = np.concatenate(([component.b_k], component.A_k))
            component_sigma2 = component.sigma2
            current_gating_coefficients = np.column_stack((current_gating_coefficients, component_alpha)) if current_gating_coefficients is not None else component_alpha.reshape(-1, 1)
            current_regression_coefficients = np.column_stack((current_regression_coefficients, component_beta)) if current_regression_coefficients is not None else component_beta.reshape(-1, 1)
            current_sigma2 = np.append(current_sigma2, component_sigma2) if current_sigma2 is not None else np.array([component_sigma2])

        #print("Number of Components:", len(self.dendrogram_tree[level]))
        # print("Level:", level)
        # print("Gating Coefficients:")
        # print(current_gating_cofficients)
        # print(current_regression_coefficients)
        # print(current_sigma2)


        # current_paramNMoE = ParamNMoE(self.X, self.Y, len(self.dendrogram_tree[level]), self.p, self.q)
        # current_paramNMoE.alpha = current_gating_cofficients
        # current_paramNMoE.beta = current_regression_coefficients
        # current_paramNMoE.sigma2 = current_sigma2
        #
        # current_statNMoE = StatNMoE(current_paramNMoE)
        # current_statNMoE.EStep(current_paramNMoE)
        #
        # current_paramNMoE.init_param(segmental=False)
        # reg_IRLS = current_paramNMoE.M_step(current_statNMoE, False)
        #
        # current_statNMoE.computeLikelihood(reg_IRLS)
        #
        #
        # current_statNMoE.compute_AIC_BIC_loglik(current_paramNMoE)
        # aic_value = current_statNMoE.AIC
        # bic_value = current_statNMoE.BIC
        log_likelihood = self.compute_log_likelihood(self.X, self.Y, current_gating_coefficients, current_regression_coefficients, current_sigma2)

        k = (
                current_gating_coefficients.size  # M * d
                + current_regression_coefficients.size  # M * d
                + current_sigma2.size  # M
        )

        n = self.X.shape[0]  # number of data points

        aic_value = -2 * log_likelihood + 2 * k
        bic_value = -2 * log_likelihood + k * np.log(n)

        d_n_k = self.distance_list[level]
        # print("Distance:",d_n_k)

        # print("Log Likelihood:", log_likelihood)
        #print("DL:", self.distance_list)
        # print("D_n_k:", d_n_k)
        # print("Log Likelihood:", log_likelihood)
        # print(self.X.shape[0])
        dic_value = -(d_n_k + np.log(self.X.shape[0]) * log_likelihood/self.X.shape[0])
        # print("DIC:", dic_value)
        # print("AIC:", aic_value)
        # print("BIC:", bic_value)

        return dic_value, aic_value, bic_value

    def plot_dic_aic_bic(self):
        dic_scores = []
        aic_scores = []
        bic_scores = []

        for i in range(1, len(self.dendrogram_tree)-1):
            #print("Level:", i)
            dic, aic, bic = self.calculate_dic_for_level_with_score(i)
            dic_scores.append(dic)
            aic_scores.append(aic)
            bic_scores.append(bic)

        print(dic_scores)
        self.dic = dic_scores
        self.aic = aic_scores
        self.bic = bic_scores

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.dendrogram_tree)), dic_scores, label='DIC')
        plt.plot(range(1, len(self.dendrogram_tree)), aic_scores, label='AIC')
        plt.plot(range(1, len(self.dendrogram_tree)), bic_scores, label='BIC')
        plt.title("DIC, AIC, and BIC Scores for Dendrogram Levels")
        plt.xlabel("Dendrogram Level")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

    import numpy as np

    def compute_log_likelihood(self, X, Y, alpha, beta, sigma2):
        """
        Compute the log-likelihood of a mixture of experts model with bias terms.

        Parameters:
        - X: Input data (n x 2), where each row is a 2D vector.
        - Y: Target data (n,), scalar outputs for each observation.
        - alpha: Gating parameters (3 x K), includes bias terms.
        - beta: Regression parameters (3 x K), includes bias terms.
        - sigma2: Variance parameters (K,).

        Returns:
        - log_likelihood: The log-likelihood of the model.
        """
        # Add intercept (bias term) to the input
        intercept = np.ones((X.shape[0], 1))  # (n x 1)
        X_bias = np.hstack((intercept, X))  # (n x 3)

        n, d = X_bias.shape  # d should now be 3
        K = beta.shape[1]  # Number of components

        # Compute gating probabilities (softmax over components)
        gating_scores = X_bias @ alpha  # (n x K)
        gating_probs = np.exp(gating_scores - np.max(gating_scores, axis=1, keepdims=True))  # Stabilize softmax
        gating_probs /= np.sum(gating_probs, axis=1, keepdims=True)  # (n x K)

        # Compute component densities
        component_densities = np.zeros((n, K))
        for k in range(K):
            mu_k = X_bias @ beta[:, k]  # Predicted mean for component k (n,)
            var_k = sigma2[k]
            component_densities[:, k] = (1 / np.sqrt(2 * np.pi * var_k)) * \
                                        np.exp(-0.5 * ((Y - mu_k) ** 2) / var_k)  # Gaussian density (n,)

        # Combine using the mixture model
        likelihoods = np.sum(gating_probs * component_densities, axis=1)  # Mixture likelihood for each point (n,)

        # Compute the log-likelihood
        log_likelihood = np.sum(np.log(likelihoods + 1e-9))  # Add small constant for numerical stability

        return log_likelihood

    def argmin_dic(self):
        if self.dendrogram_tree is None:
            self.create_dendrogram_tree()
        dic_scores = []
        aic_scores = []
        bic_scores = []

        for i in range(0, len(self.dendrogram_tree)-1):
            #print("Level:", i)
            dic, aic, bic = self.calculate_dic_for_level_with_score(i)
            #print(dic)
            dic_scores.append(dic)
            aic_scores.append(aic)
            bic_scores.append(bic)

        #print(dic_scores)
        self.dic = dic_scores
        self.aic = aic_scores
        self.bic = bic_scores

        return int(np.argmin(self.dic))

def json_reader(file):
    try:
        with open(file, 'r') as f:
            samples_loaded = np.array(json.load(f))
            return samples_loaded
    except Exception as e:
        print(f'Can not open JSON file: {e}')


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
# ddg.plot_dendrogram()
# #dendrogram.display_dendrogram_tree()
#
# ddg.plot_dic_aic_bic()
#
# print(ddg.argmin_dic())