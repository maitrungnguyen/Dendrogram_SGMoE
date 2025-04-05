import numpy as np
import matplotlib.pyplot as plt
from paramNMoE import ParamNMoE
from statNMoE import StatNMoE
import json

class ModelNMoE:
    """
    A class representing a fitted Normal Mixture of Experts (NMoE) model.

    Attributes:
    - param: Instance of ParamNMoE containing model parameters.
    - stat: Instance of StatNMoE containing model statistics.
    """


    def __init__(self, param:ParamNMoE, stat):
        self.param = param
        self.stat = stat

    '''
    def plot(self, what=("meancurve", "confregions", "clusters", "loglikelihood")):
        """
        Plot various aspects of the fitted model.

        Parameters:
        - what: List of plots to display. Options are:
          - "meancurve": Mean curve and expert means.
          - "confregions": Confidence regions (mean ± 2*std).
          - "clusters": Experts' means and clusters.
          - "loglikelihood": Log-likelihood values over EM iterations.
        """
        colors = plt.cm.rainbow(np.linspace(0, 1, self.param.K))

        if "meancurve" in what:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.param.X, self.param.Y, c="black", s=10, label="Data")
            plt.plot(self.param.X, self.stat.Ey, c="red", label="Estimated mean", lw=2)
            for k in range(self.param.K):
                plt.plot(self.param.X, self.stat.Ey_k[:, k], c="blue", linestyle="--", label=f"Expert {k + 1} mean")
            plt.title("Estimated Mean and Experts")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.show()

        if "confregions" in what:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.param.X, self.param.Y, c="black", s=10, label="Data")
            plt.plot(self.param.X, self.stat.Ey, c="red", label="Estimated mean", lw=2)
            plt.fill_between(
                self.param.X.flatten(),
                self.stat.Ey.flatten() - 2 * np.sqrt(self.stat.Vary.flatten()),
                self.stat.Ey.flatten() + 2 * np.sqrt(self.stat.Vary.flatten()),
                color="red",
                alpha=0.2,
                label="Confidence region (±2σ)"
            )
            plt.title("Estimated Mean and Confidence Regions")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.show()

        if "clusters" in what:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.param.X, self.param.Y, c="black", s=10, label="Data")
            for k in range(self.param.K):
                indices = self.stat.klas.flatten() == k + 1
                plt.scatter(
                    self.param.X[indices],
                    self.param.Y[indices],
                    c=[colors[k]],
                    label=f"Cluster {k + 1}",
                    s=10
                )
                plt.plot(self.param.X, self.stat.Ey_k[:, k], c=colors[k], linestyle="--", lw=2,
                         label=f"Expert {k + 1} mean")
            plt.title("Estimated Experts and Clusters")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.show()

        if "loglikelihood" in what:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(self.stat.stored_loglik) + 1), self.stat.stored_loglik, c="blue", lw=2)
            plt.title("Log-Likelihood Over EM Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Log-Likelihood")
            plt.show()
'''

    def plot(self, what=("meancurve", "confregions", "clusters", "loglikelihood")):
        """
        Plot various aspects of the fitted model, with support for multivariate inputs.

        Parameters:
        - what: List of plots to display. Options are:
          - "meancurve": Mean curve and expert means.
          - "confregions": Confidence regions (mean ± 2*std).
          - "clusters": Experts' means and clusters.
          - "loglikelihood": Log-likelihood values over EM iterations.
        """
        colors = plt.cm.rainbow(np.linspace(0, 1, self.param.K))
        X = self.param.X
        Y = self.param.Y

        if X.ndim > 1:
            # Multidimensional input detected
            print("Input X is multivariate; showing pairwise relationships and clusters.")

        if "meancurve" in what and X.ndim == 1:
            plt.figure(figsize=(8, 6))
            plt.scatter(X, Y, c="black", s=10, label="Data")
            plt.plot(X, self.stat.Ey, c="red", label="Estimated mean", lw=2)
            for k in range(self.param.K):
                plt.plot(X, self.stat.Ey_k[:, k], c="blue", linestyle="--", label=f"Expert {k + 1} mean")
            plt.title("Estimated Mean and Experts")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.show()

        if "meancurve" in what and X.ndim > 1:
            from itertools import combinations
            n_features = X.shape[1]
            feature_pairs = list(combinations(range(n_features), 2))
            plt.figure(figsize=(12, 8))
            for idx, (f1, f2) in enumerate(feature_pairs[:4]):  # Limit to first 4 pairs for readability
                plt.subplot(2, 2, idx + 1)
                plt.scatter(X[:, f1], X[:, f2], c=Y, cmap='viridis', s=10)
                plt.xlabel(f"Feature {f1 + 1}")
                plt.ylabel(f"Feature {f2 + 1}")
                plt.title(f"Feature Pair {f1 + 1} vs {f2 + 1}")
            plt.tight_layout()
            plt.show()

        if "confregions" in what and X.ndim == 1:
            plt.figure(figsize=(8, 6))
            plt.scatter(X, Y, c="black", s=10, label="Data")
            plt.plot(X, self.stat.Ey, c="red", label="Estimated mean", lw=2)
            plt.fill_between(
                X.flatten(),
                self.stat.Ey.flatten() - 2 * np.sqrt(self.stat.Vary.flatten()),
                self.stat.Ey.flatten() + 2 * np.sqrt(self.stat.Vary.flatten()),
                color="red",
                alpha=0.2,
                label="Confidence region (±2σ)"
            )
            plt.title("Estimated Mean and Confidence Regions")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.show()

        if "clusters" in what:
            plt.figure(figsize=(8, 6))
            if X.ndim == 1:
                plt.scatter(X, Y, c="black", s=10, label="Data")
                for k in range(self.param.K):
                    indices = self.stat.klas.flatten() == k + 1
                    plt.scatter(
                        X[indices],
                        Y[indices],
                        c=[colors[k]],
                        label=f"Cluster {k + 1}",
                        s=10
                    )
                    plt.plot(X, self.stat.Ey_k[:, k], c=colors[k], linestyle="--", lw=2,
                             label=f"Expert {k + 1} mean")
                plt.xlabel("X")
                plt.ylabel("Y")
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='viridis', s=10, label="Data")
                plt.title("Clusters in PCA Projection")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
            plt.title("Clusters Visualization")
            plt.legend()
            plt.show()

        if "loglikelihood" in what:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(self.stat.stored_loglik) + 1), self.stat.stored_loglik, c="blue", lw=2)
            plt.title("Log-Likelihood Over EM Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Log-Likelihood")
            plt.show()

    def summary(self):
        """
        Print a summary of the fitted NMoE model, including log-likelihood,
        AIC, BIC, ICL, regression coefficients, and variances.
        """
        print("Fitted Normal Mixture-of-Experts Model")
        print("=" * 50)

        print(f"NMoE model with K = {self.param.K} expert(s):\n")

        print(f"Log-Likelihood: {self.stat.loglik:.8f}")
        print(f"Degrees of Freedom (df): {self.param.df}")
        print(f"AIC: {self.stat.AIC:.8f}")
        print(f"BIC: {self.stat.BIC:.8f}")
        print(f"ICL: {self.stat.ICL:.8f}\n")

        print("Clustering Table (Number of Observations in Each Expert):")
        unique, counts = np.unique(self.stat.klas, return_counts=True)
        for k, count in zip(unique, counts):
            print(f"Expert {int(k)}: {count} observations")
        print()

        print("Regression Coefficients:")
        coef_labels = ["1"] + [f"X_{i}" for i in range(1, self.param.p + self.param.d)]
        for k in range(self.param.K):
            print(f"Expert {k + 1}:")
            for i, coef in enumerate(self.param.beta[:, k]):
                print(f"  {coef_labels[i]}: {coef:.8f}")
        print()

        print("Variances:")
        for k, var in enumerate(self.param.sigma2.flatten()):
            print(f"Expert {k + 1}: {var:.8f}")
        self.output_gating_coefficients()
        print()

    def output_gating_coefficients(self):
        """
        Output the gating coefficients (alpha) for the gating network.
        """
        self.param.output_gating_coefficients()

    def output_to_file(self, filename):
        """
        Output the fitted NMoE model to a file.

        Parameters:
        - filename: Name of the file to save the model.
        """
        with open(filename, "w") as f:
            f.write("Fitted Normal Mixture-of-Experts Model\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"NMoE model with K = {self.param.K} expert(s):\n\n")

            f.write(f"Log-Likelihood: {self.stat.loglik:.8f}\n")
            f.write(f"Degrees of Freedom (df): {self.param.df}\n")
            f.write(f"AIC: {self.stat.AIC:.8f}\n")
            f.write(f"BIC: {self.stat.BIC:.8f}\n")
            f.write(f"ICL: {self.stat.ICL:.8f}\n\n")

            f.write("Clustering Table (Number of Observations in Each Expert):\n")
            unique, counts = np.unique(self.stat.klas, return_counts=True)
            for k, count in zip(unique, counts):
                f.write(f"Expert {int(k)}: {count} observations\n")
            f.write("\n")

            f.write("Regression Coefficients:\n")
            coef_labels = ["1"] + [f"X_{i}" for i in range(1, self.param.p + self.param.d)]
            for k in range(self.param.K):
                f.write(f"Expert {k + 1}:\n")
                for i, coef in enumerate(self.param.beta[:, k]):
                    f.write(f"  {coef_labels[i]}: {coef:.8f}\n")
            f.write("\n")

            f.write("Variances:\n")
            for k, var in enumerate(self.param.sigma2.flatten()):
                f.write(f"Expert {k + 1}: {var:.8f}\n")
            f.write("\n")

            f.write("Gating Coefficients (alpha):\n")
            for k in range(self.param.K - 1):
                f.write(f"Expert {k + 1}:\n")
                for i, coeff in enumerate(self.param.alpha[:, k]):
                    term = f"X_{i}" if i > 0 else "1"
                    f.write(f"  {term}: {coeff:.8f}\n")
            f.write("\n")

        print(f"Model summary saved to '{filename}'.")


    def output_to_json(self, filename):
        """
        Output the fitted NMoE model to a JSON file.

        Parameters:
        - filename: Name of the file to save the model.
        """
        data = {
            "K": self.param.K,
            "p": self.param.p,
            "d": self.param.d,
            "AIC": self.stat.AIC,
            "BIC": self.stat.BIC,
            "ICL": self.stat.ICL,
            "loglik": self.stat.loglik,
            "df": self.param.df,
            "Regression Coefficients": self.param.beta.tolist(),
            "Variances": self.param.sigma2.tolist(),
            "Gating Coefficients": self.param.alpha.tolist(),
            "Clustering": self.stat.klas.tolist(),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Model data saved to '{filename}'.")

    def load_from_json(self, filename):
        """
        Load a fitted NMoE model from a JSON file.

        Parameters:
        - filename: Name of the JSON file containing the model data.
        """
        with open(filename, "r") as f:
            data = json.load(f)

        temp_param = ParamNMoE(K=data["K"], p=data["p"], q= data["d"] )
        temp_stat = StatNMoE(temp_param)
        temp_param.beta = np.array(data["Regression Coefficients"])
        temp_param.sigma2 = np.array(data["Variances"])
        temp_param.alpha = np.array(data["Gating Coefficients"])
        temp_stat.klas = np.array(data["Clustering"])
        temp_stat.AIC = data["AIC"]
        temp_stat.BIC = data["BIC"]
        temp_stat.ICL = data["ICL"]
        temp_stat.loglik = data["loglik"]
        temp_param.df = data["df"]
        self.param = temp_param
        self.stat = temp_stat
        return

    def list_of_parameters(self):
        """
        Return a list of all parameters in the model.
        """
        return {
            "K": self.param.K,
            "p": self.param.p,
            "d": self.param.d,
            "AIC": self.stat.AIC,
            "BIC": self.stat.BIC,
            "ICL": self.stat.ICL,
            "loglik": self.stat.loglik,
            "df": self.param.df,
            "Regression Coefficients": self.param.beta.tolist(),
            "Variances": self.param.sigma2.tolist(),
            "Gating Coefficients": self.param.alpha.tolist(),
            "Clustering": self.stat.klas.tolist(),
        }
