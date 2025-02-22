import numpy as np
import matplotlib.pyplot as plt
from paramNMoE import ParamNMoE

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
        coef_labels = ["1"] + [f"X^{i}" for i in range(1, self.param.p + 1)]
        for k in range(self.param.K):
            print(f"Expert {k + 1}:")
            for i, coef in enumerate(self.param.beta[:, k]):
                print(f"  {coef_labels[i]}: {coef:.8f}")
        print()

        print("Variances:")
        for k, var in enumerate(self.param.sigma2.flatten()):
            print(f"Expert {k + 1}: {var:.8f}")
        print()

    def output_gating_coefficients(self):
        """
        Output the gating coefficients (alpha) for the gating network.
        """
        self.param.output_gating_coefficients()
