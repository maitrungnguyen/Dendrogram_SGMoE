import numpy as np
from IRLS import IRLS
#import IRLS_Cpp

class ParamNMoE:
    """
    Class representing the parameters of a Normal Mixture of Experts (NMoE) model.
    """

    def __init__(self, X=np.array([]), Y=np.array([]), K=1, p=3, q=1):
        """
        Initialize the ParamNMoE object.

        Parameters:
        - X: Input features .
        - Y: Observed responses .
        - K: Number of experts.
        - p: Order of the polynomial regression for the experts.
        - q: Order of the logistic regression for the gating network.
        """
        self.X = X
        self.Y = Y
        self.n = len(Y)
        self.d = X.shape[1] if X.ndim > 1 else 1

        self.K = K
        self.p = p
        self.q = q

        # Design matrices for experts and gating network
        self.phiBeta = self.design_matrix(X)
        self.phiAlpha = self.design_matrix(X)
        self.df = (q + 1 + self.d - 1) * (K - 1) + (p + 1 + self.d - 1) * K + K

        # Initialize parameters
        self.alpha = np.zeros((q + 1 + self.d -1, K - 1))  # Gating network parameters
        self.beta = np.full((p + 1 + self.d - 1, K), np.nan)  # Experts' regression coefficients
        self.sigma2 = np.full(K, np.nan)  # Variances of the experts

    def check(self):
        print("Check {}", self.phiAlpha)
        print(self.phiBeta)

    @staticmethod
    def design_matrix(X):
        """
        Generate a design matrix of degree 1 (original features + intercept).

        Parameters:
            X (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Design matrix of shape (n_samples, n_features + 1).
        """

        # Ensure X is 2D
        #print(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        #print(X)
        # Add intercept (column of ones)
        intercept = np.ones((X.shape[0], 1))
        #print("Design Matrix", intercept)
        return np.hstack((intercept, X))

    def init_param(self, segmental=False):
        """
        Initialize the parameters alpha, beta, and sigma2.

        Parameters:
        - segmental: If True, use segmental initialization; otherwise, use random initialization.
        """
        if not segmental:
            # Random initialization of clusters
            klas = np.random.choice(self.K, self.n)

            for k in range(self.K):
                mask = klas == k
                Xk = self.phiBeta[mask]
                yk = self.Y[mask]

                if len(yk) == 0:  # Safeguard against empty clusters
                    continue

                self.beta[:, k] = np.linalg.pinv(Xk.T @ Xk) @ (Xk.T @ yk)
                self.sigma2[k] = np.mean((yk - Xk @ self.beta[:, k]) ** 2)
        else:
            # Segmental initialization
            nk = self.n // self.K
            klas = np.zeros(self.n, dtype=int)

            for k in range(self.K):
                i, j = k * nk, min((k + 1) * nk, self.n)
                Xk = self.phiBeta[i:j]
                yk = self.Y[i:j]

                self.beta[:, k] = np.linalg.pinv(Xk.T @ Xk) @ (Xk.T @ yk)
                muk = Xk @ self.beta[:, k]
                self.sigma2[k] = np.mean((yk - muk) ** 2)
                klas[i:j] = k

        # Initialize gating network parameters using IRLS
        Z = np.zeros((self.n, self.K))
        Z[np.arange(self.n), klas] = 1  # Hard assignment
        tau = Z
        res = IRLS(self.phiAlpha, tau, np.ones((self.n, 1)), self.alpha)
        #res = IRLS_Cpp.IRLS(self.phiAlpha, tau, np.ones((self.n, 1)), self.alpha)
        self.alpha = res["W"]

    def M_step(self, statNMoE, verbose_IRLS=False):
        """
        Perform the M-step of the EM algorithm to update parameters.

        Parameters:
        - statNMoE: StatNMoE object containing the statistics of the NMoE model.
        - verbose_IRLS: If True, print details during IRLS optimization.

        Returns:
        - reg_irls: Regularization value from IRLS.
        """
        # Update gating network parameters using IRLS

        res_irls = IRLS(self.phiAlpha, statNMoE.tik, np.ones((self.n, 1)), self.alpha, verbose_IRLS)
        statNMoE.piik = res_irls["piik"]  # Update piik in the StatNMoE object
        reg_irls = res_irls["reg_irls"]
        self.alpha = res_irls["W"]

        # Update regression parameters for each expert
        for k in range(self.K):
            # Weighted design matrix and response
            weights = np.sqrt(statNMoE.tik[:, k])
            Xbeta = self.phiBeta * weights[:, np.newaxis]
            yk = self.Y * weights

            self.beta[:, k] = np.linalg.pinv(Xbeta.T @ Xbeta) @ (Xbeta.T @ yk)

            # Update variance
            residuals = self.Y - self.phiBeta @ self.beta[:, k]
            self.sigma2[k] = np.sum(statNMoE.tik[:, k] * residuals**2) / np.sum(statNMoE.tik[:, k])

        return reg_irls

    def output_gating_coefficients(self):
        """
        Output the gating coefficients (alpha) for the gating network.
        """
        print("Gating Coefficients (alpha):")
        for k in range(self.K - 1):
            print(f"Expert {k + 1}:")
            for i, coeff in enumerate(self.alpha[:, k]):
                term = f"X_{i}" if i > 0 else "1"
                print(f"  {term}: {coeff:.4f}")

    def aic(self):
        """
        Calculate the Akaike Information Criterion (AIC) for the model.
        """
        return -2 * self.loglik + 2 * self.df

    def bic(self):
        """
        Calculate the Bayesian Information Criterion (BIC) for the model.
        """
        return -2 * self.loglik + self.df * np.log(self.n)


    def init_param_2(self, segmental=False):
        """
        Initialize the parameters alpha, beta, and sigma2.

        Parameters:
        - segmental: If True, use segmental initialization; otherwise, use random initialization.
        """
        if not segmental:
            # Random initialization of clusters
            klas = np.random.choice(self.K, self.n)

            for k in range(self.K):
                mask = klas == k
                Xk = self.phiBeta[mask]
                yk = self.Y[mask]

                if len(yk) == 0:  # Safeguard against empty clusters
                    continue

                #self.beta[:, k] = np.linalg.pinv(Xk.T @ Xk) @ (Xk.T @ yk)
                #self.sigma2[k] = np.mean((yk - Xk @ self.beta[:, k]) ** 2)
        else:
            # Segmental initialization
            nk = self.n // self.K
            klas = np.zeros(self.n, dtype=int)

            for k in range(self.K):
                i, j = k * nk, min((k + 1) * nk, self.n)
                Xk = self.phiBeta[i:j]
                yk = self.Y[i:j]

                self.beta[:, k] = np.linalg.pinv(Xk.T @ Xk) @ (Xk.T @ yk)
                muk = Xk @ self.beta[:, k]
                #self.sigma2[k] = np.mean((yk - muk) ** 2)
                klas[i:j] = k

        # Initialize gating network parameters using IRLS
        Z = np.zeros((self.n, self.K))
        Z[np.arange(self.n), klas] = 1  # Hard assignment
        tau = Z
        res = IRLS(self.phiAlpha, tau, np.ones((self.n, 1)), self.alpha)
        #res = IRLS_Cpp.IRLS(self.phiAlpha, tau, np.ones((self.n, 1)), self.alpha)
        self.alpha = res["W"]