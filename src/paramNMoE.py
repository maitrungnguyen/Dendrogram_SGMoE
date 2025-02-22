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
        - X: Input features (1D array).
        - Y: Observed responses (1D array).
        - K: Number of experts.
        - p: Order of the polynomial regression for the experts.
        - q: Order of the logistic regression for the gating network.
        """
        self.X = X
        self.Y = Y
        self.n = len(Y)
        self.K = K
        self.p = p
        self.q = q

        # Design matrices for experts and gating network
        self.phiBeta = self.design_matrix(X, p)
        self.phiAlpha = self.design_matrix(X, q)

        self.df = (q + 1) * (K - 1) + (p + 1) * K + K

        # Initialize parameters
        self.alpha = np.zeros((q + 1, K - 1))  # Gating network parameters
        self.beta = np.full((p + 1, K), np.nan)  # Experts' regression coefficients
        self.sigma2 = np.full(K, np.nan)  # Variances of the experts

    @staticmethod
    def design_matrix(x, order):
        """
        Generate a design matrix for polynomial regression up to a given order.

        Parameters:
        - x: Input values (1D array).
        - order: Polynomial order.

        Returns:
        - Design matrix (2D array).
        """
        return np.vander(x, N=order + 1, increasing=True)

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
        '''
        print("self.phiAlpha", self.phiAlpha)
        print("tau", tau)
        print("np.ones((self.n, 1))", np.ones((self.n, 1)))
        print("self.alpha", self.alpha)'''
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
                term = f"X^{i}" if i > 0 else "1"
                print(f"  {term}: {coeff:.4f}")
