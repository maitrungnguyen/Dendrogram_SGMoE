import numpy as np
from multinomial import multinomialLogit

class StatNMoE:
    def __init__(self, paramNMoE):
        n, K = paramNMoE.n, paramNMoE.K
        self.n = n
        self.K = K

        self.piik = np.full((n, K), np.nan)  # Probabilities
        self.z_ik = np.full((n, K), np.nan)  # Hard segmentation matrix
        self.klas = np.full((n, 1), np.nan)  # Clustering labels
        self.Ey_k = np.full((n, K), np.nan)  # Expected mean for each expert
        self.Ey = np.full((n, 1), np.nan)    # Expected overall mean
        self.Var_yk = np.full(K, np.nan)     # Variance for each expert
        self.Vary = np.full((n, 1), np.nan)  # Overall variance
        self.loglik = -np.inf                # Log-likelihood
        self.com_loglik = -np.inf            # Complete-data log-likelihood
        self.stored_loglik = []              # Log-likelihood history
        self.BIC = -np.inf                   # Bayesian Information Criterion
        self.ICL = -np.inf                   # Integrated Completed Likelihood
        self.AIC = -np.inf                   # Akaike Information Criterion
        self.log_piik_fik = np.zeros((n, K)) # Log joint probabilities
        self.log_sum_piik_fik = np.full((n, 1), np.nan)
        self.tik = np.zeros((n, K))          # Posterior probabilities (responsibilities)

    def MAP(self):
        """
        Perform Maximum A Posteriori (MAP) to compute z_ik and klas.
        """
        # Find the most probable expert for each observation
        ikmax = np.argmax(self.tik, axis=1)
        self.z_ik = np.eye(self.K)[ikmax]  # Hard segmentation
        self.klas = ikmax + 1              # Cluster labels (1-based)

    def computeLikelihood(self, reg_irls):
        """
        Compute the log-likelihood using the given regularization value from IRLS.
        """
        self.loglik = np.sum(self.log_sum_piik_fik) + reg_irls

    def computeStats(self, paramNMoE):
        """
        Compute BIC, AIC, ICL for the EM algorithm based on the parameters in ParamNMoE.
        """
        n, K = self.n, self.K


        self.Ey_k = paramNMoE.phiBeta @ paramNMoE.beta

        self.Ey = np.sum(self.piik * self.Ey_k, axis=1, keepdims=True)

        self.Var_yk = paramNMoE.sigma2.flatten()

        self.Vary = (
            np.sum(self.piik * (self.Ey_k**2 + self.Var_yk), axis=1, keepdims=True)
            - self.Ey**2
        )

        self.BIC = self.loglik - (paramNMoE.df * np.log(n) / 2)
        self.AIC = self.loglik - paramNMoE.df

        zik_log_piik_fk = self.z_ik * self.log_piik_fik
        sum_zik_log_fik = np.sum(zik_log_piik_fk, axis=1)
        self.com_loglik = np.sum(sum_zik_log_fik)

        self.ICL = self.com_loglik - (paramNMoE.df * np.log(n) / 2)

    def compute_AIC_BIC_loglik(self, paramNMoE):
        """
        Compute the AIC, BIC, and log-likelihood for the current model.
        """
        n, K = self.n, self.K

        self.AIC = -2 * self.loglik + 2 * paramNMoE.df
        self.BIC = -2 * self.loglik + paramNMoE.df * np.log(n)


    def EStep(self, paramNMoE):
        """
        Perform the E-step of the EM algorithm.
        """
        n, K = self.n, self.K

        # Compute piik using multinomial logistic regression
        self.piik = multinomialLogit(
            paramNMoE.alpha,
            paramNMoE.phiAlpha,
            np.ones((n, K)),
            np.ones((n, 1)),
            True,
        )["piik"]

        for k in range(K):
            muk = paramNMoE.phiBeta @ paramNMoE.beta[:, k]
            sigma2k = paramNMoE.sigma2[k]

            self.log_piik_fik[:, k] = (
                np.log(self.piik[:, k])
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * np.log(sigma2k)
                - 0.5 * ((paramNMoE.Y - muk)**2) / sigma2k
            )

        max_log = np.max(self.log_piik_fik, axis=1, keepdims=True)
        self.log_sum_piik_fik = max_log + np.log(
            np.sum(np.exp(self.log_piik_fik - max_log), axis=1, keepdims=True)
        )

        log_tik = self.log_piik_fik - self.log_sum_piik_fik
        self.tik = np.exp(log_tik)

    def log_likelihood(self):
        """
        Compute the log-likelihood of the model.
        """
        res = multinomialLogit(
            self.paramNMoE.alpha,
            self.paramNMoE.phiAlpha,
            self.tik,
            np.ones((self.n, 1)),
            True,
        )
