import numpy as np


def sampleUnivNMoE(alphak, betak, sigmak, X):
    """
    Generate a sample from a Normal Mixture of Experts (NMoE) model for 2D input.

    Parameters:
    - alphak: Gating network parameters (K-1 x 1)
    - betak: Regression coefficients (K x p, p is the number of features)
    - sigmak: Covariance matrix for each expert (K x p x p)
    - X: Input data (n x p, where n is the number of data points, and p is the number of features)

    Returns:
    - y: The predicted values based on the experts (n x 1)
    - gamma: The cluster weights (n x K)
    """
    n, p = X.shape  # n: number of samples, p: number of features (2 in this case)
    K = betak.shape[0]  # Number of experts (clusters)

    # Calculate gating network outputs (responsibilities)
    gamma = np.zeros((n, K))  # Cluster responsibilities (n x K)
    for k in range(K):
        # Here, we apply the logistic gating function for each expert
        gamma[:, k] = 1 / (1 + np.exp(-(np.dot(X, betak[k, :]) + alphak[k])))

    # Normalize gamma so each row sums to 1 (soft assignments)
    gamma /= gamma.sum(axis=1, keepdims=True)

    # Sample from each expert
    y = np.zeros(n)
    for k in range(K):
        # Multivariate Normal sampling for each expert (assuming a diagonal covariance for simplicity)
        expert_samples = np.random.multivariate_normal(betak[k], sigmak[k], size=n)
        y += gamma[:, k] * expert_samples[:, 0]  # Weighted sum of expert predictions

    return y, gamma


# Example usage with 2D input
alphak = np.array([0, 8])  # Gating network parameters (1D example, modify for multi-class)
betak = np.array([[0, -2.5], [0, 2.5]])  # Coefficients for 2 experts (2D coefficients for 2 features)
sigmak = np.array([np.eye(2), np.eye(2)])  # Covariance matrices for each expert (identity for simplicity)
X = np.random.randn(100, 2)  # 2D input data


y, gamma = sampleUnivNMoE(alphak, betak, sigmak, X)
print("Predicted Values:", y)
print("Cluster Weights:", gamma)
