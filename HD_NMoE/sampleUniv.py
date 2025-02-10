import numpy as np
from scipy.stats import multinomial, norm
from multinomial import multinomialLogit
import json

def design_matrix(X):
    """
    Generate a design matrix of degree 1 (original features + intercept).

    Parameters:
        X (np.ndarray): Input array of shape (n_samples, n_features).

    Returns:
        np.ndarray: Design matrix of shape (n_samples, n_features + 1).
    """

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # Add intercept (column of ones)
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

'''def multinomial_logit(alphak, XAlpha, zeros, ones):
    logits = np.dot(XAlpha, alphak)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    piik = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return {"piik": piik}
    '''

def sample_univ_nmoe(alphak, betak, sigmak, X):
    n, n_features = X.shape
    K = betak.shape[1]

    # Build the regression design matrices
    XBeta = design_matrix(X)
    XAlpha = design_matrix(X)

    y = np.zeros(n)
    z = np.zeros((n, K))
    zi = np.zeros(n, dtype=int)

    # Calculate the mixing proportions piik
    piik = multinomialLogit(alphak, XAlpha, np.zeros((n, K)), np.ones((n, 1)))["piik"]

    for i in range(n):
        zik = multinomial.rvs(n=1, p=piik[i, :])

        mu = np.dot(XBeta[i, :], betak[:, zik.argmax()])
        sigma = sigmak[zik.argmax()]

        y[i] = norm.rvs(loc=mu, scale=sigma)
        z[i, :] = zik
        zi[i] = zik.argmax() + 1

    # Statistics (means, variances)
    Ey_k = np.dot(XBeta, betak)
    Ey = np.sum(piik * Ey_k, axis=1)
    Vary_k = sigmak ** 2
    Vary = np.sum(piik * (Ey_k ** 2 + Vary_k), axis=1) - Ey ** 2

    stats = {
        "Ey_k": Ey_k,
        "Ey": Ey,
        "Vary_k": Vary_k,
        "Vary": Vary
    }

    return {"y": y, "zi": zi, "z": z, "stats": stats}


# # Generate 2D input X and parameters
# np.random.seed(42)
# n_samples = 10000
# n_features = 2
# n_components = 5
#
# # Generate X from a normal distribution
# X = np.random.normal(0, 1, size=(n_samples, n_features))
#
# # Define parameters
# alphak = np.array([ [ 0.3,  0.5, -0.2 ,-0.1],
#                     [ 1 ,   1 , 0.5 , 0.3],
#                     [-1,    0.5 , 0.3 , -1.2],
#                     ])
# betak = np.array([  [ 0.5  ,0.5 , 0.5 , -0.6 , 0.3],
#                     [ 1.5,  0.6,  1.1 , 1.5 , 0.2],
#                     [ 0.9,  0.5, -0.3 , -0.1 , 0.4]
#                     ])
# sigmak = np.array([0.9, 0.5, 1.2 , 0.7 , 0.5])
#
#
# # Flatten X for the sampling function
# #flattened_X = X.flatten()
#
# # Sample data
# data = sample_univ_nmoe(alphak, betak, sigmak, X)
#
# # Save the data to a JSON file
# output_path = "../data/output_data_2D_2.json"
#
# '''output_data = {
#     "X": X.tolist(),
#     "alphak": alphak.tolist(),
#     "betak": betak.tolist(),
#     "sigmak": sigmak.tolist(),
#     "y": data["y"].tolist(),
#     "zi": data["zi"].tolist(),
#     "z": data["z"].tolist(),
#     "stats": {
#         "Ey_k": data["stats"]["Ey_k"].tolist(),
#         "Ey": data["stats"]["Ey"].tolist(),
#         "Vary_k": data["stats"]["Vary_k"].tolist(),
#         "Vary": data["stats"]["Vary"].tolist()
#     }
# }'''
#
#
# output_data = {
#     "X": X.tolist(),
#     "y": data["y"].tolist(),
# }
#
# with open(output_path, "w") as f:
#     json.dump(output_data, f, indent=4)
