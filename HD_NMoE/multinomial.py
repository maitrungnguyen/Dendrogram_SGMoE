import numpy as np

def multinomialLogit(W, X, Y, Gamma, verbose=False, verboseIterator=0):
    """
    Multinomial logistic regression model.

    Parameters:
    - W: Parameters of the logistic model (shape: (q+1, K-1)).
    - X: Design matrix for the logistic model (shape: (n, q+1)).
    - Y: Fuzzy partition matrix (posterior probabilities) (shape: (n, K)).
    - Gamma: Cluster weights (shape: (n, 1)).

    Returns:
    - loglik: Log-likelihood of the logistic model parameters.
    - piik: Probabilities for each class (shape: (n, K)).
    """
    n, q = X.shape
    _, K = Y.shape

    # Validate dimensions
    if q != W.shape[0]:
        raise ValueError("W must have q + 1 rows and X must have q + 1 columns.")

    # Add the null vector for the last class if W has only K-1 columns
    if W.shape[1] == (K - 1):
        Wc = np.hstack([W, np.zeros((q, 1))])
    elif W.shape[1] == K:
        Wc = W
    else:
        raise ValueError("W must have K-1 or K columns.")

    if (n != Y.shape[0]) or (n != Gamma.shape[0]):
        raise ValueError("X, Y, and Gamma must have the same number of rows (n).")
    #print("X: ",X)
    #print("W: ",Wc)
    # Calculate XW (linear transformation)
    XW = X @ Wc

    # Numerical stability: Subtract max value in each row
    maxm = np.max(XW, axis=1, keepdims=True)

    XW = XW - maxm
    # Avoid overflow and underflow by clamping values
    XW = np.clip(XW, -745.1, 709.78)  # Safe range for exp()

    # Calculate exp(XW)
    expXW = np.exp(XW)

    # Probabilities for each class
    piik = expXW / np.sum(expXW, axis=1, keepdims=True)

    # Calculate Gamma matrix for element-wise multiplication
    GammaMat = Gamma * np.ones((1, K))

    # Log-likelihood computation
    loglik = np.sum(
        GammaMat * (Y * XW) - (GammaMat * Y) * np.log(np.sum(expXW, axis=1, keepdims=True))
    )
    #print("Prob ik: ",piik)
    #exit(-1)
    return {"loglik": loglik, "piik": piik}
