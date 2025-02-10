import numpy as np
from multinomial import multinomialLogit


def IRLS(X, Tau, Gamma, Winit, verbose=False, max_iter=300, lambda_reg=1e-9):
    """
    Iteratively Reweighted Least Squares (IRLS) for multinomial logistic regression.

    Parameters:
    - X: Design matrix for the logistic weights (n x q).
    - Tau: Hard or fuzzy partition of the data (posterior probabilities) (n x K).
    - Gamma: Cluster weights (hard or smooth) (n x 1).
    - Winit: Initial parameter matrix (q x K-1).
    - verbose: Whether to print intermediate log-likelihoods.
    - max_iter: Maximum number of iterations.
    - lambda_reg: Regularization parameter for L2 penalty.

    Returns:
    - W: Estimated parameter matrix (q x K-1).
    - piik: Logistic probabilities (n x K).
    - loglik: Log-likelihood of the logistic model.
    - LL: List of log-likelihood values during training.
    - reg_irls: Regularization value.
    """
    n, K = Tau.shape
    q = X.shape[1]  # q here is the number of predictors
    #print(X)
    #print(Tau)
    #print(Winit)
    I = np.eye(q * (K - 1))  # Identity matrix for regularization
    W_old = Winit
    piik_old = np.zeros((n, K))
    LL = []

    # Initial computation of log-likelihood and probabilities
    result = multinomialLogit(W_old, X, Tau, Gamma)
    loglik_old = result["loglik"] - lambda_reg * np.linalg.norm(W_old)**2
    piik_old = result["piik"]
    #print("piik_old",piik_old)
    iter = 0
    converge = False

    if verbose:
        print(f"IRLS: Iteration {iter}, Log-likelihood: {loglik_old}")
        #print(W_old)

    while not converge and iter < max_iter:
        # Gradient computation
        gw_old = np.zeros_like(W_old)
        for k in range(K - 1):
            gwk = Gamma[:, 0] * (Tau[:, k] - piik_old[:, k])
            for qq in range(q):
                gw_old[qq, k] = np.dot(gwk, X[:, qq])

        # Hessian computation
        Hw_old = np.zeros((q * (K - 1), q * (K - 1)))
        for k in range(K - 1):
            for ell in range(K - 1):
                delta_kl = 1 if k == ell else 0
                gwk = Gamma[:, 0] * piik_old[:, k] * (delta_kl - piik_old[:, ell])
                for qqa in range(q):
                    for qqb in range(q):
                        Hw_old[k * q + qqa, ell * q + qqb] -= np.dot(X[:, qqa] * gwk, X[:, qqb])
                        #print(X[:, qqa])
                        #print(X[:, qqb])
                        #exit(1)

        # Regularization
        Hw_old += lambda_reg * I
        gw = gw_old.flatten() - lambda_reg * W_old.flatten()

        # Newton-Raphson update
        #print("W_old",W_old)
        #print("Hw_old",Hw_old)
        #print("gw",gw)
        w_new = W_old.flatten() - np.linalg.solve(Hw_old, gw)
        W = w_new.reshape(W_old.shape)

        # Update probabilities and log-likelihood
        result = multinomialLogit(W, X, Tau, Gamma)
        loglik = result["loglik"] - lambda_reg * np.linalg.norm(W)**2
        piik = result["piik"]

        # Step-size adaptation if log-likelihood decreases
        stepsize = 1
        while loglik < loglik_old:
            stepsize /= 2
            w_new = W_old.flatten() - stepsize * np.linalg.solve(Hw_old, gw)
            W = w_new.reshape(W_old.shape)
            result = multinomialLogit(W, X, Tau, Gamma)
            loglik = result["loglik"] - lambda_reg * np.linalg.norm(W)**2
            piik = result["piik"]

        # Convergence criteria
        rel_change = np.abs((loglik - loglik_old) / loglik_old)
        abs_change = np.abs(loglik - loglik_old)
        converge = rel_change <= 1e-7 or abs_change <= 1e-6

        # Update old values
        W_old = W
        piik_old = piik
        loglik_old = loglik
        LL.append(loglik)
        iter += 1

        if verbose:
            print(f"IRLS: Iteration {iter}, Log-likelihood: {loglik}")
            #print(W)

    if verbose:
        if converge:
            print(f"IRLS converged in {iter} iterations.")
        else:
            print(f"IRLS did not converge within {max_iter} iterations.")

    reg_irls = -lambda_reg * np.linalg.norm(W)**2
    return {"W": W, "piik": piik, "loglik": loglik, "LL": LL, "reg_irls": reg_irls}