import numpy as np
from paramNMoE import ParamNMoE
from statNMoE import StatNMoE
from modelNMoE import ModelNMoE

def emNMoE(X, Y, K, p=3, q=1, n_tries=1, max_iter=1500, threshold=1e-6, verbose=False, verbose_IRLS=False):
    """
    EM algorithm for fitting a Normal Mixture of Experts (NMoE) model.
    Parameters:
    - X: 1D NumPy array of inputs (covariates).
    - Y: 1D NumPy array of responses (outputs).
    - K: Number of experts.
    - p: Order of polynomial regression for experts (default: 3).
    - q: Order of logistic regression for the gating network (default: 1).
    - n_tries: Number of EM runs (default: 1). The best solution is returned.
    - max_iter: Maximum number of iterations for the EM algorithm (default: 1500).
    - threshold: Convergence threshold for relative log-likelihood change (default: 1e-6).
    - verbose: If True, print log-likelihood values during iterations.
    - verbose_IRLS: If True, print IRLS criterion values during the M-step.
    Returns:
    - Fitted ModelNMoE object.
    """
    top = 0
    try_EM = 0
    best_loglik = -np.inf

    while try_EM < n_tries:
        try_EM += 1

        if n_tries > 1 and verbose:
            print(f"EM try number: {try_EM}")

        # Initialization
        param = ParamNMoE(X=X, Y=Y, K=K, p=p, q=q)
        param.init_param(segmental=False)

        stat = StatNMoE(paramNMoE=param)

        iter = 0
        converge = False
        prev_loglik = -np.inf

        while not converge and iter <= max_iter:
            # E-Step
            stat.EStep(param)

            # M-Step
            reg_irls = param.M_step(stat, verbose_IRLS)

            # Compute log-likelihood
            stat.computeLikelihood(reg_irls)

            iter += 1
            if verbose:
                print(f"EM NMoE: Iteration {iter} | log-likelihood: {stat.loglik}")

            if prev_loglik - stat.loglik > 1e-5:
                if verbose:
                    print(f"Warning: EM log-likelihood is decreasing from {prev_loglik} to {stat.loglik}!")
                top += 1
                if top > 20:
                    break

            if prev_loglik != -np.inf:
                converge = abs((stat.loglik - prev_loglik) / prev_loglik) <= threshold
            prev_loglik = stat.loglik

            stat.stored_loglik.append(stat.loglik)

        if stat.loglik > best_loglik:
            statSolution = stat
            paramSolution = param
            best_loglik = stat.loglik

        if n_tries > 1 and verbose:
            print(f"Max value of the log-likelihood: {stat.loglik}")

    statSolution.MAP()
    statSolution.computeStats(paramSolution)

    if n_tries > 1 and verbose:
        print(f"Max value of the log-likelihood: {statSolution.loglik}")

    return ModelNMoE(param=paramSolution, stat=statSolution)
