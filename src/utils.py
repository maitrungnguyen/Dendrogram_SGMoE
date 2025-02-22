import numpy as np

def designmatrix(x, p, q=None, n=1):
    """
    Generate the design matrices for polynomial regression and logistic regression.
    """
    order_max = p
    if q is not None:
        order_max = max(p, q)

    # Create the base matrix
    X = np.column_stack([x ** i for i in range(order_max + 1)])

    # Design matrix for Beta (polynomial regression)
    XBeta = X[:, :p + 1]
    XBeta = np.tile(XBeta, (n, 1))

    # Design matrix for Alpha (logistic regression)
    if q is not None:
        Xw = X[:, :q + 1]
        Xw = np.tile(Xw, (n, 1))
    else:
        Xw = None

    return {"Xw": Xw, "XBeta": XBeta}

def ones(n, d, g=1):
    """
    Create a matrix or array filled with ones.
    """
    if g == 1:
        return np.ones((n, d))
    else:
        return np.ones((n, d, g))

def zeros(n, d, g=1):
    """
    Create a matrix or array filled with zeros.
    """
    if g == 1:
        return np.zeros((n, d))
    else:
        return np.zeros((n, d, g))

def rand(n, d, g=1):
    """
    Create a matrix or array with random values from a uniform distribution.
    """
    if g == 1:
        return np.random.rand(n, d)
    else:
        return np.random.rand(n, d, g)

def repmat(M, n, d):
    """
    Replicate a matrix along rows and columns.
    """
    return np.tile(M, (n, d))

def drnorm(n, d, mean, sd):
    """
    Draw random samples from a normal distribution.
    """
    return np.random.normal(mean, sd, (n, d))

def lognormalize(M):
    """
    Perform log normalization on a matrix.
    """
    if not isinstance(M, np.ndarray):
        M = np.array(M)

    n, d = M.shape
    a = np.max(M, axis=1)
    return M - repmat(a + np.log(np.sum(np.exp(M - repmat(a[:, None], 1, d)), axis=1)), 1, d)

def normalize(A, dim=None):
    """
    Normalize an array to sum to 1 along a specified dimension.
    """
    if dim is None:
        z = np.sum(A)
        s = z + (z == 0)  # Prevent division by zero
        M = A / s
    elif dim == 1:  # Normalize each column
        z = np.sum(A, axis=0)
        s = z + (z == 0)
        M = A / s[None, :]
    elif dim == 2:  # Normalize each row
        z = np.sum(A, axis=1)
        s = z + (z == 0)
        M = A / s[:, None]

    return {"M": M, "z": z}
