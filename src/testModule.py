import IRLS_Cpp
import numpy as np
import IRLS
import multinomial

def test_multinomialLogit():
    W = [[1.0, 0.5], [0.3, 0.2]]
    X = [[1.0, 2.0], [3.0, 4.0]]
    Y = [[0.6, 0.4], [0.7, 0.3]]
    Gamma = [[1.0, 0.0], [0.0, 1.0]]

    result = IRLS_Cpp.multinomialLogit(W, X, Y, Gamma)

    piik, loglik = result

    print("Probabilities (piik):")
    for row in piik:
        print(row)
    print(f"Log-likelihood: {loglik}")



def test_IRLS():
    # Test data
    X = [[1.0, 2.0], [3.0, 4.0]]
    Tau = [[0.5, 0.5], [0.3, 0.7]]  # Partition probabilities
    Gamma = [[1.0, 0.0], [0.0, 1.0]]  # Cluster weights
    Winit = [[0.1, 0.2], [0.3, 0.4]]  # Initial weights

    # Call the function
    result = IRLS_Cpp.IRLS(X, Tau, Gamma, Winit, verbose=True)

    # Print results
    print("Updated weights (W):")
    for row in result:
        print(row)

    # Simple assertions to check output shape and values
    assert len(result) == len(Winit)
    assert len(result[0]) == len(Winit[0])

    result2 = IRLS.IRLS(X, Tau, Gamma, Winit, verbose=True)

    print("Updated weights (W):")
    for row in result2:
        print(row)
    # Simple assertions to check output shape and values
    assert len(result2) == len(Winit)  # Number of rows should match Winit
    assert len(result2[0]) == len(Winit[0])  # Number of columns should match Winit


'''
if __name__ == "__main__":
    print("Testing multinomialLogit:")
    test_multinomialLogit()
    print("\nTesting IRLS:")
    test_IRLS()'''
