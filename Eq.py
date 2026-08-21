#weduofficial
import math
import numpy as np


# =====================================================================
# 1. Iterative Summation Version
#    Evaluates: (( sum_{n=0}^K (-1)^(n+K) * binom(K, n) * A[n+g] )_{K=0}^{T-g})_{g=0}^{T}
# =====================================================================
def calculate_reverse_triangle_iterative(A):
    """Evaluates the nested summation equation iteratively.

    Parameters:
        A (list or array): Input array of pyramid items before differences.

    Returns:
        dict: A dictionary with keys 'results', 'T', and 'terms_needed'.
    """
    T = len(A) - 1
    if T < 0:
        raise ValueError("Input array A must contain at least one item.")

    terms_needed = math.comb(T + 3, 3)  # Tetrahedral number
    outer_results = []

    # Outer loop: g from 0 to T
    for g in range(T + 1):
        inner_results = []

        # Middle loop: K from 0 to T - g
        for K in range(T - g + 1):

            # Inner summation: n from 0 to K
            sum_val = 0
            for n in range(K + 1):
                sign = (-1) ** (n + K)
                binom_coeff = math.comb(K, n)
                term = sign * binom_coeff * A[n + g]
                sum_val += term

            inner_results.append(sum_val)

        outer_results.append(inner_results)

    return {
        "results": outer_results,
        "T": T,
        "terms_needed": terms_needed,
    }


# =====================================================================
# 2. Matrix Multiplication Version
#    Evaluates: ((P_g * M * P_g^T) * (Q_g * L))_{g=0}^T
# =====================================================================
def calculate_reverse_triangle_matrix(A):
    """Evaluates the matrix formulation of the reverse triangle equation.

    Parameters:
        A (list or array): Input array of pyramid items before differences.

    Returns:
        dict: A dictionary with keys 'results', 'T', and 'terms_needed'.
    """
    T = len(A) - 1
    if T < 0:
        raise ValueError("Input array A must contain at least one item.")

    # 1. Base Matrix M of size (T + 1) x (T + 1)
    M = np.zeros((T + 1, T + 1), dtype=int)
    for k in range(T + 1):
        for n in range(k + 1):
            M[k, n] = ((-1) ** (n + k)) * math.comb(k, n)

    # 2. Vector L of size (T + 1) x 1
    L = np.array(A, dtype=int).reshape((T + 1, 1))

    outer_results = []

    # Outer loop over shift index g
    for g in range(T + 1):
        dim = T + 1 - g  # Current dimension (T + 1 - g)

        # 3. Projection Matrix P_g of size (dim) x (T + 1)
        P_g = np.zeros((dim, T + 1), dtype=int)
        for i in range(dim):
            P_g[i, i] = 1

        P_g_T = P_g.T  # Transpose of P_g, size (T + 1) x dim

        # 4. Selection/Shift Matrix Q_g of size (dim) x (T + 1)
        Q_g = np.zeros((dim, T + 1), dtype=int)
        for i in range(dim):
            Q_g[i, i + g] = 1

        # Matrix multiplications:
        # Step A: Reduced M -> (P_g @ M @ P_g_T)
        M_reduced = P_g @ M @ P_g_T

        # Step B: Shifted L -> (Q_g @ L)
        L_shifted = Q_g @ L

        # Step C: Result Vector -> M_reduced @ L_shifted
        res_vector = M_reduced @ L_shifted

        outer_results.append(res_vector.flatten().tolist())

    terms_needed = math.comb(T + 3, 3)

    return {
        "results": outer_results,
        "T": T,
        "terms_needed": terms_needed,
    }


# =====================================================================
# Execution & Verification
# =====================================================================
if __name__ == "__main__":
    A_input = [10, 20, 35, 60]

    # Compute using both algorithms
    res_iterative = calculate_reverse_triangle_iterative(A_input)
    res_matrix = calculate_reverse_triangle_matrix(A_input)

    print(f"Input Array A: {A_input}")
    print(f"T (len(A) - 1): {res_iterative['T']}")
    print(
        f"Amount of terms needed (Tetrahedral Number): {res_iterative['terms_needed']}"
    )

    print("\n--- Iterative Version Result ---")
    for row in res_iterative["results"]:
        print(row)

    print("\n--- Matrix Version Result ---")
    for row in res_matrix["results"]:
        print(row)

    # Verification check
    is_match = res_iterative["results"] == res_matrix["results"]
    print(f"\nDo both implementations match identically? -> {is_match}")
