import numpy as np

# Fungsi untuk melakukan dekomposisi LU Gauss
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Inisialisasi matriks L dengan diagonal 1
    for i in range(n):
        L[i][i] = 1
    
    for k in range(n):
        # Bagian atas matriks U
        for j in range(k, n):
            U[k][j] = A[k][j] - sum(L[k][p] * U[p][j] for p in range(k))
        # Bagian bawah matriks L
        for i in range(k+1, n):
            L[i][k] = (A[i][k] - sum(L[i][p] * U[p][k] for p in range(k))) / U[k][k]
    
    return L, U

# Fungsi untuk mencari solusi dengan dekomposisi LU Gauss
def solve_with_lu_gauss(A, b):
    L, U = lu_decomposition(A)
    
    # Menyelesaikan Ly = Pb
    y = np.linalg.solve(L, b)
    
    # Menyelesaikan Ux = y
    x = np.linalg.solve(U, y)
    
    return x

# Contoh sistem persamaan linear
A = np.array([[2, 1, -1], [4, 1, 1], [1, -1, 3]])
b = np.array([2, 7, 3])

# Memanggil fungsi untuk mencari solusi
solution = solve_with_lu_gauss(A, b)
print("Solusi sistem persamaan linear dengan metode dekomposisi LU Gauss:", solution)
