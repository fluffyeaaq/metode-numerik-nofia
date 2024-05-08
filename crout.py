import numpy as np

# Fungsi untuk melakukan dekomposisi Crout
def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for j in range(n):
        U[j][j] = 1
        
        for i in range(j, n):
            L[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            
        for i in range(j+1, n):
            U[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(j))) / L[j][j]
    
    return L, U

# Fungsi untuk mencari solusi dengan dekomposisi Crout
def solve_with_crout(A, b):
    L, U = crout_decomposition(A)
    
    # Menyelesaikan Ly = b
    y = np.linalg.solve(L, b)
    
    # Menyelesaikan Ux = y
    x = np.linalg.solve(U, y)
    
    return x

# Contoh sistem persamaan linear
A = np.array([[2, 1, -1], [4, 1, 1], [1, -1, 3]])
b = np.array([2, 7, 3])

# Memanggil fungsi untuk mencari solusi
solution = solve_with_crout(A, b)
print("Solusi sistem persamaan linear dengan metode dekomposisi Crout:", solution)
