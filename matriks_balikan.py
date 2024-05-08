import numpy as np

# Fungsi untuk mencari solusi dengan metode matriks balikan
def solve_with_inverse(A, b):
    try:
        # Mencari matriks balikan dari A
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Matriks tidak memiliki matriks balikan.")
        return None
    
    # Mengalikan matriks balikan dengan vektor hasil
    x = np.dot(A_inv, b)
    
    return x

# Matriks koefisien
A = np.array([[2, 3, -1],
              [3, 2, 2],
              [1, -1, 1]])

# Vektor hasil
b = np.array([7, 11, 1])

# Memanggil fungsi untuk mencari solusi
solution = solve_with_inverse(A, b)
if solution is not None:
    print("Solusi sistem persamaan linear dengan metode matriks balikan:", solution)
