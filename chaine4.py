import numpy as np


trans_mat1 = np.array([[0, 1/2, 1/2],[1/3, 1/3, 1/3],[1/2, 1/2, 0]])
trans_mat2 = np.array([[1/3, 1/2, 1/6],[1/3, 1/3, 1/3],[0, 0, 1]])
trans_mat3 = np.array([[0, 0, 1, 0],[0, 0, 1, 0],[1, 0, 0, 0],[1/2, 1/2, 0, 0]])




power1_10 = np.linalg.matrix_power(trans_mat1, 1)
print(power1_10)
print("\n")
power1_50 = np.linalg.matrix_power(trans_mat1, 3)
print(power1_50)
print("\n")
power1_100 = np.linalg.matrix_power(trans_mat1, 10)
print(power1_100)
print("\n")
print("\n")
print("\n")



power2_10 = np.linalg.matrix_power(trans_mat2, 1)
print(power2_10)
print("\n")
power2_50 = np.linalg.matrix_power(trans_mat2, 3)
print(power2_50)
print("\n")
power2_100 = np.linalg.matrix_power(trans_mat2, 10)
print(power2_100)
print("\n")
print("\n")
print("\n")



power3_10 = np.linalg.matrix_power(trans_mat3, 1)
print(power3_10)
print("\n")
power3_50 = np.linalg.matrix_power(trans_mat3, 3)
print(power3_50)
print("\n")
power3_100 = np.linalg.matrix_power(trans_mat3, 10)
print(power3_100)
print("\n")