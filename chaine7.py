import numpy as np
import random
import matplotlib.pyplot as plt

def transition_funct(matrix, state):
    prob = random.uniform(0, 1)
    if prob <= matrix[state][0]:
        return 0
    matrix_count = 0
    for i in range(1, len(matrix)):
        matrix_count += matrix[state] [i - 1]
        if prob >= matrix_count and prob <= matrix[state][i] + matrix_count:
            return i
    return 0

trans_mat2 = np.array([[1/3, 1/2, 1/6],[1/3, 1/3, 1/3],[0, 0, 1]])

T2 = []
for i in range(1000):
    state = 0
    for j in range(100):
        state = transition_funct(trans_mat2, state)
        if state == 1:
            T2.append(j + 1)
            break
        if state == 2:
            T2.append(None)
            break
    if len(T2) < i + 1:
        T2.append(None)

none_count = 0
not_none_count = 0
T2sum = 0
for i in range(len(T2)):
    if T2[i] is None:
        none_count += 1
    else:
        not_none_count += 1
        T2sum += T2[i]

print(f'Number of failures: {none_count}')
print(f'Number of success: {not_none_count}')
print(f'Average of T2: {T2sum/not_none_count}')