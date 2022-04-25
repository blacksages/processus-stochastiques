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

steps_list = []
for i in range(1000):
    state = 0
    steps = 0
    while(state != 2):
        state = transition_funct(trans_mat2, state)
        steps += 1
    steps_list.append(steps)

print(f'Average of time steps: {sum(steps_list)/len(steps_list)}')