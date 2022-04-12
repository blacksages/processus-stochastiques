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

trans_mat1 = np.array([[0, 1/2, 1/2],[1/3, 1/3, 1/3],[1/2, 1/2, 0]])
trans_mat2 = np.array([[1/3, 1/2, 1/6],[1/3, 1/3, 1/3],[0, 0, 1]])
trans_mat3 = np.array([[0, 0, 1, 0],[0, 0, 1, 0],[1, 0, 0, 0],[1/2, 1/2, 0, 0]])

def showgraph(trans_mat):
    fig, axes = plt.subplots(len(trans_mat))
    for start_state in range(len(trans_mat)):
        axes[start_state].set(xlabel='temps t', ylabel='P(Xt)')
        axes[start_state].set_title(f'From state {start_state+1}')
        axes[start_state].label_outer()

        state = start_state
        states_count = [[0] for _ in range(len(trans_mat))]
        states_count[state][0] = 1
        for i in range(1, 100):
            state = transition_funct(trans_mat,state)
            for j in range(len(states_count)):
                if j == state:
                    states_count[j].append(states_count[j][i-1]+1)
                else:
                    states_count[j].append(states_count[j][i-1])
        states_prob = [[c/(i+1) for i,c in enumerate(states_count[s])]
                    for s in range(len(states_count))]
        for i in range(len(states_prob)):
            axes[start_state].plot(states_prob[i], label = f'state {i+1}')
    plt.subplots_adjust(hspace=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

showgraph(trans_mat1)
showgraph(trans_mat2)
showgraph(trans_mat3)