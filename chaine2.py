import numpy as np
import random


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



for initial_state in range(len(trans_mat1)):
    next_state = initial_state
    print("\\begin{tikzpicture}")
    print(f'\t\\node[state, initial] ({1}) {{{initial_state + 1}}};')
    for i in range(2,11):
        next_state = transition_funct(trans_mat1, next_state)
        print(f'\t\\node[state, right of = {i-1}, xshift = 0.3cm] ({i}) {{{next_state+1}}};')
    
    print("\t\\draw", end="")
    print(f'\t[->]({1}) edge ({2})')
    for i in range(3,10):
        print(f'\t\t\t[->]({i-1}) edge ({i})')
    print(f'\t\t\t[->]({9}) edge ({10});')
    
    print("\\end{tikzpicture}")


print("\n")
print("\n")
print("\n")


for initial_state in range(len(trans_mat2)):
    next_state = initial_state
    print("\\begin{tikzpicture}")
    print(f'\t\\node[state, initial] ({1}) {{{initial_state + 1}}};')
    for i in range(2,11):
        next_state = transition_funct(trans_mat2, next_state)
        print(f'\t\\node[state, right of = {i-1}, xshift = 0.3cm] ({i}) {{{next_state+1}}};')
        
    print("\t\\draw", end="")
    print(f'\t[->]({1}) edge ({2})')
    for i in range(3,10):
        print(f'\t\t\t[->]({i-1}) edge ({i})')
    print(f'\t\t\t[->]({9}) edge ({10});')
    
    print("\\end{tikzpicture}")

print("\n")
print("\n")
print("\n")


for initial_state in range(len(trans_mat3)):
    next_state = initial_state
    print("\\begin{tikzpicture}")
    print(f'\t\\node[state, initial] ({1}) {{{initial_state + 1}}};')
    for i in range(2,11):
        next_state = transition_funct(trans_mat3, next_state)
        print(f'\t\\node[state, right of = {i-1}, xshift = 0.3cm] ({i}) {{{next_state+1}}};')
        
    print("\t\\draw", end="")
    print(f'\t[->]({1}) edge ({2})')
    for i in range(3,10):
        print(f'\t\t\t[->]({i-1}) edge ({i})')
    print(f'\t\t\t[->]({9}) edge ({10});')
    
    print("\\end{tikzpicture}")




#power1_10 = np.linalg.matrix_power(trans_mat1, 10)
#print(power1_10"\n")
#power2_50 = np.linalg.matrix_power(trans_mat2, 50)
#print(power2_50)
#power3_100 = np.linalg.matrix_power(trans_mat3, 100)
#print(power3_100)

