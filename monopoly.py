import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_states = 43
to_prison = 33
prison = [11, 12, 13]
chance = [7, 25, 39]
chance_states = [0, prison[0], 14, 18, 27, 42]
chancellerie = [2, 20, 36]
chancellerie_states = [0, 1, prison[0]]
extra_throws = 2
throws_prison = False
start_dist = [0 for _ in range(n_states)]
start_dist[0] = 1

def handle_states(states, dist, dice1, dice2, next_state, throw):
    if dice1 == dice2 and throw < extra_throws:
        for state in states:
            temp_dist = proba(state, throw + 1)
            for i in range(n_states):
                dist[i] += (1/36)*(1/16)*temp_dist[i]
        temp_dist = proba(next_state, throw + 1)
        for i in range(n_states):
            dist[i] += (1/36)*((16 - len(states))/16)*temp_dist[i]
    else:
        dist[next_state] += (1/36) * (16 - len(states))/16
        for state in states:
            dist[state] += (1/36) * (1/16)

def get_next_state(start_state, dice1, dice2, throw):
    move = dice1 + dice2
    if start_state in prison:
        if dice1 == dice2 or start_state == prison[len(prison) - 1]:
            next_state = prison[len(prison) - 1] + move
        else:
            next_state = start_state + 1
    elif dice1 == dice2 and throw == extra_throws and throws_prison == True:
        next_state = prison[0]
    else:
        next_state = start_state + move
        if start_state < prison[0] <= next_state:
            next_state += len(prison)
    if next_state%n_states == to_prison:
        return prison[0]
    else:
        return next_state%n_states

def proba(start_state, throw = 0):
    next_state = -1
    dist = [0 for _ in range(n_states)]
    if throw > extra_throws:
        return dist
    for dice1 in range(1, 7):
        for dice2 in range(1, 7):
            next_state = get_next_state(start_state, dice1, dice2, throw)
            if next_state in chance:
                handle_states(chance_states, dist, dice1, dice2, next_state, throw)
            elif next_state in chancellerie:
                handle_states(chancellerie_states, dist, dice1, dice2, next_state, throw)
            elif dice1 == dice2 and extra_throws > 0:
                temp_dist = proba(next_state, throw + 1)
                for i in range(n_states):
                    dist[i] += (1/36)*temp_dist[i]
            else:
                dist[next_state] += 1/36
    return  dist

def prob_xt(matrix, x0, t):
    power = np.linalg.matrix_power(matrix, t)
    return np.dot(x0,power)

def showgraph(trans_mat, initial_dist, times):
    fig, axes = plt.subplots(len(times))
    for i,t in enumerate(times):
        axes[i].set(xlabel='Temps t', ylabel='P(Xt)')
        axes[i].set_title(f't = {t}')
        axes[i].label_outer()
        pxt = prob_xt(trans_mat, initial_dist, t)
        axes[i].set(ylim=(0.,1.), xlim=(0.,len(trans_mat)+1))
        axes[i].bar([i + 1 for i in range(len(dist))], pxt, 0.9)
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.show()

extra_throws = 2
throws_prison = True
dist = [proba(i) for i in range(n_states)]

showgraph(dist, start_dist, [1,3,10])

max_t = 40
pi_y = [0 for i in range(n_states)]
for t in range(1,max_t + 1):
    xt = prob_xt(dist, start_dist, t)
    for i in range(n_states):
        pi_y[i] += xt[i]
for i in range(n_states):
    pi_y[i] /= max_t

with open('cases.txt', 'r') as file:
    spaces = [line for line in file]
with open('monopoly4.txt', 'w', encoding="utf-8") as file:
    file.write('============\n')
    file.write('==== 4b ====\n')
    file.write('============\n')
    pi_prison = sum([pi_y[i] for i in range(prison[0], prison[len(prison) - 1] + 1)])
    data = zip([el  for l in [[space[:-1] for space in spaces[:prison[0] - 1]],
                              ['Prison', 'Prison (visite)'],
                              [space[:-1] for space in spaces[prison[0]:]]]
                    for el in l],
               [el  for l in [pi_y[:prison[0] - 1],
                              [pi_prison, pi_y[prison[0] - 1]],
                              pi_y[prison[len(prison) - 1] + 1:]]
                    for el in l])
    index = [el for l in [[str(i) for i in range(1, prison[0])],
                          [str(prison[0]) + 'a', str(prison[0]) + 'b'],
                          [str(i) for i in range(prison[0] + 1, len(spaces) + 1)]]
                for el in l]
    columns = ['Case', 'Proportion moyenne du temps']
    df = pd.DataFrame(data = data, index = index, columns = columns)
    file.write(df.to_string()+'\n')
    file.write('============\n')
    file.write('==== 4c ====\n')
    file.write('============\n')
    bad_df = df.index.isin([str(prison[0]) + 'a', str(prison[0]) + 'b'])
    file.write(f'Rue à acheter en priorité: {df.loc[df.loc[~bad_df][columns[1]].idxmax()][columns[0]]}')
