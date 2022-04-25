import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

n_states = 43
to_prison = 33
prison = [11, 12, 13]
chance = [7, 25, 39]
chance_states = [0, prison[0], 14, 18, 27, 42]
chancellerie = [2, 20, 36]
chancellerie_states = [0, 1, prison[0]]
throws_to_prison = 3
throws_prison = False

def handle_states(states, dist, next_state):
    dist[next_state] += (1/36) * (16 - len(states))/16
    for state in states:
        dist[int(next_state/n_states) * n_states + state] += (1/36) * (1/16)

def get_next_state(start_state, dice1, dice2):
    move = dice1 + dice2
    offset = int(start_state/n_states) * n_states if throws_prison is True else 0

    if start_state in prison:
        if dice1 == dice2 or start_state%n_states == prison[len(prison) - 1]:  # On sort de prison
            next_state = prison[len(prison) - 1] + move
            if throws_prison is True and dice1 == dice2:
                offset = n_states
        else: # On reste en prison
            next_state = start_state + 1
    elif throws_prison is True\
         and dice1 == dice2\
         and int(1 + start_state/n_states) == throws_to_prison: # On a fait trop de double => prison
        next_state = prison[0]
        offset = 0
    else:
        offset += n_states if throws_prison is True and dice1 == dice2 else 0
        next_state = start_state + move
        if start_state%n_states < prison[0] <= next_state%n_states:
            next_state += len(prison)
    if next_state%n_states == to_prison:
        return prison[0]
    else:
        return offset + next_state%n_states

def proba(start_state):
    next_state = -1
    if throws_prison is True:
        dist = [0 for i in range(throws_to_prison * n_states)]
    else:
        dist = [0 for i in range(n_states)]
    for dice1 in range(1, 7):
        for dice2 in range(1, 7):
            next_state = get_next_state(start_state, dice1, dice2)
            if next_state in chance:
                handle_states(chance_states, dist, next_state)
            elif next_state in chancellerie:
                handle_states(chancellerie_states, dist, next_state)
            else:
                dist[next_state] += 1/36
    return  dist

def prob_xt(matrix, x0, t):
    power = np.linalg.matrix_power(matrix, t)
    return np.dot(x0,power)

def compute_pi(trans_mat, max_t = 1000):
    pi = [0 for i in range(len(trans_mat))]
    pxt = [0 for i in range(len(trans_mat))]
    pxt[0] = 1
    for t in range(1,max_t + 1):
        pxt = prob_xt(trans_mat, pxt, 1)
        for i in range(n_states):
            pi[i] += pxt[i]
    for i in range(n_states):
        pi[i] /= max_t
    return pi

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

def avg_distance(from_state, to_state, trans_mat, n_step = 1000):
    total = 0
    for i in range(n_step):
        next_state = from_state
        count = 0
        while next_state != to_state or count == 0:
            count += 1
            next_state = transition_funct(trans_mat, next_state)
        total += count
    return total/n_step

def savegraph(trans_mat, initial_dist, times):
    fig, axes = plt.subplots(len(times))
    color = ['blue' for i in range(len(trans_mat))]
    for i in prison:
        color[i] = 'black'
    for i in chance:
        color[i] = 'green'
    for i in chancellerie:
        color[i] = 'brown'
    for i,t in enumerate(times):
        axes[i].set(xlabel='Temps t', ylabel='P(Xt)')
        axes[i].set_title(f't = {t}')
        axes[i].label_outer()
        pxt = prob_xt(trans_mat, initial_dist, t)
        axes[i].set(ylim=(0.,0.2), xlim=(0.,len(trans_mat)+1))
        axes[i].bar([i + 1 for i in range(len(trans_mat))], pxt, 0.9, color = color)
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.savefig('monopoly2.4a')

throws_prison = False
trans_mat = [proba(i) for i in range(n_states)]
start_dist = [0 for i in range(n_states)]
start_dist[0] = 1
savegraph(trans_mat, start_dist, [1,3,10])

import pandas as pd 
pd.DataFrame(np.array(trans_mat)).to_csv("monopoly_trans_mat2.2.csv")
pi = compute_pi(trans_mat)

with open('cases.txt', 'r', encoding="utf-8") as file:
    spaces = [line for line in file]
with open('monopoly2.4et2.6.txt', 'w', encoding="utf-8") as file:
    file.write('============\n')
    file.write('==== 4b ====\n')
    file.write('============\n')
    pi_prison = sum([pi[i] for i in range(prison[0], prison[len(prison) - 1] + 1)])
    data = zip([el  for l in [[space[:-1] for space in spaces[:prison[0] - 1]],
                              ['Prison', 'Prison (visite)'],
                              [space[:-1] for space in spaces[prison[0]:]]]
                    for el in l],
               [el  for l in [pi[:prison[0] - 1],
                              [pi_prison, pi[prison[0] - 1]],
                              pi[prison[len(prison) - 1] + 1:]]
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
    best_property = df.loc[df.loc[~bad_df][columns[1]].idxmax()]
    file.write(f'Rue à acheter en priorité: {best_property.index.tolist()[0]}. {best_property[columns[0]]}\n')
    file.write('============\n')
    file.write('==== 4d ====\n')
    file.write('============\n')
    file.write(f'Temps moyen jusqu\'à la prison: {avg_distance(0, prison[0], trans_mat)}\n')
    file.write(f'Temps moyen entre deux allers en prison: {avg_distance(prison[len(prison) - 1], prison[0], trans_mat)}\n')
    file.write('============\n')
    file.write('==== 6 ====\n')
    file.write('============\n')
    throws_to_prison = 3
    throws_prison = True
    start_dist = [0 for i in range(throws_to_prison * n_states)]
    start_dist[0] = 1
    trans_mat = [proba(i) for i in range(throws_to_prison * n_states)]
    pi = compute_pi(trans_mat)
    pi_tables = [pi[i * n_states:(i + 1) * n_states] for i in range(throws_to_prison)]
    pi = [sum(both_pi) for both_pi in zip(*pi_tables)]
    pi_prison = sum([pi[i] for i in range(prison[0], prison[len(prison) - 1] + 1)])
    file.write(f'Proportion moyenne du temps en prison: {pi_prison}\n')
    file.write(f'Temps moyen jusqu\'à la prison: {avg_distance(0, prison[0], trans_mat)}\n')
    file.write(f'Temps moyen entre deux allers en prison: {avg_distance(prison[len(prison) - 1], prison[0], trans_mat)}\n')

