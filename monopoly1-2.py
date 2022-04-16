import matplotlib.pyplot as plt
import numpy as np

n_states = 42
prison = [31, 32]
chance = [7, 22, 38]
chance_states = [0, 11, 15, 24, prison[0], 41]
chancellerie = [2, 17, 35]
chancellerie_states = [0, 1, prison[0]]

def handle_states(states, dist, dice1, dice2, next_state, extra_throws):
    if dice1 == dice2 and extra_throws > 0:
        for state in states:
            temp_dist = proba(state, extra_throws - 1)
            for i in range(n_states):
                dist[i] += (1/36)*(1/16)*temp_dist[i]
        temp_dist = proba(next_state, extra_throws - 1)
        for i in range(n_states):
            dist[i] += (1/36)*((16 - len(states))/16)*temp_dist[i]
    else:
        dist[next_state] += (1/36) * (16 - len(states))/16
        for state in states:
            dist[state] += (1/36) * (1/16)

def get_next_state(start_state, dice1, dice2):
    move = dice1 + dice2
    if start_state in prison:
        if dice1 == dice2:
            next_state = prison[len(prison) - 1] + move
        elif start_state == prison[len(prison) - 1]:
            next_state = prison[0] - 1
        else:
            next_state = start_state + 1
    else:
        next_state = start_state + move
        if start_state < prison[0] <= next_state:
            next_state += len(prison)
    return next_state%n_states
def proba(start_state, extra_throws):
    next_state = -1
    dist = [0 for _ in range(n_states)]
    if extra_throws < 0:
        return dist
    for dice1 in range(1, 7):
        for dice2 in range(1, 7):
            next_state = get_next_state(start_state, dice1, dice2)
            if next_state in chance:
                handle_states(chance_states, dist, dice1, dice2, next_state, extra_throws)
            elif next_state in chancellerie:
                handle_states(chancellerie_states, dist, dice1, dice2, next_state, extra_throws)
            elif dice1 == dice2 and extra_throws > 0:
                temp_dist = proba(next_state, extra_throws - 1)
                for i in range(n_states):
                    dist[i] += (1/36)*temp_dist[i]
            else:
                dist[next_state] += 1/36
    return  dist

# prison = [-1]
# chance = []
# chancellerie = []

dist = [proba(i, 2) for i in range(n_states)]

# plt.ylabel('Probability')
# plt.xlabel('Data')
# for i in range(len(dist)):
#     plt.clf()
#     plt.ylim((0.,1.))
#     plt.xlim((0,len(dist)+1))
#     plt.title(f'start state: {i}')
#     plt.bar([i + 1 for i in range(len(dist))], dist[i], 0.9)
#     plt.pause(0.2)

def prob_xt(matrix, x0, t):
    power = np.linalg.matrix_power(matrix, t)
    return np.dot(x0,power)

dist0 = [0 for _ in range(n_states)]
dist0[0] = 1
pxts = []
for j in range(0, 30):
    pxts.append(prob_xt(dist, dist0, j))

# for i in range(len(pxts)):
#     plt.clf()
#     plt.xlabel("Temps t")
#     plt.ylabel("P(Xt)")
#     plt.ylim((0.,1.))
#     plt.xlim((0.,len(dist)+1))
#     plt.title(f't = {i}')
#     plt.tight_layout()
#     plt.bar([i + 1 for i in range(len(dist))], pxts[i], 0.9)
#     plt.pause(0.2)

plt.clf()
plt.xlabel("Temps t")
plt.ylabel("P(Xt)")
plt.ylim((0.,1.))
plt.xlim((0.,len(dist)+1))
plt.tight_layout()
plt.bar([i + 1 for i in range(len(dist))], pxts[0], 0.9)
plt.pause(0.5)
plt.bar([i + 1 for i in range(len(dist))], pxts[1], 0.9)
plt.pause(0.5)
plt.bar([i + 1 for i in range(len(dist))], pxts[2], 0.9)
plt.pause(0.5)
plt.bar([i + 1 for i in range(len(dist))], pxts[3], 0.9)
plt.show()