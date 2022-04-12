from cProfile import label
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.operators.unary import reverse
from networkx.algorithms.sparsifiers import _lightest_edge_dicts
from networkx.classes.function import all_neighbors, neighbors, nodes

#x0 est supposé être un vecteur colone
def prob_xt(matrix, x0, t):
    power = np.linalg.matrix_power(matrix, t)
    return np.dot(x0,power)



def showgraph(matrix, dists):
    pxts = [[] for _ in range(len(dists))]
    for i in range(len(dists)):
        for j in range(0, 30):
            pxts[i].append(prob_xt(matrix, dists[i], j))

    fig, axes = plt.subplots(3)
    for i in range(len(dists)):
        axes[i].set(xlabel='temps t', ylabel='P(Xt)', ylim=(0.,1.))
        axes[i].set_title(f'distribution n°{i}')
        axes[i].label_outer()
        for j in range(len(matrix)):
            axes[i].plot([pxt[j] for pxt in pxts[i]], label = f'state {j+1}')

    plt.subplots_adjust(hspace=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

trans_mat1 = np.array([[0, 1/2, 1/2],[1/3, 1/3, 1/3],[1/2, 1/2, 0]])
trans_mat2 = np.array([[1/3, 1/2, 1/6],[1/3, 1/3, 1/3],[0, 0, 1]])
trans_mat3 = np.array([[0, 0, 1, 0],[0, 0, 1, 0],[1, 0, 0, 0],[1/2, 1/2, 0, 0]])

mat1_st_x0 = np.array([1/3, 1/3, 1/3]) 
mat1_nd_x0 = np.array([1/6, 1/2, 1/3])
mat1_rd_x0 = np.array([1/10, 5/10, 2/5])


mat2_st_x0 = np.array([1/4, 1/2, 1/4]) 
mat2_nd_x0 = np.array([0, 1, 0])
mat2_rd_x0 = np.array([1/10, 9/10, 0])

mat3_st_x0 = np.array([1/4, 1/4, 1/4, 1/4]) 
mat3_nd_x0 = np.array([0, 0, 0, 1])
mat3_rd_x0 = np.array([1/3, 1/2, 0, 1/6])

showgraph(trans_mat1, [mat1_st_x0, mat1_nd_x0, mat1_rd_x0])
showgraph(trans_mat2, [mat2_st_x0, mat2_nd_x0, mat2_rd_x0])
showgraph(trans_mat3, [mat3_st_x0, mat3_nd_x0, mat3_rd_x0])
# prendre 3 differanet au choix dont 1 qui va etre uniforme (tous les meme)
# calcul prob_xt de 0 a 100 
#fait le graphe en fonction du temps (un graphe pour les 3 etat)