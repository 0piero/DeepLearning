from HeapMin import HeapMin
import numpy as np

class Graph:
    def __init__(self, vertices):
        self.v = vertices
        self.gr = np.array([[None]*self.v for i in range(self.v)])

    def adiciona_aresta(self, u, v, w):
        self.gr[u-1][v-1] = w
        self.gr[v-1][u-1] =  w

    def show(self):
            print(np.array(self.gr))



