from Graph import Graph
import numpy as np
from random import random
import matplotlib.pyplot as plt
class Node:
    def __init__(self, bias , weight):

        self.bias = bias
        self.weight = weight

    def __repr__(self):
        return f"{self.weight}"
class MLP(Graph):
    def __init__(self, input_layer_size, dense_layer_size, output_layer_size, dense_layers):
        super().__init__(input_layer_size + dense_layer_size * dense_layers + output_layer_size)
        self.loss = []
        self.error = np.zeros((output_layer_size))
        self.inp = input_layer_size
        self.hid = dense_layer_size
        self.dl = dense_layers
        self.out = output_layer_size


        #initialize weights and bias for each layer
        w = self.initialize_weights(input_layer_size, dense_layer_size)
        b = self.initialize_bias(input_layer_size, dense_layer_size)
        #w = [[0.1,0.1,0.1],[0.1,0.1,0.1]]
        #b = [[0.1,0.1,0.1],[0.1,0.1,0.1]]
        m = 0
        for i in range(1, input_layer_size+1):
            n = 0
            for j in range(input_layer_size + 1, input_layer_size + dense_layer_size+1):


                node = Node(b[m][n], w[m][n])
                self.adiciona_aresta(i, j,node)
                n+=1
            m += 1

        k = 0
        m = 0
        w = self.initialize_weights(dense_layer_size, dense_layer_size)
        #w = [[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]]
        #b = [[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]]
        b = self.initialize_bias(dense_layer_size, dense_layer_size)
        for j in range(input_layer_size + 1, input_layer_size + dense_layer_size * (dense_layers-1) + 1):
            n = 0
            if k > dense_layer_size - 1:
                k = 0
                m = 0
            for i in range(j + dense_layer_size - k, j + dense_layer_size*2 - k):

                node = Node(b[m][n], w[m][n])
                self.adiciona_aresta(j, i, node)
                n += 1
            k += 1
            m += 1

        m = 0
        w = self.initialize_weights(dense_layer_size, output_layer_size)
        b = self.initialize_bias(dense_layer_size, output_layer_size)
        #w = [[0.1],[0.1],[0.1]]
        #b = [[0.1], [0.1], [0.1]]
        for i in range(self.v - output_layer_size - dense_layer_size + 1, self.v - output_layer_size + 1):
            n=0
            for j in range(self.v - output_layer_size + 1, self.v + 1):

                node = Node(b[m][n], w[m][n])
                self.adiciona_aresta(i, j,node)
                n+=1
            m+=1
    
    def shape(self):
        s = [(self.inp,self.hid)]
        for i in range(self.dl-1):
            s.append((self.hid,self.hid))
        s.append((self.hid,self.out))
        return s
    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
    @staticmethod
    def sigmoid_derivative(z):
        return MLP.sigmoid(z) * (1 - MLP.sigmoid(z))
    @staticmethod
    def mse_derivative(output, expected):
        return (expected - output)

    def backprop(self, x, y):
        aux = []
        activation = x
        activations = [np.array(x)]
        h = []

        for i in range(0, self.inp):

            aux.append(np.multiply([node.weight for node in self.gr[i][self.inp:self.inp+self.hid]], activation[i]))
        #print(aux)
        #print([node.bias for node in self.gr[0][self.inp:self.inp+self.hid]])
        aux = sum(aux) + [node.bias for node in self.gr[0][self.inp:self.inp+self.hid]]
        h.append(aux)
        activation = self.sigmoid(aux)
        activations.append(activation)
        aux = []



        for i in range(self.inp, self.inp + self.hid * (self.dl-1), self.hid):
            k = 0
            for j in range(i, i + self.hid):

                aux.append(np.multiply([node.weight for node in self.gr[j][i + self.hid:i+self.hid + self.hid]], activation[k]))
                k+=1

            aux = sum(aux) +[node.bias for node in self.gr[i][i + self.hid:i+self.hid + self.hid]]

            h.append(aux)
            activation = self.sigmoid(aux)
            activations.append(activation)
            aux = []

        j = 0
        for i in range(self.v-self.out-self.hid,self.v-self.out):

            aux.append(np.multiply([node.weight for node in self.gr[i][self.v-self.out:self.v]], activation[j]))
            j+=1
        aux = sum(aux)+[node.bias for node in self.gr[self.v-self.out-self.hid][self.v-self.out:self.v]]
        h.append(aux)
        activation = self.sigmoid(aux)
        activations.append(activation)
        del aux
        #print(f"activations {activations}")
        #print(f"h {h}")
        self.error = np.sum([np.square(self.mse_derivative(activations[-1], y)), self.error], axis = 0)
        dE = self.mse_derivative(activations[-1], y) * self.sigmoid_derivative(h[-1])
        #print(dE)
        weight_d = [np.dot(np.transpose([activations[-2]]),[dE])]
        #   print(weight_d)

        bias_d = [np.dot(np.transpose([np.ones(activations[-2].shape)]),[dE])]
        #print(bias_d)

        W = []
        k = self.v - self.out - self.hid
        for i in range(2,len(activations)):

            if i == 2:

                for j in range(self.hid):
                    W.append([node.weight for node in self.gr[self.v-self.out-self.hid:self.v-self.out, self.v-self.out:self.v][j]])


                dE = np.dot(dE, np.transpose(W)) * [self.sigmoid_derivative(h[-i])]
                #print(dE)
                weight_d.append(np.dot(np.transpose([activations[-(i+1)]]), dE))
                bias_d.append(np.dot(np.transpose([np.ones(activations[-(i+1)].shape)]), dE))
                #print(weight_d)
                #print(bias_d)

                k -= self.hid
            else:

                for j in range(self.hid):


                    W.append([node.weight for node in self.gr[k:k+self.hid, k+self.hid:k+ self.hid * 2][j]])

                dE = np.dot(dE, np.transpose(W[(i-2)*self.hid:(i-1)*self.hid])) * [self.sigmoid_derivative(h[-i])]
                #print(dE)
                weight_d.append(np.dot(np.transpose([activations[-(i + 1)]]), dE))
                bias_d.append(np.dot(np.transpose([np.ones(activations[-(i+1)].shape)]), dE))
                #print(weight_d)
                #print(bias_d)
                k -= self.hid

        return weight_d, bias_d


    def mini_batch(self, inputs, targets, epochs, batch_size, learning_rate):

        for i in range(epochs):
            dE_dW = [np.zeros(s) for s in reversed(self.shape)]
            dE_dB = [np.zeros(s) for s in reversed(self.shape)]
            #randi = np.random.choice(range(0,len(inputs)), batch_size, replace = False)
            r = np.random.permutation(range(0,len(inputs)))
            for k in range(0,len(inputs)-batch_size,batch_size):
                for l in r[k:k+batch_size]:
                    dw, db = self.backprop(inputs[l], targets[l])

                    dE_dW = [np.sum(np.array([a, b]), axis=0) for a, b in zip(dE_dW, dw)]
                    dE_dB = [np.sum(np.array([a, b]), axis=0) for a, b in zip(dE_dW, db)]

                self.gradient_descent(dE_dW, dE_dB, learning_rate / batch_size)
            #for j in randi:
                #dw, db = self.backprop(inputs[j],targets[j])

                #dE_dW = [np.sum(np.array([a,b]), axis = 0) for a,b in zip(dE_dW,dw)]
                #dE_dB = [np.sum(np.array([a,b]), axis = 0) for a,b in zip(dE_dW,db)]


            #self.gradient_descent(dE_dW, dE_dB, learning_rate/batch_size)
            if i%10 == 0:
                self.loss.append(np.sum(self.error)/batch_size)
                print(f"loss: {np.sum(self.error)/batch_size}")
            self.error = np.zeros((self.out))

    def gradient_descent(self, dw, db, l):

        for i in range(self.inp):
            k = 0
            for j in range(self.inp, self.inp + self.hid):

                self.gr[i][j].weight += l * dw[-1][i][k]
                self.gr[i][j].bias += l * db[-1][i][k]
                self.gr[j][i].weight += l * dw[-1][i][k]
                self.gr[j][i].bias += l * db[-1][i][k]
                k += 1

        k = 0
        m = 2
        n = 0
        for j in range(self.inp, self.inp + self.hid * (self.dl - 1)):
            o = 0
            if k > self.hid - 1:
                k = 0
                n = 0
                o = 0
                m += 1
            for i in range(j + self.hid - k, j + self.hid * 2 - k):

                self.gr[j][i].weight += l * dw[-m][n][o]
                self.gr[j][i].bias += l * db[-m][n][o]
                self.gr[i][j].weight += l * dw[-m][n][o]
                self.gr[i][j].bias += l * db[-m][n][o]
                o += 1

            n += 1
            k += 1

        n = 0
        for i in range(self.v - self.out - self.hid, self.v - self.out):
            m = 0
            for j in range(self.v - self.out, self.v):

                self.gr[j][i].weight += l * dw[0][n][m]
                self.gr[j][i].bias += l * db[0][n][m]
                self.gr[i][j].weight += l * dw[0][n][m]
                self.gr[i][j].bias += l * db[0][n][m]
                m += 1
            n += 1
    def evaluate(self, inputs):
        aux = []
        activation = inputs


        for i in range(0, self.inp):
            aux.append(np.multiply([node.weight for node in self.gr[i][self.inp:self.inp + self.hid]], activation[i]))
        # print(aux)
        # print([node.bias for node in self.gr[0][self.inp:self.inp+self.hid]])
        aux = sum(aux) + [node.bias for node in self.gr[0][self.inp:self.inp + self.hid]]

        activation = self.sigmoid(aux)

        aux = []

        for i in range(self.inp, self.inp + self.hid * (self.dl - 1), self.hid):
            k = 0
            for j in range(i, i + self.hid):
                aux.append(np.multiply([node.weight for node in self.gr[j][i + self.hid:i + self.hid + self.hid]],
                                       activation[k]))
                k += 1

            aux = sum(aux) + [node.bias for node in self.gr[i][i + self.hid:i + self.hid + self.hid]]


            activation = self.sigmoid(aux)

            aux = []

        j = 0
        for i in range(self.v - self.out - self.hid, self.v - self.out):
            aux.append(np.multiply([node.weight for node in self.gr[i][self.v - self.out:self.v]], activation[j]))
            j += 1
        aux = sum(aux) + [node.bias for node in self.gr[self.v - self.out - self.hid][self.v - self.out:self.v]]

        activation = self.sigmoid(aux)

        del aux
        return activation

    def initialize_weights(self, n, m):
        # Receive number of nodes from the previous layer as n and further layer as m
        # Also return a np array of shape(n,m)
        # Xavier uniform distribution
        # number of nodes in the previous layer
        # n = 2
        # m = 3
        # calculate the range for the weights
        lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))
        # generate random numbers
        numbers = np.random.rand(n, m)
        r_max = np.max(numbers)
        r_min = np.min(numbers)
        range = r_max - r_min
        # scale to the desired range
        scaled = ((numbers - r_min) / (range)) * (upper - lower) + lower
        # summarize
        # print(scaled)
        # print(numbers)
        # print(scaled)
        # print(lower, upper)
        # print(scaled.min(), scaled.max())
        # print(scaled.mean(), scaled.std())
        #print(scaled)
        return scaled
    def initialize_bias(self, n, m):
        lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))
        numbers = np.random.rand(n, m)
        r_max = np.max(numbers)
        r_min = np.min(numbers)
        range = r_max - r_min
        scaled = ((numbers - r_min) / (range)) * (upper - lower) + lower
        return scaled


if __name__ == "__main__":
    items = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])
    #print(f"{items}")
    #print(f"{targets}")
    net = MLP(2, 3, 1, 2)
    #net.show()
    net.mini_batch(items, targets, 500, 64, 0.1)
    plt.plot(range(0, 500, 10), net.loss)
    plt.show()
    #print(net.evaluate([0.1,0.1]))
    #net.backprop([0.1,0.1],[0.2])

