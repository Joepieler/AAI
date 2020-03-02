import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1+np.exp(-x))

def sigmoid_derivate(x):
    return x * (1 * x)

input_vector = np.array([[0,0,1],
                         [1,1,1],
                         [1,0,1],
                         [0,1,1]])
np.random.seed(1)

output = np.array([[0,1,1,0]]).T

synaptic_weights = 2 * np.random.random((3, 1)) -1


class NeuralNetwork:

    def __init__(self, size ):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1))-1

class Neuron:

    def __init__(self, NumberOfInput):
        self.weight = 2 * np.random.random((NumberOfInput, 1)) -1


    def calc(self, input):
        return sigmoid(np.dot(input, self.weight))

    def train(self, input, ActualOutput, iterations):
        for i in range(iterations):
            calc_output = self.calc(input)
            error = ActualOutput - calc_output
            adjustments = error * sigmoid_derivate(calc_output)
            self.weight += np.dot(input.T, adjustments)

    def print(self):
        for i in range(len(self.weight)):
            print(self.weight[i])


a = Neuron(3)
a.train(input_vector, output, 100)

a.print()
print(a.calc(input_vector))



