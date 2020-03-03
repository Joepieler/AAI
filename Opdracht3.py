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

    def __init__(self, inputSize, numberHiddenLayers, sizeHiddenLayers, numberOfOuputs ):
        self.hidden_neurons = [[Neuron(inputSize) for i in range(sizeHiddenLayers)]for i in range(numberHiddenLayers)]
        self.output_neurons = [Neuron(len(self.hidden_neurons)) for i in range(numberOfOuputs)]

    def calc(self, input):
        for i in range(len(self.hidden_neurons)):
            calc_output = [self.hidden_neurons[i][j].calc(input) for j in range(len(self.hidden_neurons[0]))]
        return [i.calc(calc_output) for i in self.output_neurons]

    def train(self, input, ActualOutput, iterations):
        for i in range(iterations):
            for j in range(len(self.output_neurons)):
                calc_output = self.calc(input)
                error = ActualOutput - calc_output
                #adjustments = error * sigmoid_derivate(calc_output)




class Neuron:

    def __init__(self, NumberOfInput):
        self.weight = 2 * np.random.random((NumberOfInput, 1)) -1
        self.Bias = 0
        self.threshold = 0.5

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



'''setup of gates '''
NOR_input = np.array([[0,0,0],
                     [0,0,1],
                     [0,1,0],
                     [0,1,1],
                     [1,0,0],
                     [1,0,1,],
                     [1,1,0],
                     [1,1,1]])
NOR_output = np.array([[1,0,0,0,0,0,0,0]]).T

XOR_input = np.array([[0,0],
                      [0,1],
                      [1,0],
                      [1,1]])
XOR_output = np.array([[0,1,1,0]]).T

'''4.1 NOR gate 3 inputs'''
NOR_gate = Neuron(3)
#NOR_gate.setWeight()

'''4.2 Neural ADDER'''


'''4.3 XOR Gate 2 inputs'''

HiddenNeuron0 = Neuron(2)
HiddenNeuron1 = Neuron(2)
OutputNeuron = Neuron(2)

for i in XOR_input:
    print(OutputNeuron.calc(np.array([HiddenNeuron0.calc(i), HiddenNeuron1.calc(i)]).T))

for t in range(100):
    for i in range(len(XOR_input)):
        OutputNeuron.train(np.array([HiddenNeuron0.calc(XOR_input[i]),
                                     HiddenNeuron1.calc(XOR_input[i])]).T,
                           XOR_output[i],
                           10000)
        HiddenNeuron0.train()
        HiddenNeuron0.train()

for i in XOR_input:
    print(OutputNeuron.calc(np.array([HiddenNeuron0.calc(i), HiddenNeuron1.calc(i)]).T))

# print(HiddenNeuron0.calc(XOR_input))
# print(HiddenNeuron1.calc(XOR_input))
# HiddenNeuron0.print()
# HiddenNeuron1.print()

#print(OutputNeuron.calc(np.array([HiddenNeuron0.calc(XOR_input), HiddenNeuron1.calc(XOR_input)])))









# a = Neuron(3)
# a.train(input_vector, output, 1000)
#
# a.print()
# print(a.calc(input_vector))



