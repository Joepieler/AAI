import numpy as np

def sigmoid(x, derivate=False):
  if(derivate==True):
    return (x * (1-x))

  return 1 / (1 + np.exp(-x))


#fuctions for the iris assigment
def name_to_array(x):
  if x == b'Iris-setosa':
    return np.array([1,0,0])
  elif x == b'Iris-versicolor':
    return np.array([0,1,0])
  elif x == b'Iris-virginica':
    return np.array([0,0,1])


def nomelize(data):
  for j in range(len(data[0])):
    lst = [i[j] for i in data]
    highest = max(lst)
    lowest = min(lst)
    for i in data:
      i[j] -= lowest
      i[j] = (i[j]/highest)
  return data


def probability(output, actoalouput):
  xcorrect = 0
  for i in range(len(output)):
    if np.where(output[i] == max(output[i])) == np.where(actoalouput[i] == max(actoalouput[i])):
     xcorrect += 1
  return  xcorrect

def dataReader(file):
  return nomelize(np.genfromtxt(file, delimiter=",", usecols=[0, 1, 2, 3]))


def nameReader(file):
  return np.genfromtxt(file, delimiter=",", usecols=[4], converters={4: lambda x: name_to_array(x)})



'''4.1 NOR gate 3 inputs'''
print ("opdracht 4.1")

#setup gates
NOR_input = np.array([[0,0,0],
                     [0,0,1],
                     [0,1,0],
                     [0,1,1],
                     [1,0,0],
                     [1,0,1,],
                     [1,1,0],
                     [1,1,1]])
NOR_output = np.array([[1,0,0,0,0,0,0,0]]).T


class Neuron:

  def __init__(self, NumberOfInput):
    self.weight = 2 * np.random.random((NumberOfInput, 1)) - 1
    self.Bias = 0
    self.threshold = 0.5

  def setWeight(self, weight):
    for i in range(len(weight)):
      self.weight[i] = weight[i]


  def calc(self, input):
    return sigmoid(np.dot(input, self.weight))

  def train(self, input, ActualOutput, iterations):
    for i in range(iterations):
      calc_output = self.calc(input)
      error = ActualOutput - calc_output
      adjustments = error * sigmoid(calc_output, True)
      self.weight += np.dot(input.T, adjustments)

  def print(self):
    for i in range(len(self.weight)):
      print(self.weight[i])

NOR_gate = Neuron(3)
NOR_gate.setWeight([-1, -1, -1])
print(NOR_gate.calc(NOR_input))
NOR_gate.print()

'''4.2 Neural ADDER'''


'''4.3 XOR Gate 2 inputs
I \/H\
  ||  > O
I /\H/

I Wanted to do this OOP but i couldn't get it to work 
'''
print("opdracht 4.3")


#The input
XOR_input = np.array([[0,0],
                      [0,1],
                      [1,0],
                      [1,1]])
#The output
XOR_output = np.array([[0],[1],[1],[0]])

#random seed
np.random.seed(2020)

#the weights in a matrix -1 is the bias
syn0 = 2 * np.random.random((2, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

#training
for i in range(10000):

  #layers
  layer0 = XOR_input
  layer1 = sigmoid(np.dot(layer0, syn0))
  layer2 = sigmoid(np.dot(layer1, syn1))

  #backpropagation
  layer2_error = XOR_output - layer2

  #Error gets printed
  if (i % 10000) == 0:
    print("error " + str(np.mean(np.abs(layer2_error))))

  layer2_delta = layer2_error * sigmoid(layer2, True)

  layer1_error = layer2_delta.dot(syn1.T)

  layer1_delta = layer1_error * sigmoid(layer1, True)

  #Update the weights
  syn1 += layer1.T.dot(layer2_delta)
  syn0 += layer0.T.dot(layer1_delta)

'''Test of the network'''
layer0 = np.array([[0,1]])
layer1 = sigmoid(np.dot(layer0, syn0))
layer2 = sigmoid(np.dot(layer1, syn1))
print("output is must be 1", layer2)

'''4.4 Iris dataset'''
print("opdracht4.4")
file_train = "iris.data"
file_validate = "bezdekIris.data"
input = dataReader(file_train)
output = nameReader(file_train)

syn0 = 2 * np.random.random((4, 4)) - 1
syn1 = 2 * np.random.random((4, 3)) - 1

#training
for i in range(10000):

  #layers
  layer0 = input
  layer1 = sigmoid(np.dot(layer0, syn0))
  layer2 = sigmoid(np.dot(layer1, syn1))

  #backpropagation
  layer2_error = output - layer2

  #Error gets printed
  if (i % 1000) == 0:
    print("error " + str(np.mean(np.abs(layer2_error))))

  layer2_delta = layer2_error * sigmoid(layer2, True)

  layer1_error = layer2_delta.dot(syn1.T)

  layer1_delta = layer1_error * sigmoid(layer1, True)

  #Update the weights
  syn1 += layer1.T.dot(layer2_delta)
  syn0 += layer0.T.dot(layer1_delta)


#test of the network
input = dataReader(file_validate)
output = nameReader(file_validate)

layer0 = input
layer1 = sigmoid(np.dot(layer0, syn0))
layer2 = sigmoid(np.dot(layer1, syn1))
number_of_good = probability(layer2, output)
print(number_of_good, " of the " ,len(output), " good that is %", number_of_good / len(output) * 100)

'''
it has now 1000000 iteration and it takes a minut on my laptop for 91% 


'''








