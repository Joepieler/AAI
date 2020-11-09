import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
import pickle


def getEuclideanDistance(data1, data2):
  return np.linalg.norm(data1 - data2)

'''
Makes K random points
'''
def determinecluster_k(dataset, k):
  return [dataset[i] for i in random.sample(range(0, len(dataset)-1),k)]


"""
Find the index of a lowest element in a array
"""
minindex = lambda x: x.index(min(x))


'''
determine the nearest center of a point
'''
def determinenearestcenter(point, centers):
  return minindex([getEuclideanDistance(c, point) for c in centers])


'''
For each point in dataset determine de nearest center
'''
def multideterminenearestcenter(dataset, centers):
  return [determinenearestcenter(d, centers) for d in dataset]



averigecenter = lambda x : np.sum(x, axis=0) / len(x)
'''
determine for one cluster the cente
'''
def determineCenture(dataset, clusters, center):
  return averigecenter([dataset[d] for d in range(len(dataset)) if clusters[d] == center])


'''
for every cluster determine new centure
'''
def multidetermineCenture(dataset, clusters, centroids):
  return np.array([determineCenture(dataset, clusters, i) for i in range(len(centroids))])

'''
K Means Recursive function
'''
def K_means(dataset, centroids):
  new = np.array(multidetermineCenture(dataset, multideterminenearestcenter(dataset, centroids), centroids))
  if not np.array_equal(new, centroids):
    return K_means(dataset, new)
  return new


'''
This function returns the most common labal of a cluster
'''
def givelabel(dataset, datalabel, centroids):
  lst = [[] for i in range(len(centroids))]
  for i in range(len(dataset)):
    lst[multideterminenearestcenter(dataset, centroids)[i]].append(datalabel[i])
  return [(Counter(l).most_common(1)[0][0]) for l in lst]


'''
Sums the distance from points to closest centers
'''
def sumdistanceclusters(dataset, dataclusters, centeres):
  return sum([getEuclideanDistance(dataset[i], centeres[dataclusters[i]]) for i in range(len(dataset))])


'''
Plot the data on a plot screen
'''
def plot(x, y):
  plt.plot(x, y)
  plt.xlabel("Number of clusters")
  plt.ylabel("distance between clusters")
  plt.title("K-Means Clustering")
  plt.show()


'''
Draws the elbow for k
'''
def elbow(dataset, max_k):
  x = list(range(1, max_k))
  y = []
  for k in range(1, max_k):
    clusters = K_means(dataset, determinecluster_k(dataset, k))
    y.append(sumdistanceclusters(dataset, multideterminenearestcenter(dataset, clusters), clusters))
  plot(x, y)


if __name__ == "__main__":

  '''Create Training data'''
  layer_output = []
  labels = []

  pickle_in = open("layer_output.pickle", "rb")
  layer_output = pickle.load(pickle_in)
  pickle_in.close()

  pickle_in = open("Y_labels", "rb")
  labels = pickle.load(pickle_in)
  pickle_in.close()


  elbow(layer_output, 20)
  clusters = K_means(layer_output, determinecluster_k(layer_output, 5))
  print(givelabel(layer_output, labels, clusters))







