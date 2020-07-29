import Setup
import Opdracht1 as f
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

dataSet = "dataset1.csv"
validatieSet = "validation1.csv"
dayZ = "dayz.csv"


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
  return minindex([f.getEuclideanDistance(c, point) for c in centers])


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
  return sum([f.getEuclideanDistance(dataset[i], centeres[dataclusters[i]]) for i in range(len(dataset))])


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

  '''Create labels for Training data'''
  labelsTrainingData = []
  trainingData = Setup.datesReader(dataSet)
  Setup.addLabels(trainingData, labelsTrainingData)
  trainingData = Setup.dataReader(dataSet)

  norm = False

  if norm == True:
    low = Setup.findlowest(trainingData)
    high = Setup.findhigest(trainingData)
    Setup.normaliseer(trainingData, low, high)

  elbow(trainingData, 50)

  print("The best K is 3 because there are always 3 different seasons coming out (summer, fall / spring, winter)")


