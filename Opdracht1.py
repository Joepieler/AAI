import numpy as np
from collections import Counter
import Setup

def getEuclideanDistance(data1, data2):
  return np.linalg.norm(data1 - data2)

'''
predics for 1 point the most common label in k closeds points 
'''
def predict(datapoint,trainingdata,traininglabel, k):
  distlist = [(getEuclideanDistance(trainingdata[i], datapoint), traininglabel[i]) for i in range(len(trainingData))]
  distlist.sort(key=lambda x: x[0])
  distlist = distlist[0:k]
  distlist = [x[1] for x in distlist]
  return (Counter(distlist).most_common(1)[0][0])

'''
predicts the label for every point in dataset the most common label in k closeds points.
'''
def multpredict(dataset,traingdata,traininglabel,k):
  return [predict(i,traingdata,traininglabel,k) for i in dataset]

'''
validate if the label that is assigned is correct for all the predictions.
returns percentage 
'''
def validation(lstdata, lstpredict):
  return sum([lstdata[i] == lstpredict[i] for i in range(len(lstdata))]) / len(lstdata) * 100

'''returns the index of the highest number in a list'''
maxindex = lambda x : x.index(max(x))

'''
Returns the best K that has the most predictions correct
'''
def determine_k(dataset, datalabel, traingdata, traininglabel):
  return  maxindex([validation(multpredict(dataset, traingdata, traininglabel, k), datalabel) for k in range(1,len(traingdata)) ])
  #lst = []
  #for k in range(1,len(traingdata)):
  # lst.append(validation(multpredict(dataset, traingdata, traininglabel, k), datalabel))
  #return maxindex(lst)#lst.index(max(lst))


#Create labels for Training data
labelsTrainingData = []
trainingData = Setup.datesReader(Setup.dataSet)
Setup.addLabels(trainingData, labelsTrainingData)
trainingData = Setup.dataReader(Setup.dataSet)

#validation data
labelValidationData = []
validationData = Setup.datesReader(Setup.validatieSet)
Setup.addLabels1(validationData, labelValidationData)
validationData = Setup.dataReader(Setup.validatieSet)

daystopredict = Setup.dataReader(Setup.days)


norm = False

if __name__ == "__main__":

  if norm == True:
    low = Setup.findlowest(trainingData)
    high = Setup.findhigest(trainingData)
    Setup.normaliseer(trainingData, low, high)
    Setup.normaliseer(validationData, low, high)


  print (validation(multpredict(validationData, trainingData, labelsTrainingData, determine_k(validationData, labelValidationData, trainingData, labelsTrainingData)), labelValidationData), "% good")
  print(multpredict(daystopredict, trainingData, labelsTrainingData, 57))