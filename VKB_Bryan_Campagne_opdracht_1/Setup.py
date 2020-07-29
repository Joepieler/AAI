'''
Setup file for the assignments
'''

import numpy as np

dataSet = "dataset1.csv"
validatieSet = "validation1.csv"
days = "days.csv"

def dataReader(csvFile):
  return np.genfromtxt(csvFile, delimiter=";",
                                usecols=[1,2,3,4,5,6,7],
                                converters={5: lambda s: 0 if s == b"-1" else float(s),
                                            7: lambda s: 0 if s == b"-1" else float(s)})

def datesReader(csvFile):
  return np.genfromtxt(csvFile, delimiter=";", usecols=[0])

'''
Add labels for 2000 year 
'''
def addLabels(dates, labels):
  for date in dates:
    if date < 20000301:
      labels.append("winter")
    elif 20000301 <= date < 20000601:
      labels.append("lente")
    elif 20000601 <= date < 20000901:
      labels.append("zomer")
    elif 20000901 <= date < 20001201:
      labels.append("herfst")
    else: # from 01-12 to end of year
      labels.append("winter")
'''
Add labels for 2001 year 
'''
def addLabels1(dates, labels):
  for date in dates:
    if date < 20010301:
      labels.append("winter")
    elif 20010301 <= date < 20010601:
      labels.append("lente")
    elif 20010601 <= date < 20010901:
      labels.append("zomer")
    elif 20010901 <= date < 20011201:
      labels.append("herfst")
    else: # from 01-12 to end of year
      labels.append("winter")

def findhigest(data):
  highest = []
  for i in range(len(data[0])):
    datalist = []
    for day in data:
      datalist.append(day[i])
    highest.append(max(datalist))
  return highest

def findlowest(data):
  lowest =[]
  for i in range(len(data[0])):
    datalist = []
    for day in data:
      datalist.append(day[i])
    lowest.append(min(datalist))
  return lowest

def normaliseer(data, lowest, highest):
  for day in data:
    for i in range(len(day)):
      day[i] -= lowest[i]
      day[i] = day[i] / (highest[i] - lowest[i])
