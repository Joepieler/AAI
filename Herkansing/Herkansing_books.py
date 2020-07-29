import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt

print(tf.__version__)

print("start time: " + datetime.now().strftime("%H:%M:%S"))
StartTime = datetime.now()

bookratings = np.genfromtxt("output.csv", delimiter=",", usecols=(0,1), loose=False, invalid_raise=False, dtype=str)
TotalLength = 121764


imagelist = []
for i in range(TotalLength):#92034):
    imagelist.append(cv2.imread("covers/" + str(i) + ".jpg"))

'''
Gets the bad and good ones
and filters the null images
'''
def GetGood(book_ratings, image_list):
    tmp_imagelist = []
    tmp_bookranking = []

    for i in range(TotalLength):
        if book_ratings[i][1] in ["bad", "good"]:
            try:
                if image_list[i].all() != None:
                    tmp_imagelist.append(imagelist[i])
                    if [book_ratings[i]] == "bad":
                         tmp_bookranking.append(0)
                    else:
                        tmp_bookranking.append(1)
            except(AttributeError):
                continue

    return tmp_bookranking, tmp_imagelist


BookRatings, ImageList = GetGood(bookratings, imagelist)

print(len(BookRatings))
TotalLength = len(BookRatings)

'''normaliseren van de data'''
'''0/255 to 0/1'''
def scale(X):
    if type(X) == np.ndarray:
        return (X[0] + X[1] + X[2]) / 3 / 255.0

TmpImageList = []
for Image in range(len(ImageList)):
    TmpImageList.append(np.ndarray([75,50,1], dtype=np.float64))
    #TmpImageList[Image].shape = ImageList[Image].shape
    for i in range(75):
        try:
            for j in range(50):
                TmpImageList[Image][i][j] = scale(ImageList[Image][i][j])

        except(IndexError):
            TmpImageList[Image][i][j] = 0
            continue



ImageList = TmpImageList

'''
split the set in a training set and a test set
first 2/3 is traing
last 1/3 is test
'''

'''Training'''
TrainingImages = np.ndarray([int(TotalLength / 3 * 2),75,50,1], dtype=np.float64)
TrainingRatings = np.ndarray(int(TotalLength / 3 * 2), dtype=np.uint8)

for i in range(int(TotalLength/3*2)):
    TrainingImages[i] = ImageList[i]
    TrainingRatings[i] = BookRatings[i]

print("lenght of the TraingSet: " + str(len(TrainingImages)))

'''Test'''
TestImages = np.ndarray([int(TotalLength / 3),75,50,1], dtype=np.float64)
TestRatings = np.ndarray(int(TotalLength / 3), dtype=np.uint8)

for i in range(int(TotalLength/3*2 + 1), TotalLength):
    TestImages[i - int(TotalLength / 3 * 2 + 1)] = ImageList[i]
    TestRatings[i - int(TotalLength / 3 * 2 + 1)] = BookRatings[i]


print("lenght of the TestSet: " + str(len(TestImages)))

print("test 0")

model0 = keras.Sequential([
    keras.layers.Flatten(input_shape=(75,50)),
    keras.layers.Dense(128, activation='softmax'),
    keras.layers.Dense(128, activation='softmax'),
    keras.layers.Dense(128, activation='softmax'),
    keras.layers.Dense(1)
])

model0.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model0.fit(TrainingImages, TrainingRatings, epochs=10)

print("time after training: " + datetime.now().strftime("%H:%M:%S"))
#print("Time it took" + (datetime.now()-StartTime).strftime("%H:%M:%S"))

TestLoss, TestAccuracy = model0.evaluate(TestImages,  TestRatings, verbose=2)
print("Test 0 accuracy: ", TestAccuracy)
#print("Time it took" + (datetime.now()-StartTime).strftime("%H:%M:%S"))




