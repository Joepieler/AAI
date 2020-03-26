'''5.1
a.
u + V
4     0   4
1  +  9 = 10 = u + v
2    -6   -4

b.
u - w

-4    3    -7
1  - -2 =  3  = u - w
2    -1    3

c.
2v
    0    0
2 * 9 =  18 = 2v
   -6   -12

d.
3u - 2v + w

  -4   -12
3  1 =  3   = 3v
   2    6

   0    0
2  9  = 18 = 2v
  -6   -12

-12  0     3   -9
3  - 18 + -2 = -17 = 3u - 2v + w
6   -12   -1   17

e.
x + y - y = x

f.
2x + u = ?. 3d vector + 2d vector can't
'''

'''5.2
a.
u|v = u0 * v0 + u1 * v1 + u2 * v2 = -4 * 0 + 1 * 9 + 2 * -6 = -3

b.
v|u = v0 * u0 + v1 * u1 + v2 * u2 = 0 * -4 + 9 * 1 + -6 * 2 = -3 

c. 
w|x = can't because the x vector has only 2 items and w vector has 3

d. 
(u|v) w
uit al bleek dat het inproduct 
             3    -9
-3w = -3 *  -2 =   6
            -1     3

e.
((u|v)w|w) = ((-3)w|w)
            3   -9
-3w = -3 * -2 =  6 | w = -3w | w = -9 * 3 + 6 * -2 +  3 * -1 = -27 + -12 + -3 = -42 = ((u|v)w|w)
      -    -1    3 
      
f.
((x|y)w|w) 

((1 * 0 + 5 * 8)w|w)
((40)w|w)
            3   120
40w = 40 * -2 = -80 
           -1   -40

40w|w = 120 * 3 + -80 * -2 + -40 *-1 = -360 + 160 + 40 = -160 = ((x|y)w|w)

g.
Can't because (x|y)x vector is 2d and w  vector is 3d
'''

'''5.3
a. 
au
0 -1    3    -8 -8
1  0  * 5  = 8  0

had niet genoeg tijd om af te maken helaas.
'''

'''5.9'''
import numpy as np
import pickle , gzip , os
from urllib import request
from pylab import imshow , show , cm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras import optimizers
import tensorflow as tf

url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
    request.urlretrieve(url , "mnist.pkl.gz")

f = gzip.open('mnist.pkl.gz', 'rb')
train_set , valid_set , test_set = pickle.load(f, encoding ='latin1')
f.close()

def get_image ( number ):
    (X, y) = [img[ number ] for img in train_set ]
    return (np.array(X), y)

def view_image ( number ):
    (X, y) = get_image( number )
    print(" Label : %s" % y)
    imshow(X.reshape (28 ,28) , cmap=cm.gray)
    show()



#create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])


y_train = list()
for y in train_set[1]:
  tmp = [0 for i in range(10)]
  tmp[y] = 1
  y_train.append(np.array(tmp))


model.fit(train_set[0], train_set[1], epochs=3)

a,b = model.evaluate(valid_set[0], valid_set[1])
print(a,b)
a,b = model.evaluate(test_set[0], test_set[1])
print(a,b)