from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import time as time
def sigmoid(x):
    delta = 1e-7
    return 1.0/(1.0+np.exp((-1.0*x)-delta))
def softmax(x):
    x = x - np.max(x, axis=0)
    x = np.exp(x)
    x = x / np.sum(x, axis=0)
    return x
def sigmoid_dash(x):
    t = sigmoid(x)
    return t * (1.0-t)
def cross_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

start = time.time()
train_im = loadtxt("train-images.txt") #訓練データ,60000行784列

train_la = loadtxt("train-labels.txt") #訓練データ,60000行1列

#中間層,出力層のweight
w2 = np.random.randn(100,784)
w3 = np.random.randn(10,100)

#入力層:ユニット数784個
#中間層:ユニット数100個,活性化関数はシグモイド
#出力層:ユニット数10個,活性化関数はソフトマックス

train_label = np.matrix(train_la) 
train_image = np.array(train_im) / 256
X = np.array([0])
Y = np.array([0])


for i in range(1, 60000):
    x_dash = np.matrix(train_image[i,:]).T #入力を転置
    x1 = np.array(x_dash) #おまじない
    d_dash = train_label[:,i]
    z2 = sigmoid(w2.dot(x1))
    z3 = softmax(w3.dot(z2))

    if d_dash == 0:
        d = np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    elif d_dash == 1:
        d = np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
    elif d_dash == 2:
        d = np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]])
    elif d_dash == 3:
        d = np.array([[0], [0], [0], [1], [0], [0], [0], [0], [0], [0]])
    elif d_dash == 4:
        d = np.array([[0], [0], [0], [0], [1], [0], [0], [0], [0], [0]])
    elif d_dash == 5:
        d = np.array([[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]])
    elif d_dash == 6:
        d = np.array([[0], [0], [0], [0], [0], [0], [1], [0], [0], [0]])
    elif d_dash == 7:
        d = np.array([[0], [0], [0], [0], [0], [0], [0], [1], [0], [0]])
    elif d_dash == 8:
        d = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1], [0]])
    elif d_dash == 9:
        d = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [1]])

   
    grad_w3 = (z3-d).dot(z2.T)
    w3 -= 0.05*grad_w3
    delta1 = w3.T.dot(z3-d)
    grad_w2 = delta1 * sigmoid_dash(z2)
    grad_w21 = grad_w2.dot(x1.T)
    w2 -= 0.05*grad_w21

    loss = cross_error(z3, d)
    print(i)
    xi = np.array([i])
    yi = np.array([loss])
    X = np.hstack((X, xi))
    Y = np.hstack((Y, yi))
    plt.plot(i, cross_error(z3, d), 'ro')

x = np.array(X) #おまじない
y = np.array(Y) #おまじない
    
elapsed_time = time.time() - start
print(elapsed_time)
plt.plot(x, np.poly1d(np.polyfit(x, y, 15))(x), label='d=3')
plt.show()
