import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

#np.random.seed(10)

def feedforward(w, x_input, N):
    x_bias = np.reshape(x_input,[N+1,1])
    return np.dot(w,x_bias)

def grad(err, x_input):
    grad_weight = err*x_input
    return grad_weight


# B.1.フォルダ (regression) 内のデータファイル (data.csv) を読み込め.
data_csv = pd.read_csv('../data/data.csv')
data_import = np.array(data_csv)


# B.2.読み込んだデータを可視化せよ.
x = data_import[:,0]
y = data_import[:,1]
plt.scatter(x,y, color='blue', s=5,label='data set')
x_max = np.max(data_import[:,0])
x_min = np.min(data_import[:,0])
y_max = np.max(data_import[:,1])
y_min = np.min(data_import[:,1])
plt.xlim(x_min-1,x_max+1)
plt.ylim(y_min-1,y_max+1)
plt.grid(color='gray')
plt.legend()
plt.show()


# B.3.読み込んだデータを訓練データとテストデータに分割せよ.
train_number = 800
data_import = shuffle(data_import)
data_train ,data_test = np.split(data_import,[train_number])

# B.4.N (1 ≤ N ≤ 5) 次関数を用いて回帰を行え.ただし,回帰には勾配降下法を用いよ.
# B.5.回帰の結果を元に,データと回帰線を可視化せよ.

# ウェイトの定義
w1 = np.random.random(2)
w2 = np.random.random(3)
w3 = np.random.random(4)
w4 = np.random.random(5)
w5 = np.random.random(6)
w = np.array([np.random.random(i+2) for i in range(5)])


#1次
start1 = time.time()
for i in range (1000):
    x1_sample = data_train
    x1_input = np.array([x1_sample[:,0]])
    bias = np.ones(train_number)
    x1_input = np.c_[x1_input.T,bias]
    y_target = x1_sample[:,1]
    y_out = np.dot(w1, x1_input.T)
    errors = y_target - y_out
#    print(errors.shape)
#    print(x1_input.T.shape)
    w1 += 0.001*np.dot(errors.T,x1_input)
time1 = time.time()-start1

x1_plot = np.arange(x_min-1, x_max+1, 0.1)
y1_plot = w1[0]*x1_plot+w1[1]
plt.plot(x1_plot,y1_plot,label='1-order prediction')
plt.title('result')

#2次
start2 = time.time()
for i in range(data_train.shape[0]):
    x2_sample = data_train[i]
    x2_input = np.array([x2_sample[0]**2,x2_sample[0],1])
    y_target = x2_sample[1]
    print(x2_input)
    y_out = feedforward(w2, x2_input, 2)
    errors = y_target - y_out
    w2 += 0.01*errors*x2_input
time2 = time.time()-start2
    
y2_plot = w2[0]*(x1_plot**2)+w2[1]*x1_plot+w2[2]
plt.plot(x1_plot,y2_plot,label='2-order prediction')

#5次
start5 = time.time()
for i in range(data_train.shape[0]):
    x5_sample = data_train[i]
    x5_input = np.array([x5_sample[0]**5,x5_sample[0]**4,x5_sample[0]**3,x5_sample[0]**2,x5_sample[0],1])
    y5_target = x5_sample[1]
    y5_out = feedforward(w5, x5_input, 5)
    errors = y5_target - y5_out
    w5 += 0.001*errors*x5_input
time5 = time.time()-start5
    
y5_plot = w5[0]*(x1_plot**5)+w5[1]*(x1_plot**4)+w5[2]*(x1_plot**3)+w5[3]*(x1_plot**2)+w5[4]*x1_plot+w5[5]

#3次
start3 = time.time()
for i in range(data_train.shape[0]):
    x3_sample = data_train[i]
    x3_input = np.array([x3_sample[0]**3,x3_sample[0]**2,x3_sample[0],1])
    y3_target = x3_sample[1]
    y3_out = feedforward(w3, x3_input, 3)
    errors = y3_target - y3_out
    w3 += 0.01*errors*x3_input
time3 = time.time()-start3
    
y3_plot = w3[0]*(x1_plot**3)+w3[1]*(x1_plot**2)+w3[2]*(x1_plot**1)+w3[3]
plt.plot(x1_plot,y3_plot,label='3-order prediction')

#4次
start4 = time.time()
for i in range(data_train.shape[0]):
    x4_sample = data_train[i]
    x4_input = np.array([x4_sample[0]**4,x4_sample[0]**3,x4_sample[0]**2,x4_sample[0],1])
    y4_target = x4_sample[1]
    y4_out = feedforward(w4, x4_input, 4)
    errors = y4_target - y4_out
    w4 += 0.001*errors*x4_input
time4 = time.time()-start4
    
y4_plot = w4[0]*(x1_plot**4)+w4[1]*(x1_plot**3)+w4[2]*(x1_plot**2)+w4[3]*x1_plot+w4[4]
plt.plot(x1_plot,y4_plot,label='4-order prediction')
plt.plot(x1_plot,y5_plot,label='5-order prediction')

x = data_import[:,0]
y = data_import[:,1]
plt.xlim(x_min-1,x_max+1)
plt.ylim(y_min-1,y_max+1)
plt.scatter(x,y, color='blue', s=5,label='data set')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(color='gray')
plt.legend()
plt.show()

# B.6.N (1 ≤ N ≤ 5) 次近似の結果からテスト誤差 (RMSE) を小数第3位まで求めよ.
#1次

print('N   RMSE  TIME')
error1 = 0
for i in range(data_test.shape[0]):
    x1t_sample = data_test[i]
    x1t_input = np.array([x1t_sample[0],1])
    y1t_target = x1t_sample[1]
    y1_out = feedforward(w1, x1t_input, 1)
    errors = y1t_target - y1_out
    error1 += errors**2
rmse1 = np.sqrt(error1/data_test.shape[0])
print('N=1','%.3f'%rmse1,'%.3f'%time1)

#5次
error5 = 0
for i in range(data_test.shape[0]):
    x5t_sample = data_test[i]
    x5t_input = np.array([x5t_sample[0]**5,x5t_sample[0]**4,x5t_sample[0]**3,x5t_sample[0]**2,x5t_sample[0],1])
    y5t_target = x5t_sample[1]
    y5_out = feedforward(w5, x5t_input, 5)
    errors = y5t_target - y5_out
    error5 += errors**2
rmse5 = np.sqrt(error5/data_test.shape[0])


#2次
error2 = 0
for jj in range(data_test.shape[0]):
    x2t_sample = data_test[jj]
    x2t_input = np.array([x2t_sample[0]**2,x2t_sample[0],1])
    y2t_target = x2t_sample[1]
    y2_out = feedforward(w2, x2t_input, 2)
    errors = y2t_target - y2_out
    error2 += errors**2
rmse2 = np.sqrt(error2/data_test.shape[0])
print('N=2','%.3f'%rmse2,'%.3f'%time2)

#3次
error3 = 0
for ll in range(data_test.shape[0]):
    x3t_sample = data_test[ll]
    x3t_input = np.array([x3t_sample[0]**3,x3t_sample[0]**2,x3t_sample[0],1])
    y3t_target = x3t_sample[1]
    y3_out = feedforward(w3, x3t_input, 3)
    errors = y3t_target - y3_out
    error3 += errors**2
rmse3 = np.sqrt(error3/data_test.shape[0])
print('N=3','%.3f'%rmse3,'%.3f'%time3)

#4次
error4 = 0
for mm in range(data_test.shape[0]):
    x4t_sample = data_test[mm]
    x4t_input = np.array([x4t_sample[0]**4,x4t_sample[0]**3,x4t_sample[0]**2,x4t_sample[0],1])
    y4t_target = x4t_sample[1]
    y4_out = feedforward(w4, x4t_input, 4)
    errors = y4t_target - y4_out
    error4 += errors**2
rmse4 = np.sqrt(error4/data_test.shape[0])
print('N=4','%.3f'%rmse4,'%.3f'%time4)

print('N=5','%.3f'%rmse5,'%.3f'%time5)
    
