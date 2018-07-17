import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

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
plt.xlim(x_min+x_min/10,x_max+x_max/10)
plt.ylim(y_min+y_min/10,y_max+y_max/10)
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
w = np.array([np.random.random(i+2) for i in range(5)])

def deru(x_input, sample, w):
    y_target = sample[:,1]
    y_out = np.dot(w, x_input.T)
    errors = y_target - y_out
    return errors

def rmse(x_input, sample, w):
    errors = deru(x_input, sample, w)
    error2 = np.sum(errors**2)
    return np.sqrt(error2/sample.shape[0])

n=5 # 次数
time_sokutei = np.zeros(n)
start1 = time.time()
x1_sample = data_train
x_input = np.array([x1_sample[:,0]]).T
x = [x_input**i for i in range(n+1)] 
x2 = np.empty((x_input.shape[0],n+1)) #次数+1
for i in range(n+1):
    x2[:,i] = x[i][:,0]
for j in range (n):
    for i in range (1000):
        start = time.time()
        errors = deru(x2[0::,0:j+2], x1_sample, w[j])
        w[j] += 0.00001*np.dot(errors.T, x2[0::,0:j+2])
        time_sokutei[j] += time.time()-start

plt.title('result')

x1_plot = np.arange(x_min, x_max, 0.1)
x_plot = np.empty((n+1,len(x1_plot)))
y_plot = np.empty((n,len(x1_plot)))
x2_plot = [x1_plot**i for i in range(n+1)]
for i in range(n+1): # 次数+1
    x_plot[i,:] = x2_plot[i]
for i in range(n):
    y_plot[i] = np.dot(w[i], x_plot[0:i+2,0::])
    plt.plot(x1_plot,y_plot[i],label='{}-order prediction'.format(i+1))
x = data_import[:,0]
y = data_import[:,1]
plt.scatter(x,y, color='blue', s=5,label='data set')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(color='gray')
plt.legend()
plt.show()

# B.6.N (1 ≤ N ≤ 5) 次近似の結果からテスト誤差 (RMSE) を小数第3位まで求めよ.
x_sample = np.array(data_test[:,0]).T
#print(x_sample)
x_test = [x_sample**i for i in range(n+1)]
x2_test = np.empty((data_test.shape[0],n+1)) #次数+1
for i in range(n+1):
    x2_test[:,i] = x_test[i]

for i in range(n):
    rmse1 = rmse(x2_test[0::,0:i+2],data_test, w[i])
    print('N={} {:.3f} {:.3f}'.format(i+1,rmse1,time_sokutei[i]))
    
