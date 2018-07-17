import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sympy
import time

np.random.seed(10)

def ryoushi(x):
    return np.where(x>0,1,-1)

def step(x):
    return 1*(x>0)

def nn(w, x_input, number):
    x_bias = np.array(x_input).T
    return np.dot(w, x_bias)

# A.1.フォルダ (classification) 内のデータファイル (data.csv) を読み込め.
data_csv = pd.read_csv("../data/data.csv")
#print(data_csv[data_csv['y']==1])

# A.2.読み込んだデータを可視化せよ.
temp1 = data_csv[data_csv['y']==1]['x1']
temp2 = data_csv[data_csv['y']==1]['x2']
temp3 = data_csv[data_csv['y']==-1]['x1']
temp4 = data_csv[data_csv['y']==-1]['x2']
x1 = np.array(data_csv['x1'])
x2 = np.array(data_csv['x2'])
y = np.array(data_csv['y'])

x1_max = np.max(data_csv['x1'])
x1_min = np.min(data_csv['x1'])
x2_max = np.max(data_csv['x2'])
x2_min = np.min(data_csv['x2'])

plt.scatter(temp1,temp2,marker='.',label='class1')
plt.scatter(temp3,temp4,marker='.',label='class2')
plt.legend()
plt.xlim(x1_min-1,x1_max+1)
plt.ylim(x2_min-1,x2_max+1)
plt.title('data set')
plt.show()

# A.3.読み込んだデータを訓練データとテストデータに分割せよ.
train_number = 1800
test_number = 200
data_import = np.array(data_csv)
data_import = shuffle(data_import)
train_csv, test_csv = np.split(data_csv,[train_number])
#print(test_csv)
train, test = np.array(train_csv), np.array(test_csv)


# A.4.ADALINE Gradient Descent を用いてデータを2つのクラスに分類せよ.
#訓練
w = np.random.random(3)
w = np.reshape(w,(1,3))
learning_rate = 0.1
for i in range(train_number):
    train_sample = train[i]
    x_input_1, y_target = np.split(train_sample,[2])
    x_input = np.ones([3])
    x_input[0:2] = x_input_1[0:2]
    #print(x_input)
    #print(x_input)
    z = nn(w, x_input,1)
    errors = y_target-z
    w += learning_rate*errors*x_input
#テスト
test_input, target = np.hsplit(test,[2])
target = ryoushi(target).T
test_input = np.c_[test_input, np.ones([test_number])]
#print(target)
z_out = ryoushi(nn(w, test_input, test_number))
#print(z_out)


# A.5.分類の結果を元に,データと決定領域を可視化せよ.
xx1, xx2 = np.arange(x1_min-1,x1_max+1,0.01),np.arange(x2_min-1,x2_max+1,0.01)
Z = np.zeros((xx1.shape[0],xx2.shape[0]))
time_start = time.time()
xx1, xx2 = np.meshgrid(xx1,xx2)
Z = xx1*w[0,0]+xx2*w[0,1]+w[0,2]
Z = ryoushi(Z)
#print('{:.3f}'.format(time.time()-time_start))
plt.contourf(xx1,xx2,-Z, alpha=0.4)

temp11 = test_csv[test_csv['y']==1]['x1']
temp12 = test_csv[test_csv['y']==1]['x2']
temp13 = test_csv[test_csv['y']==-1]['x1']
temp14 = test_csv[test_csv['y']==-1]['x2']
x11 = np.array(test_csv['x1'])
x12 = np.array(test_csv['x2'])
y1 = np.array(test_csv['y'])

plt.grid(color='white')
plt.scatter(temp11,temp12,marker='.',label='class1')
plt.scatter(temp13,temp14,marker='.',label='class2')

plt.legend()
plt.xlim(x1_min-1,x1_max+1)
plt.ylim(x2_min-1,x2_max+1)
plt.title('result')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# A.6.テストデータから正答率 (accuracy) を小数第3位まで求めよ.
check = target-z_out
#print(check)
clear = np.sum(check == 0)
print('accuracy {:.3f}'.format(clear/test_number))

# A.7.各クラスの適合率 (precision) ,再現率 (recall) ,F値 (F-value) を小数第3位まで求めよ.
# class1
tpfp_index = np.where(z_out==1)
#print(tpfp_index[1].shape)
tp = np.sum(target[tpfp_index]==1)
precision = tp/(tpfp_index[1].shape)[0]
print('class1 precision {:.3f}'.format(precision))

tpfn_index = np.where(target==1)
recall = tp/(tpfn_index[1].shape)[0]
#tp = np.sum(z_out[tpfn_index]==1)
#print(tpfn_index[1])
#print(tp)
print('class1 recall {:.3f}'.format(recall))

f_measure = (2*recall*precision)/(recall+precision)
print('class1 F-measure {:.3f}'.format(f_measure))

# class2
fntn_index = np.where(z_out==-1)
#print(fntn_index)
tn = np.sum(target[fntn_index]==-1)
precision_2 = tn/(fntn_index[1].shape)[0]
#print(tn)
print('class2 precision {:.3f}'.format(precision_2))

fptn_index = np.where(target==-1)
#print(fptn_index)
recall_2 = tn/(fptn_index[1].shape)[0]
#print(recall)
print('class2 recall {:.3f}'.format(recall_2))

f_measure_2 = (2*recall_2*precision_2)/(recall_2+precision_2)
print('class2 F-measure {:.3f}'.format(f_measure_2))
