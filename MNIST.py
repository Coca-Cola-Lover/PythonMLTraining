
# coding: utf-8

# In[27]:


import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def delta_sigmoid(x):
    return (1-x)*x

def ReLU(x):
    return np.maximum(0,x)

def delta_ReLU(x):
    return 1 * (x>0)

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a

def tanh(x):
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

def delta_tanh(x):
    return 1 - x**2

def cross_error(output, target):
    delta = 1e-7 # オーバーフロー対策
    return -np.sum(target * np.log(output + delta))


class NeuralNetwork:
    def __init__(self,w_input_middle1,w_middle1_middle2,w_middle2_output):
        self.w_input_middle1 = w_input_middle1
        self.w_middle1_middle2 = w_middle1_middle2
        self.w_middle2_output = w_middle2_output
        
    def foward(self,x_input): # 順伝播
        '''
        入力層784+1個、第1中間層500+1個(sigmoid)、第2中間層200+1個(sigmoid)、出力層10個(ソフトマックス)
        '''
        x_input = np.append(x_input,1) # バイアスの挿入、ここではベクトルになっている
        self.x_input = np.reshape(x_input,[785,1]) # 縦ベクトルに変形
        self.middle1_in = np.dot(self.w_input_middle1,x_input)
        self.middle1_out = sigmoid(self.middle1_in)    
        #self.middle1_out = ReLU(self.middle1_in)
        
        self.middle1_out[0] = 1 # 1行目の出力を1として上書きし、バイアスとする。
        self.middle1_out = np.reshape(self.middle1_out,[501,1]) # 縦ベクトルに変形
        self.middle2_in = np.dot(self.w_middle1_middle2,self.middle1_out)
        self.middle2_out = sigmoid(self.middle2_in)
        #self.middle2_out = ReLU(self.middle2_in)
        
        self.middle2_out[0] = 1 # 1行目の出力を1として上書きし、バイアスとする。
        self.middle2_out = np.reshape(self.middle2_out,[201,1]) # 縦ベクトルに変形
        self.output_in = np.dot(self.w_middle2_output,self.middle2_out)
        self.output = softmax(self.output_in)

        return self.output,self.middle1_out,self.middle2_out

    def error(self,target):
        return self.output - target

    def grad_weight_middle2_output(self,error_sum):
        self.grad_w_middle2_output = np.dot(error_sum,self.middle2_out.T)
        self.delta_out = error_sum
        return self.grad_w_middle2_output,self.delta_out

    
    def grad_weight_middle1_middle2(self):
        self.E_h1_h2 = np.dot(self.w_middle2_output.T,self.delta_out)
        #self.delta_middle2 = self.E_h1_h2 * delta_ReLU(self.middle2_out)
        self.delta_middle2 = self.E_h1_h2 * delta_sigmoid(self.middle2_out)
        self.grad_w_middle1_middle2 = np.dot(self.delta_middle2,self.middle1_out.T)
        return self.grad_w_middle1_middle2,self.delta_middle2
 
    
    def grad_weight_input_middle1(self,x_input):
        self.E_in_h1 = np.dot(self.w_middle1_middle2.T,self.delta_middle2)
        #self.delta_middle1 = self.E_in_h1 * delta_ReLU(self.middle1_out)
        self.delta_middle1 = self.E_in_h1 * delta_sigmoid(self.middle1_out)
        self.grad_w_input_middle1 = np.dot(self.delta_middle1,self.x_input.T)
        return self.grad_w_input_middle1

    def new_weight(self):
        self.w_input_middle1 -= self.grad_w_input_middle1 * 0.05
        self.w_middle1_middle2 -= self.grad_w_middle1_middle2 * 0.05
        self.w_middle2_output -= self.grad_w_middle2_output * 0.05

if __name__ == '__main__':
    # データの準備
    mnist = datasets.fetch_mldata('MNIST original', data_home="./datasets/")
    data = mnist.data
    target = mnist.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.02)

    w_input_middle1 = np.random.randn(501,785)
    w_middle1_middle2 = np.random.randn(201,501)
    w_middle2_output = np.random.randn(10,201)

    NN = NeuralNetwork(w_input_middle1, w_middle1_middle2, w_middle2_output)
    loopnumber = 50000
    
    # 学習開始
    for i in range(2): # 合計100000枚の学習
        X_train, y_train = shuffle(X_train, y_train) # ループ毎にシャッフル
        for loop in range(loopnumber):
            x_input = X_train[loop]/255
            output = NN.foward(x_input)[0]
            #print(output)

            target = int(y_train[loop]) # 整数型に変換、これがないとエラーになる
            target = np.eye(10)[target] # one hot vector
            target = np.reshape(target,[10,1])
            #print(target)

            loss = NN.error(target) # 誤差の蓄積
            #print(loss)

            crosserror = cross_error(output,target)
            #print(crosserror)

            # 逆伝播
            NN.grad_weight_middle2_output(loss)
            NN.grad_weight_middle1_middle2()
            NN.grad_weight_input_middle1(x_input)
            # 更新
            NN.new_weight()
            
            loop_sum = loopnumber*i+loop
            print(loop_sum)
            
            # 10000周ごとにテストを行う
            if loop_sum % 1000 == 0 or loop_sum == 99999:
                X_test, y_test = shuffle(X_test, y_test)
                X_test_train, y_test_train = shuffle(X_train, y_train)
                clear_1 = 0
                clear_2 = 0
                for test in range(1000):
                    input_1 = X_test[test]/255
                    input_2 = X_test_train[test]/255
                    output_1 = NN.foward(input_1)
                    output_2 = NN.foward(input_2)
                    answer_test = np.argmax(output_1[0])
                    answer_train = np.argmax(output_2[0])
                    if answer_test == y_test[test]:
                        clear_1 += 1
                    if answer_train == y_test_train[test]:
                        clear_2 += 1
                acc_test = clear_1/1000
                acc_train = clear_2/1000
                print(acc_test)
                plt.plot(loop_sum, acc_test, 'ro')
                plt.plot(loop_sum, acc_train, 'bo')
    plt.show()

