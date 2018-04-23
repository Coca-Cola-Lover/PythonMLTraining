'''
https://note.nkmk.me/python-scikit-learn-svm-mnist/
から引用
'''

from sklearn import datasets, svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#データ準備
mnist = datasets.fetch_mldata('MNIST original', data_home="./datasets/")
data = mnist.data / 255
target = mnist.target

#シャッフル
data, target = shuffle(data, target)

#データ分割
#トレーニング5000回、テスト100回
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=100, train_size=5000)

#ソフトマージンSVM
clf = svm.SVC()

#X_trainをy_trainにフィッティング
clf.fit(X_train, y_train)

pre = clf.predict(X_test)
ac_score = metrics.accuracy_score(y_test, pre)
print(ac_score) #0.93
