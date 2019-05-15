'''
逻辑回归：from sklearn.linear_model import LogisticRegression
朴素贝叶斯：from sklearn.naive_bayes import GaussianNB
K-近邻：from sklearn.neighbors import KNeighborsClassifier
决策树：from sklearn.tree import DecisionTreeClassifier
支持向量机：from sklearn import svm
'''

import numpy as np
from sklearn.utils import check_random_state
from sklearn import svm, datasets
import sklearn.model_selection as ms
import matplotlib.pyplot as plt

#load data
iris = datasets.load_iris()
rng = check_random_state(42)
perm = rng.permutation(iris.target.size)
iris_data = iris.data[perm]
iris_target = iris.target[perm]
print(iris_target)
#拆分数据
x_train, x_test, y_train, y_test = ms.train_test_split(iris_data, iris_target, random_state=1, train_size=0.6)

#训练模型
for id, k in enumerate(['linear', 'rbf']):
    clf = svm.SVC(kernel=k).fit(x_train, y_train)
    print(np.mean(clf.predict(x_test) == y_test))
    plt.figure(id)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=clf.predict(x_test))
    # 画图
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出前两个特征的散点图
    x1_min, x1_max = iris_data[:, 0].min(), iris_data[:, 0].max()  # 第0列的范围
    x2_min, x2_max = iris_data[:, 1].min(), iris_data[:, 1].max()  # 第一列的范围
    plt.xlabel(u'花萼长度', fontsize=13)
    plt.ylabel(u'花萼宽度', fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'鸢尾花SVM二特征分类', fontsize=15)

plt.figure(id+1)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)   # 圈中测试集样本

plt.show()


'''
svm.SVC():
sample:
    sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                    tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
                    decision_function_shape=None,random_state=None)

C：C-SVC的惩罚参数C,默认值是1.0.
    C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。
    C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
  　　0 – 线性：u'v
 　　 1 – 多项式：(gamma*u'*v + coef0)^degree
  　　2 – RBF函数：exp(-gamma|u-v|^2)
  　　3 –sigmoid：tanh(gamma*u'*v + coef0)
degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
probability ：是否采用概率估计？.默认为False
shrinking ：是否采用shrinking heuristic方法，默认为true
tol ：停止训练的误差值大小，默认为1e-3
cache_size ：核函数cache缓存大小，默认为200
class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
verbose ：允许冗余输出
max_iter ：最大迭代次数。-1为无限制。
decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
random_state ：数据洗牌时的种子值，int值
主要调节的参数有：C、kernel、degree、gamma、coef0。

.fit()：用于训练SVM，具体参数已经在定义SVC对象的时候给出了，这时候只需要给出数据集X和X对应的标签Y即可。
.predict(): 基于以上的训练，对预测样本T进行类别预测，因此只需要接收一个测试集T，该函数返回一个数组表示个测试样本的类别。
'''


'''
train_test_split():
- train_data：所要划分的样本特征集 
- train_target：所要划分的样本结果
- test_size：三种类型。float，int，None，可选参数。
    float：0.0-1.0之间。代表测试数据集占总数据集的比例。
    int：代表测试数据集具体的样本数量。
    None：设置为训练数据集的补。
    default：默认设置为0.25，当且train_size没有设置的时候，如果有设置，则按照train_size的补来计算。
- train_size：三种类型。float，int，None。
    float：0.0-1.0之间，代表训练数据集占总数据集的比例。
    int：代表训练数据集具体的样本数量。
    None：设置为test_size的补。
    default：默认为None。
- random_state：三种类型。int，random state instance，None。
    int：是随机数生成器的种子。每次分配的数据相同。
    randomstate：random_state是随机数生成器的种子。
    None：随机数生成器是使用了np.random的random state。
    种子相同，产生的随机数就相同。种子不同，即使是不同的实例，产生的种子也不相同。
- shuffle：布尔值，可选参数。默认是None。在划分数据之前先打乱数据。如果shuffle=FALSE，则stratify必须是None。
- stratify：array-like或者None，默认是None。如果不是None，将会利用数据的标签将数据分层划分。
    若为None时，划分出来的测试集或训练集中，其类标签的比例也是随机的。
    若不为None时，划分出来的测试集或训练集中，其类标签的比例同输入的数组中类标签的比例相同，可以用于处理不均衡的数据集。
'''
