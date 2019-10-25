# %% [code]
import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% [code]
import numpy as np
import pandas as pd
from sklearn import datasets

df = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris=datasets.load_iris()
print(type(iris))

# %% [code]
df.head()


# %% [code]
x=pd.DataFrame(df[["PetalLengthCm","PetalWidthCm"]])
x=x.values

mymap = {'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2 }
y=df.applymap(lambda s: mymap.get(s) if s in mymap else s)
y=pd.DataFrame(y[["Species"]])
y=y.values
y=y.reshape(y.shape[0])
y.shape

# %% [code]
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=3, random_state=42)

print(x_train.shape,"\n",x_test.shape)



# %% [code]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)

x_train_sc=sc.transform(x_train)
x_test_sc=sc.transform(x_test)


# %% [code]
#svm

from sklearn.svm import SVC
svm=SVC(kernel='rbf', random_state=42, gamma=.10,C=1.0)
svm.fit(x_train_sc,y_train)
print("svm train score : {:.2f}".format(svm.score(x_train_sc,y_train)))

print("svm test score : {:.2f}".format(svm.score(x_test_sc,y_test)))



# %% [code]
import warnings


def versiontuple(v):
    return tuple(map(int, (v.split("."))))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

# %% [code]
x_test.shape

# %% [code]
plot_decision_regions(x_test_sc, y_test, svm)

# %% [code]
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

# %% [code]
knn.fit(x_train_sc,y_train)

plot_decision_regions(x_test_sc,y_test,knn)


# %% [code]
# xgboost classifier

import xgboost as xg

xg_class=xg.XGBClassifier()
xg_class.fit(x_train_sc,y_train)

print("xgboost train score {:.02f}".format(xg_class.score(x_train_sc,y_train)),"\nxgboost test score : {:.02f}".format(xg_class.score(x_test_sc,y_test)))


# %% [code]
plot_decision_regions(x_test_sc,y_test,xg_class)
