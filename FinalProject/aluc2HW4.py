# -*- coding: utf-8 -*-
"""
Homework4:
This one singular file runs a variety of different machine learning methods
including: Perceptron, Logistic Regression, Support Vector Machine, Decision 
Tree Learning, Random Forest Tree Learning, and K Nearest Neighbors. By using
these machine learning methods, we can take the Iris dataset to make predictions
on different breeds of flowers. 
This code takes the full dataset, performs preprocessing to make the data more
accessible, train the machine with a smaller subset, and test the machine
learning methods with the entire set and prints out the number of misclassified
items and the accuracy of each method.

@editor: Andrew Luc, ID#1205939874
"""

from sklearn import datasets
import numpy as np


#Load all of the Iris dataset
iris = datasets.load_iris() 
X = iris.data              
y = iris.target

#Seperate training and test data, Use 30% of the data for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Scale and fit data to remove any bias for machine learning
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)                
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



def percfun(): #Linear Perceptron Machine Learning Method
    from sklearn.linear_model import Perceptron
    ppn = Perceptron( max_iter=60, tol=0.001, eta0=0.1, random_state=0)
    #max_iter=maximum number of passes for training data
    #tol= stopping criterion.
    #eta0=update multiplier constant
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)                
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

def logregfun(): #Logistic Regression Machine Learning Method
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1000.0, random_state=0)
    #C=inverse regularization strength
    lr.fit(X_train_std, y_train)
    
    y_pred = lr.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


def SVMfun(): #Support Vector Machine learning method
    from sklearn.svm import SVC
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    y_pred = svm.predict(X_test_std)                
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

def dtlfun(): #Decision Tree Learning machine learning method
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=3 ,random_state=0)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_test)                
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    
def rforestfun(): #Random Forest Tree machine learning method
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(criterion='entropy', n_estimators=5 ,random_state=0, n_jobs=2)
    forest.fit(X_train,y_train)
    y_pred = forest.predict(X_test)                
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

def neighfun(): #K Nearest Neighbors machine learning method
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
    knn.fit(X_train_std,y_train)
    y_pred = knn.predict(X_test_std)                
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    
# Dictionary to call a each function in a for loop
options = {0 : percfun,
           1 : logregfun,
           2 : SVMfun,
           3 : dtlfun,
           4 : rforestfun,
           5 : neighfun
}
# Dictionary to grab each function name in a for loop
option_names = {0 : "Perceptron",
           1 : "Logistic Regression",
           2 : "Support Vector Machine",
           3 : "Data Tree Learning",
           4 : "Random Tree Forests",
           5 : "K Nearest Neighbors"
}

# For loop to print each machine learning function and it's accuracy and
# misclassified samples
for index in range(0,6):
    choice = index
    print('Machine Learning Method: ', option_names[choice])
    options[choice]()
