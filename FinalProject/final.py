'''
National Health and Nutrition Examination Survey (NHANES) Glossary:

LBXGH       - Glycohemoglobin, the main biological marker to diagnose diabetes
BMXWT       - Body weight
BMXBMI      - Body Mass Index
BMXWAIST    - Waist circumference
BMXHT       - Body height
BMXARMC     - Arm circumference
BMDAVSAD    - Saggital abdominal measurement
OHDEXSTS    - Overall oral health
RIAGENDR    - Gender
RIDAGEYR    - Age
DR1TSUGR    - Sugar intake
DR1TTFAT    - Fat intake
DR1TSODI    - Sodium intake
DR1TKCAL    - Calorie intake
DR1TCARB    - Carbohydrate intake
PAQ710      - How many hours spent watching TV
PAQ715      - How many hours spent playing video games
'''

from sklearn import datasets
import numpy as np
import pandas as pd


#Importing the data
labs = pd.read_csv('labs.csv', usecols=['LBXGH'])
exams = pd.read_csv('examination.csv', usecols=['BMXWT','BMXBMI','BMXWAIST','BMXHT','BMXARMC','BMDAVSAD','OHDEXSTS'])#
gender =pd.read_csv('demographic.csv', usecols=['RIAGENDR'])
age = pd.read_csv('demographic.csv', usecols=['RIDAGEYR'])
diet = pd.read_csv('diet.csv', usecols = ['DR1TSUGR','DR1TTFAT','DR1TSODI','DR1TKCAL','DR1TCARB',])
survey = pd.read_csv('questionnaire.csv', usecols=['PAQ710','PAQ715'])

#Excluding everyone 18+ years old
age[age>17]=np.nan

#Using the MayoClinic A1C test to classify diabetic/prediabetic/nondiabetic
    #Note: 3000,2000, and 1000 are used to "pre-classify" so that values aren't interrupting each other
labs[labs>6.4]=3000
labs[labs<5.7]=1000
labs[labs<100]=2000
#Classifying diabetic status with a simple 0,1, or 2
labs[labs==3000]=2  #Diabetic
labs[labs==2000]=1  #Prediabetic
labs[labs==1000]=0  #Nondiabetic
labs=labs.rename(columns = {'LBXGH':'A1C'})

#Concatenating the data
data= pd.concat([labs,exams],axis=1,join='inner')
data= pd.concat([data,age],axis=1,join='inner')
data= pd.concat([data,gender],axis=1,join='inner')
data= pd.concat([data,survey],axis=1,join='inner')
data= pd.concat([data,diet],axis=1,join='inner')

#Removing rows with NaN values
data = data.dropna()

#Seperating Training and test data. Using 30% of the data for training
#Using cross validation to diversify training sets
y=data.values[:,0]
X=data.values[:,1:]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

#Scaling and fitting data to remove any bias for the machine learning code
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

'''
#Using Principal Component Analysis to narrow down the data
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
X_train_std = pca.fit_transform(X_train_std)
X_test_std = pca.transform(X_test_std)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
'''
####
#Defining different machine learning methods into easily callable functions

def percfun():  # Linear Perceptron Machine Learning Method
    from sklearn.linear_model import Perceptron
    ppn = Perceptron(max_iter=60, tol=0.001, eta0=0.1, random_state=1)
    # max_iter=maximum number of passes for training data
    # tol= stopping criterion.
    # eta0=update multiplier constant
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

def logregfun():  # Logistic Regression Machine Learning Method
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, random_state=1)
    # C=inverse regularization strength
    lr.fit(X_train_std, y_train)

    y_pred = lr.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

def SVMfun():  # Support Vector Machine learning method
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', C=1.0, random_state=1)
    svm.fit(X_train_std, y_train)
    y_pred = svm.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

def dtlfun():  # Decision Tree Learning machine learning method
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

def rforestfun():  # Random Forest Tree machine learning method
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(criterion='entropy', n_estimators=7, random_state=1, n_jobs=3)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

def neighfun():  # K Nearest Neighbors machine learning method
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=7, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# Dictionary to call a each function in a for loop
options = {0: percfun,
           1: logregfun,
           2: SVMfun,
           3: dtlfun,
           4: rforestfun,
           5: neighfun
           }
# Dictionary to grab each function name in a for loop
option_names = {0: "Perceptron",
                1: "Logistic Regression",   ##
                2: "Support Vector Machine",
                3: "Data Tree Learning",
                4: "Random Tree Forests",
                5: "K Nearest Neighbors"
                }

# Running a for loop to determine the best machine learning method for this data
# Outputting method names, misclassified samples, and total accuracy

for index in range(0, 6):
    choice = index
    print('Machine Learning Method: ', option_names[choice])
    options[choice]()


# Generate Heatmap to show correlation between elements
import seaborn as sns
import matplotlib.pyplot as plt
colormap = plt.cm.viridis
plt.figure(figsize=(20,20))
sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, annot=True)