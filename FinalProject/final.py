


from sklearn import datasets
import numpy as np
import pandas as pd
import csv


labs = pd.read_csv('labs.csv', usecols=['SEQN','LBXGH'])
exams = pd.read_csv('examination.csv', usecols=['BMXWT','BMXBMI','BMXWAIST','BMXHT','OHDEXSTS'])#,'OHARNF','OHAROCDT','OHAROCGP','OHAROCOH'])

demo = pd.read_csv('demographic.csv', usecols=['RIAGENDR','RIDAGEYR'])
diet = pd.read_csv('diet.csv')
survey = pd.read_csv('questionnaire.csv', usecols=['PAQ710','PAQ715'])





#exams.drop(['SEQN'],axis = 1, inplace = True)
#demo.drop(['SEQN'],axis = 1, inplace = True)
diet.drop(['SEQN'],axis = 1, inplace = True)
#survey.drop(['SEQN'],axis = 1, inplace = True)

data= pd.concat([labs,exams],axis=1,join='inner')
data= pd.concat([data,demo],axis=1,join='inner')
data= pd.concat([data,survey],axis=1,join='inner')
data= pd.concat([data,diet],axis=1,join='inner')
