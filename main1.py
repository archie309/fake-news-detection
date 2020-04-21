import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 


df = pd.read_csv("C:\\Users\\DELL\\Desktop\\Python Project\\Dataset\\train.csv")
print("The shape of the Dataset is  : ",df.shape)
#(20800, 5) - > specifies 20800 rows and 5 columns ie attributes or labels in dataset 
print(df.head(10))

labels=df.label
print(labels.head(10))

x = df['text']
y = labels
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.25, random_state = 7)

vector=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vector.fit_transform(x_train.apply(lambda x: np.str_(x)))
tfidf_test = vector.transform(x_test.apply(lambda x: np.str_(x)))

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
y_pred = pac.predict(tfidf_test)
score =  accuracy_score(y_test,y_pred)
print(f'Accuracy : {round(score*100,2)}%')
print("\nConfussion Matrix :\n")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report : \n")
print(classification_report(y_test,y_pred))
