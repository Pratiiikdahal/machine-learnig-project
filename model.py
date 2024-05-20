import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

#random seed

seed=42

iris_df=pd.read_csv('data/iris.csv')

x=iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=iris_df[['Species']]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=seed,stratify=y)      
clf=KNeighborsClassifier(n_neighbors=10)

clf.fit(xtrain,ytrain)

ypred=clf.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(f'Accuracy:{accuracy}')

joblib.dump(clf,"output_models/kn_model.sav")
