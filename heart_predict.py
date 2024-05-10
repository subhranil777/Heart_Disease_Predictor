import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn import metrics
import warnings
import pickle
warnings.filterwarnings("ignore")
import math

heart_data=pd.read_csv("dataset_2180_cleveland.csv")
heart_data['thal']=pd.to_numeric(heart_data['thal'],errors='coerce',downcast='integer')
heart_data['thal'] = heart_data['thal'].fillna(heart_data['thal'].mean())

# X=heart_data.drop(columns='target',axis=1)
# Y=heart_data['target']

X = heart_data.iloc[:, :13].values
Y = heart_data.iloc[:, -1].values

print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=100)
log_reg=LogisticRegression()
log_reg.fit(X_train,Y_train)



inputt=(41,0,1,100,204,0,0,172,0,1.4,2,0,2)
final=[np.array(inputt)]

b = log_reg.predict(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))




