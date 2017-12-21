import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV   #Perforing grid search

test_size = 0.33
seed = 42
train = pd.read_csv('train.csv')
test = pd.read_csv('pred.csv')

X = train.drop(['ID','Label'],axis=1)
y = train.Label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

pos = train[train.Label==1]['ID'].count()
nag = train[train.Label==0]['ID'].count()
print(pos,nag)
xgb_clf = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=(nag/pos),
 seed=27)
print('start train')
xgb_clf.fit(X_train,y_train,verbose=True)
print(xgb_clf)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#X_finished = test.drop(['ID','Label'],axis=1)
