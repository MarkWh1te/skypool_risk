import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('pred.csv')

X = train.drop(['ID','Label'],axis=1)
y = train.Label

psvm = SVC(kernel='linear',
            #class_weight='balanced', # penalize
            verbose=100,
            probability=True)

print('start train')
psvm.fit(X, y)
print('finished train')
pred_y = psvm.predict(X)
print(accuracy_score(y,pred_y))

# submit
X_test = test.drop(['ID','Label'],axis=1)
sub = psvm.predict(X_test)
sub.to_csv('sub.csv')
