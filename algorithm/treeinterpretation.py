from treeinterpreter import treeinterpreter as ti
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.utils import shuffle


data_size =10000
data = func.reading_data(True,data_size)
data = shuffle(data)
#drop unimportant DATA
data, droped_data = func.drop_data(data)
truth = droped_data['mc_energy']

#fit and predict
RFr = RandomForestRegressor(max_depth=10, n_jobs=-1)
X=data.values
y=truth.values
trainX,testX,trainY,testY = train_test_split(X,y)
RFr.fit(trainX,trainY)
feature = RFr.feature_importances_
prediction, bias, contrebutions = ti.predict(RFr,testX)
names = list(data)
m=0
for i in range(len(testX[:4])):
    print("Instance: ",i,'\n')
    print("Bias: ",bias[i],'\n')
    print('Feature contribution: \n')
    n=0
    for i in names:
        print (i," : ", round(contrebutions[m,n],4), " , " ,round(feature[n],4))
        n+=1
    print("-"*20)
    m+=1
