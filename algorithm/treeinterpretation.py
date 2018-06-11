from treeinterpreter import treeinterpreter as ti
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import functions as func
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.utils import shuffle


data_size =10000
data = func.reading_data(True,data_size)
data = func.calc_mean_scaled_width_and_length(data)
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
    print("Prediction: ",prediction[i],"\n")
    print('Feature contribution: \n')
    n=0
    for i in names:
        print (i," : ", round(contrebutions[m,n],4), " , " ,round(feature[n],4))
        n+=1
    print("-"*20)
    m+=1


RF = RandomForestRegressor(max_depth=30,n_jobs=-1)
X=data.values
y=truth.values
trainX,testX,trainY,testY = train_test_split(X,y)
RF.fit(trainX,trainY)
feature = RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in RF.estimators_],
             axis=0)

indices = np.argsort(feature)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, names[indices[f]], feature[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), feature[indices],
       color="b", yerr=std[indices], align="center",)
plt.xticks(range(X.shape[1]),[names[i] for i in indices],rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig("plots/feautureimportance.jpg")
plt.close()

data=np.array([tree.feature_importances_ for tree in RF.estimators_])
data=data[:,indices]
position_ticks = np.arange(0,X.shape[1])+1
plt.boxplot(data,notch=False)
plt.xticks(position_ticks,[names[i] for i in indices],rotation=90)
plt.tight_layout()
plt.savefig("plots/feautureimportance_boxplot.jpg")
plt.close()
