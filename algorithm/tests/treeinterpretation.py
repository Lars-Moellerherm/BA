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
gammas = h5.File("../../data/3_gen/gammas.hdf5","r")

# Converting to pandas
gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))
max_size = gamma_array_df.shape[0]
if(data_size >= max_size):
    data_size = max_size-1

gamma_array_df = gamma_array_df.iloc[:data_size]
gamma_runs_df = gamma_runs_df.iloc[:data_size]
gamma_telescope_df = gamma_telescope_df.iloc[:data_size]


#merging of array and telescope data and shuffle of proton and gamma
gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on=list(["array_event_id",'run_id']))
gamma_merge = gamma_merge.set_index(['run_id','array_event_id'])
#there are some nan in width the needed to be deleted
gamma_merge = gamma_merge.dropna(axis=0)
data = gamma_merge
data = shuffle(data)

mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
mc_data = data[mc_attributes]
data = data.drop(mc_attributes, axis=1)

droped_information = list(['psi','phi','telescope_type_name','x','y','telescope_id','pointing_altitude',
                            'camera_name','camera_id','pointing_azimuth','r'])
droped_data = data[droped_information].copy(deep=True)
data = data.drop(droped_information,axis=1)

prediction_attributes = list(['h_max_prediction','alt_prediction','az_prediction','core_x_prediction','core_y_prediction'])
prediction_data = data[prediction_attributes]
data = data.drop(prediction_attributes, axis=1)
truth = mc_data['mc_energy']

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
