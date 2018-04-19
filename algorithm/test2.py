import itertools
import functions as func
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import scipy as sc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import auc, roc_curve, confusion_matrix, r2_score
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier

gammas = h5.File("../data/gammas.hdf5","r")

# Converting to pandas
gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

#merging of array and telescope data and shuffle of proton and gamma
data = pd.merge(gamma_array_df,gamma_telescope_df,on="array_event_id")

#prediction_attributes = list(['alt_prediction','az_prediction','core_x_prediction','core_y_prediction','gamma_energy_prediction_mean',
#                                'gamma_energy_prediction_std_x','gamma_prediction_mean','gamma_prediction_std',
#                                'gamma_energy_prediction','gamma_energy_prediction_std_y','gamma_prediction'])
#prediction_data = data[prediction_attributes]
#data = data.drop(prediction_attributes, axis=1)


#calculate the mean scaled
data = shuffle(data)

#drop unimportant DATA
mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
mc_data = data[mc_attributes]
data = data.drop(mc_attributes, axis=1)

droped_information = list(['psi','phi','telescope_type_name','x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                            'camera_name','camera_id','pointing_azimuth','r','array_event_id'])
droped_data = data[droped_information]
data = data.drop(droped_information,axis=1)

truth = mc_data['mc_energy']


#fit and predict
RFr = RandomForestRegressor(max_depth=10, n_jobs=-1)
predictions = cross_val_predict(RFr, data, truth, cv=10)


prediction_w_mean, truth_unique = func.weighted_mean_over_ID(predictions, droped_data['array_event_id'], data['intensity'], truth)

data['array_event_id'] = droped_data['array_event_id']
data = pd.merge(data, prediction_w_mean, on='array_event_id')
data = data.drop('array_event_id', axis=1)

print(data, truth_unique)
