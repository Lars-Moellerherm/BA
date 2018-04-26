import itertools
import functions as func
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import scipy as sc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns


# Import data in h5py
gammas = h5.File("../data/gamma_dl3.hdf5","r")
# Converting to pandas
gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

#merging of array and telescope data and shuffle of proton and gamma
data_merge = pd.merge(gamma_array_df,gamma_telescope_df,on="array_event_id")

data_merge = shuffle(data_merge)
prediction_attributes = list(['alt_prediction','az_prediction','core_x_prediction','core_y_prediction','gamma_energy_prediction_mean',
                                'gamma_energy_prediction_std_x','gamma_prediction_mean','gamma_prediction_std',
                                'gamma_energy_prediction','gamma_energy_prediction_std_y','gamma_prediction'])
prediction_data = data_merge[prediction_attributes]
data_merge = data_merge.drop(prediction_attributes, axis=1)

mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
mc_data = data_merge[mc_attributes]
data_merge.drop(mc_attributes, axis=1, inplace=True)


droped_information = list(['psi','phi','telescope_type_name','x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                            'camera_name','camera_id','pointing_azimuth','r','array_event_id'])
droped_data = data_merge[droped_information].copy(deep=True)
data_merge.drop(droped_information,axis=1, inplace=True)
truth = mc_data['mc_energy'].copy(deep=True)

###### Scatter Plot #######
data_merge['mc_energy'] = np.log10(truth)
data_merge['intensity'] = np.log10(data_merge['intensity'])
data_merge['Index'] = data_merge.index
s = sns.pairplot(data_merge,x_vars=list(data_merge), y_vars='Index', kind='reg')
s.savefig('plots/regplot_data_index.jpg')
plt.close()
data_merge = data_merge.drop('mc_energy',axis=1)
data_merge = data_merge.drop('Index', axis=1)


#print(list(data_merge))

#fit and predict
#RFr = RandomForestRegressor(max_depth=10, n_jobs=-1)
#X=data_merge.values
#y=truth.values
#predictions = cross_val_predict(RFr, X, y, cv=10)

#mean, std = func.plot_rel_error(predictions,y)
#plt.title(r'Relativer Fehler ($\mu$: %.2f,$\sigma$: %.2f)' % (mean, std))
#plt.show()
#plt.close()

#mean_div, std_div = func.plot_trueDIVpred(predictions,y)
#plt.title(r'Verhältnis von Truth zu Prediction ($\mu$: %.2f, $\sigma$: %.2f)' % (mean_div, std_div))
#plt.show()
#plt.close()
