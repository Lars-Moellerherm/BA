import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
from confusionMatrix import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier

def calc_with_RandomForestRegressor():

    # Import data in h5py
    gammas = h5.File("MC_Daten/gammas.hdf5","r")
    protons = h5.File("MC_Daten/protons.hdf5","r")

    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

    proton_array_df = pd.DataFrame(data=dict(protons['array_events']))
    proton_runs_df = pd.DataFrame(data=dict(protons['runs']))
    proton_telescope_df = pd.DataFrame(data=dict(protons['telescope_events']))

    #merging of array and telescope data and shuffle of proton and gamma
    gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on="array_event_id")
    proton_merge = pd.merge(proton_array_df,proton_telescope_df,on="array_event_id")

    data = pd.concat([gamma_merge , proton_merge])

    data = shuffle(data)

    # isolate mc data and drop unimportant information

    mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
    mc_data = data[mc_attributes]
    data = data.drop(mc_attributes, axis=1)

    droped_information = list(['telescope_type_name','x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                                'camera_name','camera_id','array_event_id','pointing_azimuth'])
    droped_data = data[droped_information]
    data = data.drop(droped_information,axis=1)

    #splitting into train and test data

    truth=mc_data['mc_energy']
    #truth = truth.astype('bool')
    train, test, train_truth, test_truth = train_test_split(data, truth, test_size = 0.5)

    regr = RandomForestRegressor()
    test_pred = regr.fit(train,train_truth).predict(test)

    bin_edges = np.linspace(0,0.3,30)
    plt.hist2d(test_pred, test_truth.values,bins=bin_edges, cmap="viridis")
    plt.colorbar()
    plt.grid()
    plt.plot([0,0.3],[0,0.3],color="grey")
    plt.show()
    plt.close()
    #print(test_pred[:10],test_truth.values[:10])

    data2= pd.concat([data,mc_data['mc_corsika_primary_id']],axis=1)
    train2, test2, train_truth2, test_truth2 = train_test_split(data2, truth, test_size = 0.5)

    regr2 = RandomForestRegressor()
    test_pred2 = regr2.fit(train2,train_truth2).predict(test2)

    bin_edges = np.linspace(0,0.3,30)
    plt.hist2d(test_pred, test_truth.values,bins=bin_edges, cmap="viridis")
    plt.colorbar()
    plt.grid()
    plt.plot([0,0.3],[0,0.3],color="grey")
    plt.show()
    plt.close()



calc_with_RandomForestRegressor()
