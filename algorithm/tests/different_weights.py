import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import functions as func
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import auc, roc_curve, confusion_matrix, r2_score
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier

def calc_with_RandomForestRegressor_dW():

    # Import data in h5py
    gammas = h5.File("../data/gamma_dl3.hdf5","r")

    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

    #merging of array and telescope data and shuffle of proton and gamma
    gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on="array_event_id")

    data = shuffle(gamma_merge)
    #data = gamma_merge
    # isolate mc data and drop unimportant information

    mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
    mc_data = data[mc_attributes]
    data = data.drop(mc_attributes, axis=1)

    droped_information = list(['telescope_type_name','x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                                'camera_name','camera_id','array_event_id','pointing_azimuth','r','phi','psi'])
    droped_data = data[droped_information]
    data = data.drop(droped_information,axis=1)

    prediction_attributes = list(['alt_prediction', 'az_prediction', 'core_x_prediction', 'core_y_prediction', 'gamma_energy_prediction_mean',
                               'gamma_energy_prediction_std_x', 'gamma_prediction_mean', 'gamma_prediction_std',
                               'gamma_energy_prediction', 'gamma_energy_prediction_std_y', 'gamma_prediction'])
    prediction_data = data[prediction_attributes]
    data = data.drop(prediction_attributes, axis=1)


    truth=mc_data['mc_energy']
    #train, test, train_truth, test_truth = train_test_split(data, truth, test_size = 0.5)

    RFr = RandomForestRegressor(max_depth=10, n_jobs=-1)
    predictions = cross_val_predict(RFr, data, truth, cv=10)

    # Regression with mean over same array_event_id


    data['mc_energy'] = truth
    data['array_event_id'] = droped_data['array_event_id']
    data['predictions'] = predictions

    data = func.mean_over_ID(data)
    prediction_mean = data['predicted_energy']
    truth_mean = data['mc_energy']
    data = data.drop('predicted_energy',axis=1)

    weight_attributes = list(['x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                            'camera_id','r','phi'])
    weight_attributes = weight_attributes + list(data)
    weight_attributes.remove('mc_energy')
    weight_attributes.remove('predictions')

    for att in weight_attributes:
        if(att in droped_information):
            data = func.weighted_mean_over_ID(droped_data[att],data)
        else:
            data = func.weighted_mean_over_ID(data[att],data)

        prediction_w_mean = data['predicted_energy'].copy(deep=True)
        truth_w_mean = data['mc_energy'].copy(deep=True)
        data = data.drop('predicted_energy',axis=1)

        print (' R2score von ',att,': %.2f' % r2_score(prediction_w_mean,truth_w_mean))


    print('\n r2 score auf bei normalem mean: %.2f \n' % r2_score(prediction_mean,truth_mean))



    #take Intensity

    for n in np.linspace(1,10,20):
        weight = data['intensity']**n
        data = func.weighted_mean_over_ID(weight, data)
        prediction_w_mean = data['predicted_energy'].copy(deep=True)
        truth_w_mean = data['mc_energy'].copy(deep=True)
        data = data.drop('predicted_energy',axis=1)

        print (' R2score von Intensity^',n,': %.2f' % r2_score(prediction_w_mean,truth_w_mean))


        weight = data['intensity']**(1/n)
        data = func.weighted_mean_over_ID(weight, data)
        prediction_w_mean = data['predicted_energy'].copy(deep=True)
        truth_w_mean = data['mc_energy'].copy(deep=True)
        data = data.drop('predicted_energy',axis=1)

        print ('\t R2score von Intensity^1/',n,': %.2f' % r2_score(prediction_w_mean,truth_w_mean))


calc_with_RandomForestRegressor_dW()
