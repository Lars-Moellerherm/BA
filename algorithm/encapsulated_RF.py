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

def calc_with_RandomForestRegressor():

    # Import data in h5py
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

    data = func.calc_mean_scaled_width_and_length(data)

    #shuffel
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

    # weighted mean 

    data['mc_energy'] = truth
    data['array_event_id'] = droped_data['array_event_id']
    data = func.weighted_mean_over_ID(predictions, data['intensity'], data)
    prediction_w_mean = data['predicted_energy']
    truth_w_mean = data['mc_energy']


    # use the prediction_w_mean for another RF

    data = shuffle(data)
    truth_encaps = data['mc_energy']
    ID_encaps = data['array_event_id']
    data = data.drop('array_event_id', axis=1)
    data = data.drop('mc_energy', axis=1)
    print(data,truth_encaps)

    #fit and pred
    RFr2 = RandomForestRegressor(max_depth=10, n_jobs=-1)
    predictions_encaps = cross_val_predict(RFr2, data, truth_encaps, cv=10)



    min_energy = 0.003
    max_energy = 50
    #PLOTS
        #Plots without mean
    r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy)
    plt.title("RF(with MSV) for energy estimation(R2score: %.2f)" % r2_1)
    plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV.pdf")
    plt.close()

    func.plot_error(predictions,truth)
    plt.title('error of RF(with MSV) for Energy estimation')
    plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_error.pdf")
    plt.close()

        #plots for weighted mean

            #intensity
    r2_2 = func.plot_hist2d(prediction_w_mean,truth_w_mean,min_energy,max_energy)
    plt.title("RF Regression for energy estimation with weighted mean(intensity)(R2score: %0.2f)" % r2_2 )
    plt.savefig('plots/RF/mean_scaled/RF_Regression_w_mean_MSV.pdf')
    plt.close()


    func.plot_error(prediction_w_mean,truth_w_mean)
    plt.title('the error of the RF for Energy estimation with weighted mean(intensity)')
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_errors_w_mean_MSV.pdf')
    plt.close()

        #plots for encapsulated RF

    r2_3 = func.plot_hist2d(predictions_encaps,truth_encaps,min_energy,max_energy)
    plt.title("encapsulated RF Regression for energy estimation(R2score: %.2f)" % r2_3)
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_encaps.pdf')
    plt.close()

    func.plot_error(predictions_encaps,truth_encaps)
    plt.title('the error of the encapsulated RF for Energy estimation')
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_errors_MSV_encaps.pdf')
    plt.close()


    #Print R2Score
    print('Coefficient of determination for the RandomForestRegressor: %.2f' % r2_1,
        '\n Coefficient for determination for RF with weighted mean(intensity): %.2f' % r2_2,
        '\n Coefficient for determination for encapsulated RF: %.2f' % r2_3)



calc_with_RandomForestRegressor()
