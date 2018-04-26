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

def calc_with_RandomForestRegressor():

    # Import data in h5py
    gammas = h5.File("../data/gammas.hdf5","r")

    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

    #merging of array and telescope data and shuffle of proton and gamma
    data_merge = pd.merge(gamma_array_df,gamma_telescope_df,on="array_event_id")

    #calculate the mean scaled
    data_MSV1 = func.calc_mean_scaled_width_and_length(data_merge)
    #shuffel
    data_MSV = shuffle(data_MSV1)

    #prediction_attributes = list(['alt_prediction','az_prediction','core_x_prediction','core_y_prediction','gamma_energy_prediction_mean',
    #                                'gamma_energy_prediction_std_x','gamma_prediction_mean','gamma_prediction_std',
    #                                'gamma_energy_prediction','gamma_energy_prediction_std_y','gamma_prediction'])
    #prediction_data = data_MSV[prediction_attributes]
    #data_MSV = data_MSV.drop(prediction_attributes, axis=1)

    #print('Kai s prediction R2-score: ',r2_score(prediction_data['gamma_energy_prediction'],data['mc_energy']))

    #drop unimportant DATA
    mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
    mc_data = data_MSV[mc_attributes]
    data_MSV = data_MSV.drop(mc_attributes, axis=1)

    droped_information = list(['psi','phi','telescope_type_name','x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                                'camera_name','camera_id','pointing_azimuth','r','array_event_id'])
    droped_data = data_MSV[droped_information].copy(deep=True)
    data_MSV = data_MSV.drop(droped_information,axis=1)
    truth = mc_data['mc_energy']

    print ("erster RF: ...")
    print(list(data_MSV))
    #fit and predict
    RFr = RandomForestRegressor(max_depth=10, n_jobs=-1)
    X=data_MSV.values
    y=truth.values
    predictions = cross_val_predict(RFr, X, y, cv=10)


    # weighted mean
    data_wmean = data_MSV.copy(deep=True)
    data_wmean['mc_energy'] = truth
    ID_mean = droped_data['array_event_id'].copy(deep=True)
    data_wmean['array_event_id'] = ID_mean
    data_wmean['predictions'] = predictions
    data_wmean2 = func.weighted_mean_over_ID(data_wmean['intensity'], data_wmean)
    prediction_w_mean = data_wmean2['predicted_energy'].copy(deep=True)
    truth_w_mean = data_wmean2['mc_energy'].copy(deep=True)
    data_wmean2 = data_wmean2.drop('predictions', axis=1)

    # use the prediction_w_mean for another RF

    data_encaps = data_wmean2.copy(deep=True)
    truth_encaps = data_encaps['mc_energy'].copy(deep=True)
    ID_encaps = data_encaps['array_event_id'].copy(deep=True)
    data_encaps = data_encaps.drop('array_event_id', axis=1)
    data_encaps = data_encaps.drop('mc_energy', axis=1)

    print ("zweiter RF: ...")
    print(list(data_encaps))
    #fit and pred
    RFr2 = RandomForestRegressor(max_depth=10, n_jobs=-1)
    X=data_encaps.values
    y=truth_encaps.values
    predictions_encaps = cross_val_predict(RFr2, X, y, cv=10)

    print("Plots: ...")
    min_energy = 0.003
    max_energy = 340
    #PLOTS
        #Plots without mean
    plt.subplot(211)
    r2_1 = func.plot_hist2d(predictions,truth.values,min_energy,max_energy)
    plt.title("RF(with MSV)(R2score: %.2f)" % r2_1)

        #plots for weighted mean

            #intensity
    plt.subplot(223)
    r2_2 = func.plot_hist2d(prediction_w_mean.values,truth_w_mean.values,min_energy,max_energy)
    plt.title("RFr w mean(inty)(%0.2f)" % r2_2 )



        #plots for encapsulated RF
    plt.subplot(224)
    r2_3 = func.plot_hist2d(predictions_encaps,truth_encaps.values,min_energy,max_energy)
    plt.title("encap RFr(%.2f)" % r2_3)
    plt.subplots_adjust(wspace=0.45,hspace=0.45)
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_all.pdf')
    plt.close()

    #Error Plots
    func.plot_error(predictions_encaps,truth_encaps.values)
    plt.title('the error of the encapsulated RF for Energy estimation')
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_errors_MSV_encaps.pdf')
    plt.close()


    func.plot_error(predictions,truth.values)
    plt.title('error of RF(with MSV) for Energy estimation')
    plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_error.pdf")
    plt.close()


    func.plot_error(prediction_w_mean.values,truth_w_mean.values)
    plt.title('the error of the RF for Energy estimation with weighted mean(intensity)')
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_errors_w_mean_MSV.pdf')
    plt.close()


    #Print R2Score
    print('RandomForestRegressor:\n\t Coefficient for determination: %.2f \n' % r2_1,
            '\texplained_variance score: %.2f \n' % explained_variance_score(predictions,truth.values),
            '\tmean squared error: %.2f \n' % mean_squared_error(predictions,truth.values),
            'RF with weighted mean(intensity):\n\t Coefficient for determination: %.2f \n' % r2_2,
            '\texplained_variance score: %.2f \n' % explained_variance_score(prediction_w_mean.values,truth_w_mean.values),
            '\tmean squared error: %.2f \n' % mean_squared_error(prediction_w_mean.values,truth_w_mean.values),
            'encapsulated RF:\n\t Coefficient of determination: %.2f\n' % r2_3,
            '\texplained_variance score: %.2f \n' % explained_variance_score(predictions_encaps,truth_encaps.values),
            '\tmean squared error: %.2f \n' % mean_squared_error(predictions_encaps,truth_encaps.values),)



calc_with_RandomForestRegressor()
