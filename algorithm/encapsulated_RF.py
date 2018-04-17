import itertools
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

    ID = data['array_event_id']
    unique_ID = ID.unique()
    mean_scaled_width = pd.DataFrame({'mean_scaled_w':[],'array_event_id':[]})
    mean_scaled_length = pd.DataFrame({'mean_scaled_l':[],'array_event_id':[]})
    SW = (data.width - np.mean(data.width))/sc.stats.sem(data.width)
    scaled_width = pd.DataFrame({'scaled_w':SW, 'array_event_id':ID})
    SL = (data.length - np.mean(data.length))/sc.stats.sem(data.length)
    scaled_length = pd.DataFrame({'scaled_l':SL, 'array_event_id':ID})

    for i in unique_ID:
        ntels = data.num_triggered_telescopes[data.array_event_id == i].iloc[0]
        MSW = np.sum(scaled_width.scaled_w[scaled_width.array_event_id == i])
        MSW = MSW/np.sqrt(ntels)
        MSW = pd.Series(MSW)
        MSW = pd.DataFrame({'mean_scaled_w':MSW,'array_event_id':i})
        mean_scaled_width = pd.concat([mean_scaled_width,MSW], ignore_index=True)

        MSL = np.sum(scaled_length.scaled_l[scaled_length.array_event_id == i])
        MSL = MSL/np.sqrt(ntels)
        MSL = pd.Series(MSL)
        MSL = pd.DataFrame({'mean_scaled_l':MSL,'array_event_id':i})
        mean_scaled_length = pd.concat([mean_scaled_length,MSL], ignore_index=True)


    data = pd.merge(data,mean_scaled_width, on='array_event_id')
    data = pd.merge(data, mean_scaled_length, on='array_event_id')

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

    # Regression with mean over same array_event_id
    ID = droped_data['array_event_id']
    unique_ID = ID.unique()
    pred_ID = pd.DataFrame({'predicted_energy':predictions, 'array_event_id':ID})
    truth_ID = pd.DataFrame({'mc_energy':truth, 'array_event_id':ID})
    truth_unique = pd.Series([], name='mc_energy')

        # With weighted mean  INTENSITY
    weight = data['intensity']
    weight_ID = pd.DataFrame({'intensity': weight, 'array_event_id': ID})
    prediction_w_mean = pd.Series([], name='predicted_energy')

    for i in unique_ID:
        #INTENSITY
        x = weight_ID.intensity[weight_ID.array_event_id == i]
        pred_w_mean = np.average(pred_ID.predicted_energy[pred_ID.array_event_id == i], weights=x)
        pred_w_mean = pd.Series(pred_w_mean, name='predicted_energy')
        prediction_w_mean = pd.concat([prediction_w_mean,pred_w_mean], ignore_index=True)

        #make the truth unique
        y = truth_ID.mc_energy[truth_ID.array_event_id == i].iloc[0]
        y = pd.Series(y, name='mc_energy')
        truth_unique = pd.concat([truth_unique, y], ignore_index=True)

    #PLOTS
        #Plots without mean
    min_energy = np.log10(0.003)
    max_energy = np.log10(50)
    bin_edges = np.logspace(min_energy,max_energy,60)
    plt.hist2d(predictions, truth.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid(True,which='both')
    plt.colorbar()
    plt.plot([0.003,330],[0.003,330],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("Random Forest Regression for energy estimation(R2score: %.2f)" % r2_score(predictions,truth.values))
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(max_energy)
    plt.ylim(max_energy)
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV.pdf')
    plt.close()

    error = (predictions-truth.values)**2
    bin_edges = np.logspace(np.log10(0.0001),np.log10(5),60)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.xscale('log')
    plt.title('the error of the Random Forest for Energy estimation')
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_errors_MSV.pdf')
    plt.close()

        #plots for weighted mean

            #intensity
    bin_edges = np.logspace(min_energy,max_energy,60)
    plt.hist2d(prediction_w_mean, truth_unique.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid(True, which='both')
    plt.colorbar()
    plt.plot([0.003,330],[0.003,330],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("RF Regression for energy estimation with weighted mean(intensity)(R2score: %0.2f)" % r2_score(prediction_w_mean,truth_unique.values) )
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(max_energy)
    plt.ylim(max_energy)
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_w_mean_MSV.pdf')
    plt.close()

    error = (prediction_w_mean-truth_unique.values)**2
    bin_edges = np.logspace(np.log10(0.0001),np.log10(5),60)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.xscale('log')
    plt.title('the error of the RF for Energy estimation with weighted mean(intensity)')
    #plt.show()
    plt.savefig('plots/RF/mean_scaled/RF_Regression_errors_w_mean_MSV.pdf')
    plt.close()

    #Print R2Score
    print('Coefficient of determination for the RandomForestRegressor: %.2f' % r2_score(predictions,truth.values),
        '\n Coefficient for determination for RF with weighted mean(intensity): %.2f' % r2_score(prediction_w_mean,truth_unique.values))


calc_with_RandomForestRegressor()
