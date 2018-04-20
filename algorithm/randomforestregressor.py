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

def calc_with_RandomForestRegressor():

    # Import data in h5py
    gammas = h5.File("../data/gammas.hdf5","r")

    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

    #merging of array and telescope data and shuffle of proton and gamma
    gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on="array_event_id")

    #data = shuffle(gamma_merge)
    data = gamma_merge
    # isolate mc data and drop unimportant information

    mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
    mc_data = data[mc_attributes]
    data = data.drop(mc_attributes, axis=1)

    droped_information = list(['telescope_type_name','x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                                'camera_name','camera_id','array_event_id','pointing_azimuth','r'])
    droped_data = data[droped_information]
    data = data.drop(droped_information,axis=1)

    #prediction_attributes = list(['alt_prediction', 'az_prediction', 'core_x_prediction', 'core_y_prediction', 'gamma_energy_prediction_mean',
    #                            'gamma_energy_prediction_std_x', 'gamma_prediction_mean', 'gamma_prediction_std',
    #                            'gamma_energy_prediction', 'gamma_energy_prediction_std_y', 'gamma_prediction'])
    #prediction_data = data[prediction_attributes]
    #data = data.drop(prediction_attributes, axis=1)


    truth=mc_data['mc_energy']
    #train, test, train_truth, test_truth = train_test_split(data, truth, test_size = 0.5)

    RFr = RandomForestRegressor(max_depth=10, n_jobs=-1)
    predictions = cross_val_predict(RFr, data, truth, cv=10)

    # Regression with mean over same array_event_id


    #pd.options.mode.chained_assignment = None  # default='warn'


    data['mc_energy'] = truth
    data['array_event_id'] = droped_data['array_event_id']

    data = func.mean_over_ID(predictions, data)
    prediction_mean = data['predicted_energy']
    truth_mean = data['mc_energy']
    data = data.drop('predicted_energy',axis=1)


    data = func.weighted_mean_over_ID(predictions, data['intensity'], data)
    prediction_w_mean = data['predicted_energy']
    truth_w_mean = data['mc_energy']
    data = data.drop('predicted_energy',axis=1)

    telescope = droped_data['telescope_type_name']
    mask= telescope == 'LST'
    telescope.loc[mask]=23#size of the mirror
    mask = telescope == 'MST'
    telescope.loc[mask]=12
    mask = telescope == 'SST'
    telescope.loc[mask]=4
    data['telescope_size'] = telescope
    data = func.weighted_mean_over_ID(predictions, data['telescope_size'], data)
    prediction_w2_mean = data['predicted_energy']
    truth_w2_mean = data['mc_energy']
    data = data.drop('predicted_energy',axis=1)

    min_energy = 0.003
    max_energy = 50
    #PLOTS
        #Plots without mean
    r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy)
    plt.title("RF for energy estimation(R2score: %.2f)" % r2_1)
    plt.savefig("plots/RF/weighted/RF_Regression.pdf")
    plt.close()

    func.plot_error(predictions,truth)
    plt.title('error of RF for Energy estimation')
    plt.savefig("plots/RF/weighted/RF_Regression_error.pdf")
    plt.close()

        #Plots with mean
    r2_2 = func.plot_hist2d(prediction_mean,truth_mean,min_energy,max_energy)
    plt.title("RF for energy estimation with mean(R2score: %.2f)" % r2_2)
    plt.savefig("plots/RF/weighted/RF_Regression_mean.pdf")
    plt.close()

    func.plot_error(prediction_mean,truth_mean)
    plt.title('error of RF with mean for Energy estimation')
    plt.savefig("plots/RF/weighted/RF_Regression_mean_error.pdf")
    plt.close()

        #plots for weighted mean

            #intensity
    r2_3 = func.plot_hist2d(prediction_w_mean,truth_w_mean,min_energy,max_energy)
    plt.title("RF Regression with weighted mean(intensity)(R2score: %0.2f)" % r2_3 )
    plt.savefig('plots/RF/weighted/RF_Regression_w_mean.pdf')
    plt.close()


    func.plot_error(prediction_w_mean,truth_w_mean)
    plt.title('the error of the RF with weighted mean(intensity)')
    #plt.show()
    plt.savefig('plots/RF/weighted/RF_Regression_errors_w_mean.pdf')
    plt.close()

        #plots for encapsulated RF

    r2_4 = func.plot_hist2d(prediction_w2_mean,truth_w2_mean,min_energy,max_energy)
    plt.title("RF Regression with weighted mean(telescope size)(R2score: %.2f)" % r2_4)
    #plt.show()
    plt.savefig('plots/RF/weighted/RF_Regression_w2_mean.pdf')
    plt.close()

    func.plot_error(prediction_w2_mean, truth_w2_mean)
    plt.title('the error of RF with weighted mean (telescope size)')
    #plt.show()
    plt.savefig('plots/RF/weighted/RF_Regression_errors_w2_mean.pdf')
    plt.close()


    #Print R2Score
    print('Coefficient of determination for the RandomForestRegressor: %.2f' % r2_1,
        '\n Coefficient for determination for RF with mean: %.2f' % r2_2,
        '\n Coefficient for determination for RF with weighted mean(intensity): %.2f' % r2_3,
        '\n Coefficient for determination for Rf with weighted mean(telescope size): %.2f' % r2_4)

calc_with_RandomForestRegressor()
