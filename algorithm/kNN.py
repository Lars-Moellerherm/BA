import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import auc, roc_curve, confusion_matrix, r2_score
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor

def calc_with_kNNRegressor():

    # Import data in h5py
    gammas = h5.File("../data/gammas.hdf5","r")

    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

    #merging of array and telescope data and shuffle of proton and gamma
    gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on="array_event_id")

    data = shuffle(gamma_merge)

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

    kNN = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
    predictions = cross_val_predict(kNN, data, truth, cv=10)

    # Regression with mean over same array_event_id

    ID = droped_data['array_event_id']
    pred_ID = pd.DataFrame({'predicted_energy':predictions, 'array_event_id':ID})
    unique_ID = ID.unique()
    truth_ID = pd.DataFrame({'mc_energy':truth, 'array_event_id':ID})
    truth_unique = pd.Series([], name='mc_energy')
    prediction_mean = pd.Series([],name='predicted_energy')#for the unweighted mean

    # With weighted mean  INTENSITY
    weight = data['intensity']
    weight_ID = pd.DataFrame({'intensity': weight, 'array_event_id': ID})
    prediction_w_mean = pd.Series([], name='predicted_energy')

    #size of telescope
    pd.options.mode.chained_assignment = None  # default='warn'

    telescope = droped_data['telescope_type_name']
    mask= telescope == 'LST'
    telescope.loc[mask]=23#size of the mirror
    mask = telescope == 'MST'
    telescope.loc[mask]=12
    mask = telescope == 'SST'
    telescope.loc[mask]=4
    telescope_ID = pd.DataFrame({'telescope_size': telescope, 'array_event_id': ID})
    prediction_w2_mean = pd.Series([], name='predicted_energy')



    for i in unique_ID:
        #mean
        pred_mean = np.mean(pred_ID.predicted_energy[pred_ID.array_event_id == i])
        pred_mean = pd.Series(pred_mean, name='predicted_energy')
        prediction_mean = pd.concat([prediction_mean,pred_mean], ignore_index=True)

        #weighted meaned

        #INTENSITY
        x = weight_ID.intensity[weight_ID.array_event_id == i]
        pred_w_mean = np.average(pred_ID.predicted_energy[pred_ID.array_event_id == i], weights=x)
        pred_w_mean = pd.Series(pred_w_mean, name='predicted_energy')
        prediction_w_mean = pd.concat([prediction_w_mean,pred_w_mean], ignore_index=True)

        #telescope size
        x = telescope_ID.telescope_size.loc[telescope_ID.array_event_id == i]
        pred_w2_mean = np.average(pred_ID.predicted_energy[pred_ID.array_event_id == i], weights=x)
        pred_w2_mean = pd.Series(pred_w2_mean, name='predicted_energy')
        prediction_w2_mean = pd.concat([prediction_w2_mean,pred_w2_mean], ignore_index=True)

        #make the truth unique
        y = truth_ID.mc_energy[truth_ID.array_event_id == i].iloc[0]
        y = pd.Series(y, name='mc_energy')
        truth_unique = pd.concat([truth_unique, y], ignore_index=True)



    r2_1=r2_score(predictions,truth.values)
    r2_2=r2_score(prediction_mean,truth_unique.values)
    r2_3=r2_score(prediction_w_mean,truth_unique.values)
    r2_4=r2_score(prediction_w2_mean,truth_unique.values)
    #Plots without mean
    min_energy = np.log10(0.003)
    max_energy = np.log10(50)
    bin_edges = np.logspace(min_energy,max_energy,60)
    plt.hist2d(predictions, truth.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid(True,which='both')
    plt.colorbar()
    plt.plot([0.003,330],[0.003,330],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("kNN Regression for energy estimation(R2score:%.2f)" % r2_1 )
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(max_energy)
    plt.ylim(max_energy)
    #plt.show()
    plt.savefig('plots/kNN/weighted/kNN_Regression.pdf')
    plt.close()

    error = (predictions-truth.values)**2
    bin_edges = np.logspace(np.log10(0.0001),np.log10(5),60)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.xscale('log')
    plt.title('the error of the kNN for Energy estimation')
    #plt.show()
    plt.savefig('plots/kNN/weighted/kNN_Regression_errors.pdf')
    plt.close()

    #plots with mean

    bin_edges = np.logspace(min_energy,max_energy,60)
    plt.hist2d(prediction_mean, truth_unique.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid(True,which='both')
    plt.colorbar()
    plt.plot([0.003,330],[0.003,330],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("kNN Regression for energy estimation with mean(R2score:%.2f)" % r2_2)
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(max_energy)
    plt.ylim(max_energy)
    #plt.show()
    plt.savefig('plots/kNN/weighted/kNN_Regression_mean.pdf')
    plt.close()

    error = (prediction_mean-truth_unique.values)**2
    bin_edges = np.logspace(np.log10(0.0001),np.log10(5),60)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.xscale('log')
    plt.title('the error of the kNN for Energy estimation with mean')
    #plt.show()
    plt.savefig('plots/kNN/weighted/kNN_Regression_errors_mean.pdf')
    plt.close()

    #plots for weighted mean

    #intensity
    bin_edges = np.logspace(min_energy,max_energy,60)
    plt.hist2d(prediction_w_mean, truth_unique.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid(True, which='both')
    plt.colorbar()
    plt.plot([0.003,330],[0.003,330],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("kNN Regression for energy estimation with weighted mean(intensity)(R2score:%.2f)" % r2_3)
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(max_energy)
    plt.ylim(max_energy)
    #plt.show()
    plt.savefig('plots/kNN/weighted/kNN_Regression_w_mean.pdf')
    plt.close()

    error = (prediction_w_mean-truth_unique.values)**2
    bin_edges = np.logspace(np.log10(0.0001),np.log10(5),60)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.xscale('log')
    plt.title('the error of the kNN for Energy estimation with weighted mean(intensity)')
    #plt.show()
    plt.savefig('plots/kNN/weighted/kNN_Regression_errors_w_mean.pdf')
    plt.close()

    #telescope_size
    bin_edges = np.logspace(min_energy,max_energy,60)
    plt.hist2d(prediction_w2_mean, truth_unique.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid(True, which='both')
    plt.colorbar()
    plt.plot([0.003,330],[0.003,330],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("kNN Regression for energy estimation with weighted mean(telescope_size)(R2score:%.2f)" % r2_4)
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(max_energy)
    plt.ylim(max_energy)
    #plt.show()
    plt.savefig('plots/kNN/weighted/kNN_Regression_w2_mean.pdf')
    plt.close()

    error = (prediction_w_mean-truth_unique.values)**2
    bin_edges = np.logspace(np.log10(0.0001),np.log10(5),60)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.xscale('log')
    plt.title('the error of the kNN for Energy estimation with weighted mean(telescope size)')
    #plt.show()
    plt.savefig('plots/kNN/weighted/kNN_Regression_errors_w2_mean.pdf')
    plt.close()

    #calculate the R2-score

    print('Coefficient of determination for the kNN: %.2f' % r2_1,
        '\n Coefficient of determination for kNN with mean: %.2f' % r2_2,
        '\n Coefficient for determination for kNN with weighted mean(intensity): %.2f' % r2_3,
        '\n Coefficient for determination for kNN with weighted mean(telescope size): %.2f' % r2_4)

calc_with_kNNRegressor()
