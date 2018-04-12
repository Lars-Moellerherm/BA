import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import auc, roc_curve, confusion_matrix
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

    data = shuffle(gamma_merge)

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
    train, test, train_truth, test_truth = train_test_split(data, truth, test_size = 0.5)

    RFr = RandomForestRegressor()
    predictions = cross_val_predict(RFr, data, truth, cv=10)

    # Regression with mean over same array_event_id

    ID = droped_data['array_event_id']
    pred_ID = pd.DataFrame({'predicted_energy':predictions, 'array_event_id':ID})
    unique_ID = ID.unique()
    truth_ID = pd.DataFrame({'mc_energy':truth, 'array_event_id':ID})
    prediction_mean = pd.Series([],name='predicted_energy')
    truth_unique = pd.Series([], name='mc_energy')

    # With weighted mean
    weight = droped_data['intensity']
    weight_ID = pd.DataFrame({'intensity': weight, 'array_event_id': ID})
    prediction_w_mean = pd.Series([], name='predicted_energy')

    #print(weight)


    for i in unique_ID:

        pred_mean = np.mean(pred_ID.predicted_energy[pred_ID.array_event_id == i])
        pred_mean = pd.Series(pred_mean, name='predicted_energy')
        prediction_mean = pd.concat([prediction_mean,pred_mean], ignore_index=True)

        #weighted meaned

        x = weight_ID.intensity[weight_ID.array_event_id == i]

        pred_w_mean = np.average(pred_ID.predicted_energy[pred_ID.array_event_id == i], weights=x)
        pred_w_mean = pd.Series(pred_w_mean, name='predicted_energy')
        prediction_w_mean = pd.concat([prediction_w_mean,pred_w_mean], ignore_index=True)

        y = truth_ID.mc_energy[truth_ID.array_event_id == i].iloc[0]
        y = pd.Series(y, name='mc_energy')
        truth_unique = pd.concat([truth_unique, y], ignore_index=True)


##cv.split gibt error: ValueError: Supported target types are: ('binary', 'multiclass'). Got 'continuous' instead.

    #cv = StratifiedKFold(n_splits=6)
    #for train, test in cv.split(data, truth):
    #    predictions = RFr.fit(data.iloc[train],truth.iloc[train]).predict(data.iloc[test])
#
#        test_ID = ID.iloc[test]
#        pred_ID = pd.DataFrame({'predicted_energy':predictions,'array_event_id':test_ID})
#
#        unique_ID = np.array(test_ID.unique())
#
#        x = truth.iloc[test]
#        truth_id = pd.concat([x,test_ID], axis=1)
#        truth_unique = pd.Series([], name='mc_energy')
#        prediction_mean = pd.Series([], name='mc_energy')
#        for i in unique_ID:
#            pred_mean=np.mean(pred_ID.predicted_energy[pred_ID.array_event_id==i])
#            pred_mean = pd.Series(pred_mean, index=i, name='mc_energy')
#            prediction_mean = pd.concat([prediction_mean,pred_mean])
#
#            y = truth_id.mc_energy[truth_id.array_event_id==i].iloc[0]
#            y = pd.Series(y, index=i, name='mc_energy')
#            truth_unique = pd.concat([truth_unique,y])

        #print('a')

    #Plots without mean

    bin_edges = np.linspace(0,40,30)
    plt.hist2d(predictions, truth.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid()
    plt.colorbar()
    plt.plot([0,40],[0,40],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("Random Forest Regression for energy estimation")
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    #plt.show()
    plt.savefig('plots/RF_Regression.pdf')
    plt.close()

    error = (predictions-truth.values)**2
    bin_edges = np.linspace(0,10,30)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.title('the error of the Random Forest for Energy estimation')
    #plt.show()
    plt.savefig('plots/RF_Regression_errors.pdf')
    plt.close()

    #plots with mean

    bin_edges = np.linspace(0,40,30)
    plt.hist2d(prediction_mean, truth_unique.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid()
    plt.colorbar()
    plt.plot([0,40],[0,40],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("Random Forest Regression for energy estimation with mean")
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    #plt.show()
    plt.savefig('plots/RF_Regression_mean.pdf')
    plt.close()

    error = (prediction_mean-truth_unique.values)**2
    bin_edges = np.linspace(0,10,30)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.title('the error of the Random Forest for Energy estimation with mean')
    #plt.show()
    plt.savefig('plots/RF_Regression_errors_mean.pdf')
    plt.close()

    #plots for weighted mean

    bin_edges = np.linspace(0,40,30)
    plt.hist2d(prediction_w_mean, truth_unique.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid()
    plt.colorbar()
    plt.plot([0,40],[0,40],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("Random Forest Regression for energy estimation with weighted mean")
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    #plt.show()
    plt.savefig('plots/RF_Regression_w_mean.pdf')
    plt.close()

    error = (prediction_w_mean-truth_unique.values)**2
    bin_edges = np.linspace(0,10,30)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.title('the error of the Random Forest for Energy estimation with weighted mean')
    #plt.show()
    plt.savefig('plots/RF_Regression_errors_w_mean.pdf')
    plt.close()


calc_with_RandomForestRegressor()
