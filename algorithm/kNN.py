import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor

def calc_with_knearestNeighbour():

    #reading the data
    gammas = h5.File('../data/gammas.hdf5','r')

    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

    gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df, on='array_event_id')

    data = shuffle(gamma_merge)

    # dropping unimportant information and isolating mc Data

    mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
    mc_data = data[mc_attributes]
    data = data.drop(mc_attributes, axis=1)

    drop_information = list(['telescope_type_name','x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                            'camera_name','camera_id','array_event_id','pointing_azimuth'])
    droped_data=data[drop_information]
    data = data.drop(drop_information, axis=1)

    truth = mc_data['mc_energy']

    # kNN with cross validate, train and predict

    kNN = KNeighborsRegressor(n_neighbors=6, weights='distance', )

    predictions = cross_val_predict(kNN, data, y=truth, cv=6)
    maximum = truth.idxmax()
    dim, ind = kNN.fit(data,truth).kneighbors(data.loc[maximum, :].to_frame().T)
    print(truth.iloc[ind[0]])
    # Regression with mean over same array_event_id

    ID = droped_data['array_event_id']
    pred_ID = pd.DataFrame({'predicted_energy':predictions, 'array_event_id':ID})
    unique_ID = ID.unique()
    truth_ID = pd.DataFrame({'mc_energy':truth, 'array_event_id':ID})
    prediction_mean = pd.Series([],name='predicted_energy')
    truth_unique = pd.Series([], name='mc_energy')

    for i in unique_ID:
        pred_mean = np.mean(pred_ID.predicted_energy[pred_ID.array_event_id == i])
        pred_mean = pd.Series(pred_mean, name='predicted_energy')
        prediction_mean = pd.concat([prediction_mean,pred_mean], ignore_index=True)

        y = truth_ID.mc_energy[truth_ID.array_event_id == i].iloc[0]
        y = pd.Series(y, name='mc_energy')
        truth_unique = pd.concat([truth_unique, y], ignore_index=True)

    #plotting the predictions without mean

    bin_edges = np.linspace(0,40,30)
    plt.hist2d(predictions, truth.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid()
    plt.colorbar()
    plt.plot([0,40],[0,40],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("k nearest Neighbours Regression for energy estimation")
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    #plt.show()
    plt.savefig('plots/kNN_Regression.pdf')
    plt.close()

    error = (predictions-truth.values)**2
    bin_edges = np.linspace(0,10,30)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV²$')
    plt.ylabel('counts')
    plt.title('the error of the kNN Regression for Energy estimation')
    #plt.show()
    plt.savefig('plots/kNN_Regression_errors.pdf')
    plt.close()

    #plots with mean

    bin_edges = np.linspace(0,40,30)
    plt.hist2d(prediction_mean, truth_unique.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid()
    plt.colorbar()
    plt.plot([0,40],[0,40],color="grey", label= "correct prediction")
    plt.legend()
    plt.title("kNN Regression for energy estimation with mean")
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    #plt.show()
    plt.savefig('plots/kNN_Regression_mean.pdf')
    plt.close()

    error = (prediction_mean-truth_unique.values)**2
    bin_edges = np.linspace(0,10,30)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.title('the error of the kNN for Energy estimation with mean')
    #plt.show()
    plt.savefig('plots/kNN_Regression_errors_mean.pdf')
    plt.close()



calc_with_knearestNeighbour()
