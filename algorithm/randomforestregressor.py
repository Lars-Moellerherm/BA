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
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=argparse._textwrap.dedent('''\
                                    What do you want to calculate?
                                    ------------------------------
                                        decide with --steps:
                                        1: Just the prediction from the RFRegressor
                                        2: The mean and the weighted mean with intensity and telescope size.
                                        3: The mean weighted with intensity squared und por 1/2
                                    ''')
parser.add_argument('--steps', type=int, default=3)

def RF_regressor():

    args = parser.parse_args()

    if(args.steps > 0 & args.steps < 4):
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

      #prediction_attributes = list(['alt_prediction', 'az_prediction', 'core_x_prediction', 'core_y_prediction', 'gamma_energy_prediction_mean',
        #                        'gamma_energy_prediction_std_x', 'gamma_prediction_mean', 'gamma_prediction_std',
        #                         'gamma_energy_prediction', 'gamma_energy_prediction_std_y', 'gamma_prediction'])
      #prediction_data = data[prediction_attributes]
      #data = data.drop(prediction_attributes, axis=1)


      truth=mc_data['mc_energy']

      print("Finished reading data... \n")

      #train, test, train_truth, test_truth = train_test_split(data, truth, test_size = 0.5)

      RFr = RandomForestRegressor(max_depth=10, n_jobs=-1)
      X=data.values
      y=truth.values
      predictions = cross_val_predict(RFr, X, y, cv=10)

      z = np.array([predictions,y])
      np.savetxt("data/RFr_pred_data.txt",z.T)

      print('Coefficient of determination for the RandomForestRegressor: %.2f \n' % r2_score(predictions,y),
            '\tmean squared error: %.2f \n' % mean_squared_error(predictions,y),
            "Finished predicting... \n")


    if( args.steps == 2):
        # Regression with mean over same array_event_id


        data['mc_energy'] = truth
        data['array_event_id'] = droped_data['array_event_id']
        data['predictions'] = predictions

        data = func.mean_over_ID(data)
        prediction_mean = data['predicted_energy']
        truth_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)

            #Intensity weight
        data = func.weighted_mean_over_ID(data['intensity'], data)
        prediction_w_mean = data['predicted_energy']
        truth_w_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)

            #telescope size weight
        telescope = droped_data['telescope_type_name']
        mask= telescope == 'LST'
        telescope.loc[mask]=23#size of the mirror
        mask = telescope == 'MST'
        telescope.loc[mask]=12
        mask = telescope == 'SST'
        telescope.loc[mask]=4
        data['telescope_size'] = telescope
        data = func.weighted_mean_over_ID(data['telescope_size'], data)
        prediction_w2_mean = data['predicted_energy']
        truth_w2_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)

        # writing data

        z = np.array([prediction_mean.values,truth_mean.values])
        np.savetxt("data/RFr_pred_mean_data.txt",z.T)

        z = np.array([prediction_w_mean.values,truth_w_mean.values])
        np.savetxt("data/RFr_pred_wI_mean_data.txt",z.T)

        z = np.array([prediction_w2_mean.values,truth_w2_mean.values])
        np.savetxt("data/RFr_pred_wT_mean_data.txt", z.T)

        print('\n Coefficient for determination for RF with mean: %.2f' % r2_score(prediction_mean.values,truth_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_mean.values,truth_mean.values),
        '\n Coefficient for determination for RF with weighted mean(intensity): %.2f' % r2_score(prediction_w_mean.values,truth_w_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_w_mean.values,truth_w_mean.values),
        '\n Coefficient for determination for Rf with weighted mean(telescope size): %.2f' % r2_score(prediction_w2_mean.values,truth_w2_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_w2_mean.values,truth_w2_mean.values),
        "Finished with calculating the means (Step 2)...")

    if(args.steps == 3):
            #intensity squared weight
        weight2 = data['intensity']**2
        data = func.weighted_mean_over_ID(weight2, data)
        prediction_w3_mean = data['predicted_energy']
        truth_w3_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)

            #sqrt intensity weight
        weight3 = data['intensity']**(1/2)
        data = func.weighted_mean_over_ID(weight3, data)
        prediction_w4_mean = data['predicted_energy']
        truth_w4_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)

        print('\n Coefficient for determination for Rf with weighted mean(intensity squared): %.2f' % r2_score(prediction_w3_mean,truth_w3_mean),
        '\n Coefficient for determination for Rf with weighted mean(sqrt intensity): %.2f' % r2_score(prediction_w4_mean,truth_w4_mean),
        "Finished with step 3 ...")



if __name__ == __main__:
    RF_regressor()
