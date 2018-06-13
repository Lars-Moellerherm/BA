import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import functions as func
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
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

                                        Decide if you want to consider the diffused gammas with --diffuse
                                            -default: True

                                        Decide how big your data should be with --size
                                            -you get --size events from gammas and diffuse gammas
                                             if you set --diffuse on True
                                            -default is 103663

                                        Decide if you wanna have the mean sclaled values for length and width
                                            -default. True
                                    '''))
parser.add_argument('--steps', type=int, default=2)
parser.add_argument('--size', type=int, default=103663, help="How much data you want to enquire?")
parser.add_argument('--diffuse', type=bool, default=False, help="Wanna have the diffuse gammas?")
parser.add_argument('--sv', type=bool, default=False, help="Wanna have the scaled values?")

def RF_regressor():

    args = parser.parse_args()
    data_size = args.size-1

    if(args.steps > 0 & args.steps < 4):
      data = func.reading_data(args.diffuse,data_size)

      if(args.sv):
        data = func.calc_scaled_width_and_length(data)

      data = shuffle(data)
      #data = gamma_merge
      # isolate mc data and drop unimportant information

      data, droped_data = func.drop_data(data)
      truth = droped_data['mc_energy']

      print("Finished reading data... \n")

      #train, test, train_truth, test_truth = train_test_split(data, truth, test_size = 0.5)

      RFr = RandomForestRegressor(max_depth=10, n_jobs=3,n_estimators=200)
      print("We use these attributes for the RF: \n ",list(data))
      X=data.values
      y=truth.values
      predictions = cross_val_predict(RFr, X, y, cv=10)

      z = np.array([predictions,y])
      np.savetxt("data/RFr_pred_data.txt",z.T)

      print('Coefficient of determination for the RandomForestRegressor: %.2f \n' % r2_score(predictions,y),
            '\t explained_variance score: %.2f \n' % explained_variance_score(predictions,truth.values),
            '\t mean squared error: %.2f \n' % mean_squared_error(predictions,y),
            "Finished predicting... \n")


    if( args.steps == 2):
        # Regression with mean over same array_event_id


        data['mc_energy'] = truth
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
        telescope = droped_data['telescope_type_name'].copy(deep=True)
        mask= telescope == 'LST'
        telescope.loc[mask]=23#size of the mirror
        mask = telescope == 'MST'
        telescope.loc[mask]=12
        mask = telescope == 'SST'
        telescope.loc[mask]=4
        telescope = telescope.to_frame('telescope_size')
        #telescope = telescope.reset_index()
        #data = data.reset_index()
        data = pd.concat([data.sort_index(),telescope.sort_index()],axis=1)
        #data = data.set_index(list(['run_id','array_event_id']))
        data = func.weighted_mean_over_ID(data['telescope_size'], data)
        prediction_w2_mean = data['predicted_energy']
        truth_w2_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)
        data = data.drop('telescope_size',axis=1)

            #telescope kind weight
        telescope_sens = droped_data['telescope_type_name'].copy(deep=True)
        telescope_sens = telescope_sens.to_frame()
        truth_w3 = droped_data['mc_energy']
        telescope_sens = pd.concat([telescope_sens,truth_w3],axis=1)
        mask = (telescope_sens['telescope_type_name'] == 'LST') & (telescope_sens['mc_energy'] > 3.0) # not in requiered energy range
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'LST') & (telescope_sens['mc_energy']>0.15) & (telescope_sens['mc_energy']<3) #not in full sensitivity
        telescope_sens[mask] = 1
        mask = (telescope_sens['telescope_type_name'] == 'LST') & (telescope_sens['mc_energy']<0.15) # not in requiered energy range
        telescope_sens[mask] = 2
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy'] > 50.0) # not in requiered energy range
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy'] < 0.08) # not in requiered energy range
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy']>5.0) & (telescope_sens['mc_energy']<50.0) #not in full sensitivity
        telescope_sens[mask] = 1
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy']>0.08) & (telescope_sens['mc_energy']<0.15) #not in full sensitivity
        telescope_sens[mask] = 1
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy']<5) & (telescope_sens['mc_energy']>0.15) # not in requiered energy range
        telescope_sens[mask] = 2
        mask = (telescope_sens['telescope_type_name'] == 'SST') & (telescope_sens['mc_energy'] > 300.0)
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'SST') & (telescope_sens['mc_energy'] < 1.0)
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'SST') & (telescope_sens['mc_energy']>1.0) & (telescope_sens['mc_energy']<5.0) #not in full sensitivity
        telescope_sens[mask] = 1
        mask = (telescope_sens['telescope_type_name'] == 'SST') & (telescope_sens['mc_energy']<300.0) & (telescope_sens['mc_energy']>5.0) # not in requiered energy range
        telescope_sens[mask] = 2
        telescope_sens = telescope_sens.drop('mc_energy',axis=1)
        data = pd.concat([data,telescope_sens.sort_index()],axis=1)
        data = func.weighted_mean_over_ID(data['telescope_type_name'], data)
        prediction_w3_mean = data['predicted_energy']
        truth_w3_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)
        data = data.drop('telescope_type_name',axis=1)

        # writing data

        z = np.array([prediction_mean.values,truth_mean.values])
        np.savetxt("data/RFr_pred_mean_data.txt",z.T)
        print(prediction_w_mean.shape,truth_w_mean.shape)
        z = np.array([prediction_w_mean.values,truth_w_mean.values])
        np.savetxt("data/RFr_pred_wI_mean_data.txt",z.T)

        z = np.array([prediction_w2_mean.values,truth_w2_mean.values])
        np.savetxt("data/RFr_pred_wT_mean_data.txt", z.T)

        z = np.array([prediction_w3_mean.values,truth_w3_mean.values])
        np.savetxt("data/RFr_pred_wS_mean_data.txt", z.T)

        print('\n Coefficient for determination for RF with mean: %.2f' % r2_score(prediction_mean.values,truth_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_mean.values,truth_mean.values),
        '\n Coefficient for determination for RF with weighted mean(intensity): %.2f' % r2_score(prediction_w_mean.values,truth_w_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_w_mean.values,truth_w_mean.values),
        '\n Coefficient for determination for Rf with weighted mean(telescope size): %.2f' % r2_score(prediction_w2_mean.values,truth_w2_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_w2_mean.values,truth_w2_mean.values),
        '\n Coefficient for determination for Rf with weighted mean(telescope sensitivity): %.2f' % r2_score(prediction_w3_mean.values,truth_w3_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_w3_mean.values,truth_w3_mean.values),
        "Finished with calculating the means (Step 2)...")

    if(args.steps == 3):
            #intensity squared weight
        weight2 = data['intensity']**2
        data = func.weighted_mean_over_ID(weight2, data)
        prediction_w4_mean = data['predicted_energy']
        truth_w4_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)

            #sqrt intensity weight
        weight3 = data['intensity']**(1/2)
        data = func.weighted_mean_over_ID(weight3, data)
        prediction_w5_mean = data['predicted_energy']
        truth_w5_mean = data['mc_energy']
        data = data.drop('predicted_energy',axis=1)

        print('\n Coefficient for determination for Rf with weighted mean(intensity squared): %.2f' % r2_score(prediction_w3_mean,truth_w3_mean),
        '\n Coefficient for determination for Rf with weighted mean(sqrt intensity): %.2f' % r2_score(prediction_w4_mean,truth_w4_mean),
        "Finished with step 3 ...")



if __name__ == '__main__':
    RF_regressor()
