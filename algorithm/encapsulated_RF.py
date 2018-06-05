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
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=argparse._textwrap.dedent('''\
                                    What do you want to calculate?
                                    ------------------------------
                                        decide with --steps:
                                        1: Just the prediction from the RFRegressor
                                        with mean scaled values.
                                        2: The mean weighted with intensity and distance to core.
                                        3: The encapsulated Tree with the mean as new
                                        attribute.
                                        - deafult: 3

                                        Decide between mean scaled values and not with --msv
                                            -default: True

                                        Decide if you want to consider the diffused gammas with --diffuse
                                            -default: True

                                        Decide how big your data should be with --size
                                            -you get --size events from gammas and diffuse gammas
                                             if you set --diffuse on True
                                            -default is 103663
                                    '''))
parser.add_argument('--step', type=int, default=3)
parser.add_argument('--msv', type=bool, default=True, help="Wanna have the Mean Scaled Value?")
parser.add_argument('--diffuse', type=bool, default=True, help="Wanna have the diffuse gammas?")
parser.add_argument('--size', type=int, default=103663, help="How much data you want to enquire?")


def encaps_RF():
    args = parser.parse_args()

    data_size = args.size-1


    if(args.step > 0 & args.step < 4):

        data = func.reading_data(args.diffuse,data_size)



    print("Finished with reading Data ... \n")
    if(args.msv):

        #calculate the mean scaled
        data = func.calc_mean_scaled_width_and_length(data)


        print("Finished with calculating Mean Scaled Values ... \n")

    if(args.step > 0 & args.step < 4):
        data = shuffle(data)
        #drop unimportant DATA
        data, droped_data = func.drop_data(data)
        truth = droped_data['mc_energy']

        #fit and predict
        RFr = RandomForestRegressor(max_depth=10, n_jobs=-1)
        print("We use these attributes for the first RF: \n ",list(data))
        X=data.values
        y=truth.values
        predictions = cross_val_predict(RFr, X, y, cv=10)
        z=np.array([predictions,y])

        np.savetxt("data/encaps_pred_data.txt",z.T)

        print('RandomForestRegressor:\n\t Coefficient for determination: %.2f \n' % r2_score(predictions,truth.values),
                '\texplained_variance score: %.2f \n' % explained_variance_score(predictions,truth.values),
                '\tmean squared error: %.2f \n' % mean_squared_error(predictions,truth.values),
                "Finished with the first prediction ... \n")

    if(args.step > 1):
        # weighted mean
        data_wmean = data.copy(deep=True)
        data_wmean['mc_energy'] = truth
        data_wmean['predictions'] = predictions
        data_wmean = func.weighted_mean_over_ID(data_wmean['intensity'], data_wmean)
        prediction_w_mean = data_wmean['predicted_energy']
        truth_w_mean = data_wmean['mc_energy']
        data_w2mean = data_wmean.copy(deep=True)
        data_wmean = data_wmean.drop('predictions', axis=1)


        data_w2mean = data_w2mean.drop('predicted_energy', axis=1)
        data_w2mean = func.weighted_mean_over_ID(1/data_w2mean['distance_to_core'],data_w2mean)
        prediction_w2_mean = data_w2mean['predicted_energy']
        truth_w2_mean = data_w2mean['mc_energy']

        z=np.array([prediction_w_mean.values,truth_w_mean.values])
        np.savetxt("data/encaps_pred_w_mean_data.txt",z.T)

        z=np.array([prediction_w2_mean.values, truth_w2_mean.values])
        np.savetxt("data/encaps_pred_w2_mean_data.txt",z.T)

        print('RF with weighted mean(intensity):\n\t Coefficient for determination: %.2f \n' % r2_score(prediction_w_mean.values,truth_w_mean.values),
        '\texplained_variance score: %.2f \n' % explained_variance_score(prediction_w_mean.values,truth_w_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_w_mean.values,truth_w_mean.values))

        print('RF with weighted mean(distance to core):\n\t Coefficient for determination: %.2f \n' % r2_score(prediction_w2_mean.values,truth_w2_mean.values),
        '\texplained_variance score: %.2f \n' % explained_variance_score(prediction_w2_mean.values,truth_w2_mean.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_w2_mean.values,truth_w2_mean.values),
        "Finished with calculating the weighted mean ... \n")



    if(args.step == 3):
        # use the prediction_w_mean for another RF

        data_encaps = data_wmean.copy(deep=True)
        data_encaps = shuffle(data_encaps)
        truth_encaps = data_encaps['mc_energy'].copy(deep=True)
        data_encaps = data_encaps.drop('mc_energy', axis=1)

        #fit and pred
        RFr2 = RandomForestRegressor(max_depth=10, n_jobs=-1)
        print("We use these attributes for the second RF: \n ",list(data_encaps))
        X=data_encaps.values
        y=truth_encaps.values
        predictions_encaps = cross_val_predict(RFr2, X, y, cv=10)

        z=np.array([predictions_encaps,truth_encaps.values])

        np.savetxt("data/encaps_encaps_pred_data.txt",z.T)

        print('encapsulated RF:\n\t Coefficient of determination: %.2f\n' % r2_score(predictions_encaps,truth_encaps.values),
        '\texplained_variance score: %.2f \n' % explained_variance_score(predictions_encaps,truth_encaps.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(predictions_encaps,truth_encaps.values),
        "Finished with the encapsulated prediction \n")







if __name__ == '__main__':
	encaps_RF()
