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
                                        Decide between mean scaled values and not with --sv
                                            -default: True
                                        Decide if you want to consider the diffused gammas with --diffuse
                                            -default: True
                                        Decide how big your data should be with --size
                                            -you get --size events from gammas and diffuse gammas
                                             if you set --diffuse on True
                                            -default is all data
                                    '''))
parser.add_argument('--step', type=int, default=3)
parser.add_argument('--sv', type=bool, default=True, help="Wanna have the Scaled Value?")
parser.add_argument('--diffuse', type=bool, default=True, help="Wanna have the diffuse gammas?")
parser.add_argument('--size', type=int, default=-1, help="How much data you want to enquire?")


def encaps_RF():
    args = parser.parse_args()

    data_size = args.size-1


    if(args.step > 0 & args.step < 4):

        data = func.reading_data(args.diffuse,data_size)



    print("Finished with reading Data ... \n")
    if(args.sv):

        #calculate the mean scaled
        data = func.calc_scaled_width_and_length(data)


        print("Finished with calculating Mean Scaled Values ... \n")

    if(args.step > 0 & args.step < 4):
        data = shuffle(data)
        #drop unimportant DATA
        data, droped_data = func.drop_data(data)

        #data['weight'] = droped_data['telescope_type_name']
        truth = data[['mc_energy','array_event_id','run_id']]
        data = data.drop('mc_energy',axis=1)

        #fit and predict
        RFr = RandomForestRegressor(max_depth=10, n_jobs=-1,n_estimators=100, oob_score=True, max_features='sqrt')
        train_i, test_i = train_test_split(data[['array_event_id','run_id']],test_size=0.66)
        import IPython; IPython.embed()
        X_train = data.loc[data[['array_event_id','run_id']].isin(train_i)[data[['array_event_id','run_id']].isin(train_i)==True].dropna().index]
        X_test = data.loc[data[['array_event_id','run_id']].isin(test_i)[data[['array_event_id','run_id']].isin(test_i)==True].dropna().index]
        y_train = truth.loc[truth[['array_event_id','run_id']].isin(train_i)[truth[['array_event_id','run_id']].isin(train_i)==True].dropna().index]
        y_test = truth.loc[truth[['array_event_id','run_id']].isin(test_i)[truth[['array_event_id','run_id']].isin(test_i)==True].dropna().index]

        X1 = X_train.drop(['array_event_id','run_id'],axis=1).values
        X2 = X_test.drop(['array_event_id','run_id'],axis=1).values
        y1 = y_train.drop(['array_event_id','run_id'],axis=1).values
        y2 = y_test.drop(['array_event_id','run_id'],axis=1).values

encaps_RF()
