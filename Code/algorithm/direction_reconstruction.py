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

                                        Decide between mean scaled values and not with --msv
                                            -default: True

                                        Decide if you want to consider the diffused gammas with --diffuse
                                            -default: True

                                        Decide how big your data should be with --size
                                            -you get --size events from gammas and diffuse gammas
                                             if you set --diffuse on True
                                            -default is 103663
                                    '''))
parser.add_argument('--msv', type=bool, default=False, help="Wanna have the Mean Scaled Value?")
parser.add_argument('--diffuse', type=bool, default=True, help="Wanna have the diffuse gammas?")
parser.add_argument('--size', type=int, default=103663, help="How much data you want to enquire?")



def direction_recon():

    args = parser.parse_args()

    data_size = args.size-1


    # Import data in h5py
    gammas = h5.File("../data/3_gen/gammas.hdf5","r")
    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))
    max_size = gamma_array_df.shape[0]
    if(data_size >= max_size):
        data_size = max_size-1
    gamma_array_df = gamma_array_df.iloc[:data_size]
    gamma_runs_df = gamma_runs_df.iloc[:data_size]
    gamma_telescope_df = gamma_telescope_df.iloc[:data_size]
    #merging of array and telescope data and shuffle of proton and gamma
    gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on=list(["array_event_id",'run_id']))
    gamma_merge = gamma_merge.set_index(['run_id','array_event_id'])
    #there are some nan in width the needed to be deleted
    gamma_merge = gamma_merge.dropna(axis=0)
    data = gamma_merge


    if(args.diffuse):
        gammas_diffuse = h5.File("../data/3_gen/gammas_diffuse.hdf5","r")

        gamma_diffuse_array_df = pd.DataFrame(data=dict(gammas_diffuse['array_events']))
        max_size_diffuse = gamma_diffuse_array_df.shape[0]
        if(args.size-1 >= max_size_diffuse):
            data_size = max_size_diffuse-1
        gamma_diffuse_array_df = gamma_diffuse_array_df.iloc[:data_size]
        gamma_diffuse_runs_df = pd.DataFrame(data=dict(gammas_diffuse['runs']))
        gamma_diffuse_runs_df = gamma_diffuse_runs_df.iloc[:data_size]
        gamma_diffuse_telescope_df = pd.DataFrame(data=dict(gammas_diffuse['telescope_events']))
        gamma_diffuse_runs_df = gamma_diffuse_runs_df.iloc[:data_size]

        gamma_diffuse_merge = pd.merge(gamma_diffuse_array_df,gamma_diffuse_telescope_df,on=list(["array_event_id",'run_id']))

        gamma_diffuse_merge = gamma_diffuse_merge.dropna(axis=0)
        gamma_merge = gamma_merge.reset_index()
        data = pd.concat([gamma_merge,gamma_diffuse_merge])

        data = data.set_index(['run_id','array_event_id'])
        data = data.dropna(axis=1)
        print("Using diffused data...")




    print("Finished with reading Data ... \n")

    if(args.msv):

        #calculate the mean scaled
        data = func.calc_mean_scaled_width_and_length(data)

        print("Finished with calculating Mean Scaled Values ... \n")


    data = shuffle(data)



    #drop unimportant DATA
    mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
    mc_data = data[mc_attributes]
    data = data.drop(mc_attributes, axis=1)
    droped_information = list(['phi','telescope_type_name','telescope_id','pointing_altitude',
                                'camera_name','camera_id','pointing_azimuth','r'])
    droped_data = data[droped_information].copy(deep=True)
    data = data.drop(droped_information,axis=1)
    prediction_attributes = list(['h_max_prediction','alt_prediction','az_prediction','core_x_prediction','core_y_prediction'])
    prediction_data = data[prediction_attributes]
    data = data.drop(prediction_attributes, axis=1)

    






if __name__ == '__main__':
    direction_recon()
