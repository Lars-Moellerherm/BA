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

                                        Decide between mean scaled values and not with --sv
                                            -default: True

                                        Decide if you want to consider the diffused gammas with --diffuse
                                            -default: True

                                        Decide how big your data should be with --size
                                            -you get --size events from gammas and diffuse gammas
                                             if you set --diffuse on True
                                            -default is all data
                                    '''))
parser.add_argument('--sv', type=bool, default=True, help="Wanna have the Scaled Value?")
parser.add_argument('--diffuse', type=bool, default=True, help="Wanna have the diffuse gammas?")
parser.add_argument('--size', type=int, default=-1, help="How much data you want to enquire?")


def trafo():
    args = parser.parse_args()

    data_size = args.size-1


    data = func.reading_data(args.diffuse,data_size)



    print("Finished with reading Data ... \n")
    if(args.sv):

        #calculate the mean scaled
        data = func.calc_scaled_width_and_length(data)


        print("Finished with calculating Mean Scaled Values ... \n")


    data = shuffle(data)
    #drop unimportant DATA
    data, droped_data = func.drop_data(data)
    truth = data[['mc_energy','array_event_id','run_id']]
    data = data.drop('mc_energy',axis=1)

    #fit and predict
    RFr = RandomForestRegressor(max_depth=10, n_jobs=-1,n_estimators=100, oob_score=True, max_features='sqrt')
    train_i, test_i = train_test_split(data[['array_event_id','run_id']],test_size=0.66)
    X_train = data.loc[data[['array_event_id','run_id']].isin(train_i)[data[['array_event_id','run_id']].isin(train_i)==True].dropna().index]
    X_test = data.loc[data[['array_event_id','run_id']].isin(test_i)[data[['array_event_id','run_id']].isin(test_i)==True].dropna().index]
    y_train = truth.loc[truth[['array_event_id','run_id']].isin(train_i)[truth[['array_event_id','run_id']].isin(train_i)==True].dropna().index]
    y_test = truth.loc[truth[['array_event_id','run_id']].isin(test_i)[truth[['array_event_id','run_id']].isin(test_i)==True].dropna().index]
    X1 = X_train.drop(['array_event_id','run_id'],axis=1).values
    X2 = X_test.drop(['array_event_id','run_id'],axis=1).values
    y1 = y_train.drop(['array_event_id','run_id'],axis=1).values
    y2 = y_test.drop(['array_event_id','run_id'],axis=1).values

    ####### Transformation ###########
    y1=np.log(y1+3)

    print("We use these attributes for the first RF: \n ",list(X_train.drop(['array_event_id','run_id'],axis=1)))
    RFr.fit(X1, y1)
            ############### overfitting ####################
    print("The oob_score is: ",RFr.oob_score_)
            ############# feature importance ################
    feature = RFr.feature_importances_
    indices = np.argsort(feature)[::-1]
    names = list(X_train.drop(['array_event_id','run_id'],axis=1))
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X1.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, names[indices[f]], feature[indices[f]]))
    data1=np.array([tree.feature_importances_ for tree in RFr.estimators_])
    data1=data1[:,indices]
    position_ticks = np.arange(0,X1.shape[1])+1
    plt.boxplot(data1,notch=False)
    plt.xticks(position_ticks,[names[i] for i in indices],rotation=90)
    plt.tight_layout()
    plt.savefig("plots/feautureimportance_boxplot_trafo_firstForest.pdf")
    plt.close()
        ################# prediction###############
    predictions = RFr.predict(X2)
    print("Trainiert mit:",y_train.shape[0]," \t Getestet mit: ",y_test.shape[0])

    pred = pd.DataFrame({'prediction':predictions, 'mc_energy':y_test['mc_energy'], 'array_event_id':y_test['array_event_id'], 'run_id':y_test['run_id']})

    ######## Rücktransformation ###########
    predictions_rück = np.exp(predictions)-3

    z=np.array([predictions_rück,y2[:,0]])
    np.savetxt("data/trafo_pred_data.txt",z.T)

    pred_rück = pd.DataFrame({'prediction':predictions_rück, 'mc_energy':y_test['mc_energy'], 'array_event_id':y_test['array_event_id'], 'run_id':y_test['run_id']})
    print('RandomForestRegressor:\n\t Coefficient for determination: %.2f \n' % r2_score(predictions_rück,y2[:,0]),
            '\texplained_variance score: %.2f \n' % explained_variance_score(predictions_rück,y2[:,0]),
            '\tmean squared error: %.2f \n' % mean_squared_error(predictions_rück,y2[:,0]),
            "Finished with the first prediction ... \n")



    #########  Mittelwert mit dem noch nicht Rücktransformierten ######################
    X_test_w = X_test.set_index(['run_id','array_event_id'])
    pred_w = pred.set_index(['run_id','array_event_id'])
    data_w = pd.concat([X_test_w,pred_w],axis=1).reset_index()
    truth_grouped = y_test.drop_duplicates().set_index(['run_id','array_event_id'])

    x_grouped = data_w[['prediction','array_event_id','run_id']].groupby(by=['run_id','array_event_id'])
    pred_mean = x_grouped.mean()
    pred_mean.columns = ['mean_prediction']
    pred_mean = pd.concat([pred_mean,truth_grouped], axis=1)

    ####Rücktrafo ######
    pred_mean_rück = np.exp(pred_mean['mean_prediction'])-3
    pred_mean_rück = pd.concat([pred_mean_rück,truth_grouped], axis=1)
    z=np.array([pred_mean_rück['mean_prediction'].values,pred_mean_rück['mc_energy'].values])
    np.savetxt("data/trafo_pred_mean_data.txt",z.T)


    print('RF with mean:\n\t Coefficient for determination: %.2f \n' % r2_score(pred_mean_rück['mean_prediction'].values,pred_mean_rück['mc_energy'].values),
    '\texplained_variance score: %.2f \n' % explained_variance_score(pred_mean_rück['mean_prediction'].values,pred_mean_rück['mc_energy'].values),
    '\tmean squared error: %.2f \n' % mean_squared_error(pred_mean_rück['mean_prediction'].values,pred_mean_rück['mc_energy'].values))



    ################################# use the prediction_median for another RF###############################################################
    encaps_info = ['num_triggered_telescopes','num_triggered_lst','num_triggered_mst','num_triggered_sst','total_intensity','array_event_id','run_id']
    data_encaps = X_test[encaps_info].drop_duplicates().set_index(['run_id','array_event_id'])
    data_encaps = pd.concat([data_encaps,pred_mean], axis=1)

    ######## neue Attribute berechnen #########

        ######### Mittelwert der Energien nur für die LST's ###########
    pred = data_w[['prediction','array_event_id','run_id','telescope_type_id']]
    telescope_type = pred['telescope_type_id'].copy(deep=True)
    pred = pred.drop('telescope_type_id',axis=1)
    pred_lst = pred[telescope_type==1]
    prediction_lst_max = pred_lst.groupby(by=list(['run_id','array_event_id'])).max()
    prediction_lst_min = pred_lst.groupby(by=list(['run_id','array_event_id'])).min()
    prediction_lst = pred_lst.groupby(by=list(['run_id','array_event_id'])).mean()
    prediction_lst_std =  pred_lst.groupby(by=list(['run_id','array_event_id'])).std()
    prediction_lst_max = prediction_lst_max.rename(columns = {'prediction':'max_lst_pred'})
    prediction_lst_min = prediction_lst_min.rename(columns = {'prediction':'min_lst_pred'})
    prediction_lst = prediction_lst.rename(columns = {'prediction':'mean_lst_pred'})
    prediction_lst_std = prediction_lst_std.rename(columns = {'prediction':'std_lst_pred'})

    pred_mst = pred[telescope_type==2]
    prediction_mst_max = pred_mst.groupby(by=list(['run_id','array_event_id'])).max()
    prediction_mst_min = pred_mst.groupby(by=list(['run_id','array_event_id'])).min()
    prediction_mst = pred_mst.groupby(by=list(['run_id','array_event_id'])).mean()
    prediction_mst_std =  pred_mst.groupby(by=list(['run_id','array_event_id'])).std()
    prediction_mst = prediction_mst.rename(columns={'prediction':'mean_mst_pred'})
    prediction_mst_std = prediction_mst_std.rename(columns={'prediction':'std_mst_pred'})
    prediction_mst_max = prediction_mst_max.rename(columns = {'prediction':'max_mst_pred'})
    prediction_mst_min = prediction_mst_min.rename(columns = {'prediction':'min_mst_pred'})

    pred_sst = pred[telescope_type==3]
    prediction_sst_max = pred_sst.groupby(by=list(['run_id','array_event_id'])).max()
    prediction_sst_min = pred_sst.groupby(by=list(['run_id','array_event_id'])).min()
    prediction_sst = pred_sst.groupby(by=list(['run_id','array_event_id'])).mean()
    prediction_sst_std =  pred_sst.groupby(by=list(['run_id','array_event_id'])).std()
    prediction_sst = prediction_sst.rename(columns = {'prediction':'mean_sst_pred'})
    prediction_sst_std = prediction_sst_std.rename(columns = {'prediction':'std_sst_pred'})
    prediction_sst_max = prediction_sst_max.rename(columns = {'prediction':'max_sst_pred'})
    prediction_sst_min = prediction_sst_min.rename(columns = {'prediction':'min_sst_pred'})

    data_encaps = pd.concat([data_encaps,prediction_lst,prediction_lst_std,prediction_mst,prediction_mst_std,prediction_sst,prediction_sst_std,prediction_lst_min,
                            prediction_lst_max,prediction_mst_min,prediction_mst_max,prediction_sst_min,prediction_sst_max],axis=1)
    if(args.sv):
        msl = X_test[['scaled_length','run_id','array_event_id']].groupby(by=['run_id','array_event_id']).mean()
        msw = X_test[['scaled_width','run_id','array_event_id']].groupby(by=['run_id','array_event_id']).mean()
        msl=msl.rename(columns={'scaled_length':'mean_scaled_length'})
        msw=msw.rename(columns={'scaled_width':'mean_scaled_width'})
        sl_std = X_test[['scaled_length','run_id','array_event_id']].groupby(by=['run_id','array_event_id']).std()
        sw_std = X_test[['scaled_width','run_id','array_event_id']].groupby(by=['run_id','array_event_id']).std()
        sl_std=sl_std.rename(columns={'scaled_length':'std_scaled_length'})
        sw_std=sw_std.rename(columns={'scaled_width':'std_scaled_width'})
        data_encaps = pd.concat([data_encaps,msl,msw,sl_std,sw_std],axis=1)



        ##### if there is no lst or sst or mst who has seen this event, I set the mean and std on 0 and the std is Nan if there is just one prediction
    data_encaps = data_encaps.fillna(0).reset_index()
    data_encaps = shuffle(data_encaps)
    truth_encaps = data_encaps[['mc_energy']].copy(deep=True)
    data_encaps = data_encaps.drop(['mc_energy','array_event_id','run_id'], axis=1)
    #fit and pred
    RFr2 = RandomForestRegressor(max_depth=10, n_jobs=-1,n_estimators=100,oob_score=True, max_features='sqrt')
    print("We use these attributes for the second RF: \n ",list(data_encaps))
    X2_train, X2_test, y2_train, y2_test = train_test_split(data_encaps,truth_encaps,test_size=0.5)

    ######### Trafo ###############
    y2_train = np.log(y2_train+3)

    RFr2.fit(X2_train.values,y2_train.values)

            ############### overfitting ####################
    print("The oob_score is: ",RFr2.oob_score_)

            ############# feature importance ################
    feature = RFr2.feature_importances_
    std = np.std([tree.feature_importances_ for tree in RFr2.estimators_],
                 axis=0)
    indices = np.argsort(feature)[::-1]
    names = list(data_encaps)
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X2_train.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, names[indices[f]], feature[indices[f]]))


    data2=np.array([tree.feature_importances_ for tree in RFr2.estimators_])
    data2=data2[:,indices]
    position_ticks = np.arange(0,X2_train.shape[1])+1
    plt.boxplot(data2,notch=False)
    plt.xticks(position_ticks,[names[i] for i in indices],rotation=90)
    plt.tight_layout()
    plt.savefig("plots/feautureimportance_boxplot_trafo_secondForest.pdf")
    plt.close()


            ####### Predictions ################
    prediction_encaps = RFr2.predict(X2_test.values)

        ############ Rücktrafo ###########
    prediction_encaps = np.exp(prediction_encaps)-3

    print("Trainiert mit:",y2_train.shape[0]," \t Getestet mit: ",y2_test.shape[0])
    z=np.array([prediction_encaps,y2_test['mc_energy'].values])
    np.savetxt("data/trafo_encaps_pred_data.txt",z.T)

    print('encapsulated with median_prediction RF:\n\t Coefficient of determination: %.2f\n' % r2_score(prediction_encaps,y2_test.values),
    '\texplained_variance score: %.2f \n' % explained_variance_score(prediction_encaps,y2_test.values),
    '\tmean squared error: %.2f \n' % mean_squared_error(prediction_encaps,y2_test.values),
    "Finished with the encapsulated prediction \n")



if __name__ == '__main__':
    trafo()
