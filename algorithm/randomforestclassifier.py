import confusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from scipy import interp

def calc_with_RandomForestClassifier() :
    # Import data in h5py
    gammas = h5.File("../data/gammas.hdf5","r")
    protons = h5.File("../data/protons.hdf5","r")
    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

    proton_array_df = pd.DataFrame(data=dict(protons['array_events']))
    proton_runs_df = pd.DataFrame(data=dict(protons['runs']))
    proton_telescope_df = pd.DataFrame(data=dict(protons['telescope_events']))

    #merging of array and telescope data and shuffle of proton and gamma
    gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on="array_event_id")
    proton_merge = pd.merge(proton_array_df,proton_telescope_df,on="array_event_id")

    data = pd.concat([gamma_merge , proton_merge])

    data = shuffle(data)

    # isolate mc data and drop unimportant information

    mc_attributes = list(['mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id','mc_height_first_interaction'])
    mc_data = data[mc_attributes]
    data = data.drop(mc_attributes, axis=1)

    ID = data['array_event_id']

    droped_information = list(['telescope_type_name','x','y','telescope_event_id','telescope_id','run_id_y','run_id_x','pointing_altitude',
                                'camera_name','camera_id','array_event_id','pointing_azimuth'])
    droped_data = data[droped_information]
    data = data.drop(droped_information,axis=1)

    truth=mc_data['mc_corsika_primary_id']

    #fitting and predicting with the DecisionTreeClassifier
    clf = RandomForestClassifier()
    predictions = cross_val_predict(clf,data,truth,cv=10)

    #Plotting with the confusion Matrix

    #compute Confusion matrix
    cm = confusion_matrix(truth, predictions, labels=(0,101))

    class_names=('Gamma','Proton')
    #plt.figure()
    confusionMatrix.plot_confusion_matrix(cm, classes=class_names, normalize=True,
    title='Normalized confusion matrix')
    #plt.show()
    plt.savefig('plots/CM_RFClassifier.pdf')
    plt.close()

    # ROC curve for cross_validate
    #Code from : http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    cv = StratifiedKFold(n_splits=6)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    ID = pd.Series(ID)

    tprs2 = []
    aucs2 = []

    for train, test in cv.split(data, truth):
        propability_means = pd.DataFrame({'prob_1': [], 'prob_2': []})
        probas_ = clf.fit(data.iloc[train],truth.iloc[train]).predict_proba(data.iloc[test])
        fpr, tpr, thresholds = roc_curve(truth.iloc[test], probas_[:, 1], pos_label=101)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)

        #calculate the mean of the probas over every array_event_id

        test_ID = ID.iloc[test]
        probas = pd.DataFrame({'propability_1':probas_[:, 0], 'propability_2':probas_[:, 1], 'array_event_id':test_ID})
        unique_ID = np.array(test_ID.unique())

        x = truth.iloc[test]
        truth_id = pd.concat([x,test_ID],axis=1)
        truth_unique = pd.Series([],name='mc_corsika_primary_id')
        for i in unique_ID:
            prob1_mean=np.mean(probas.propability_1[probas.array_event_id==i])
            prob2_mean=np.mean(probas.propability_2[probas.array_event_id==i])#
            prob_mean = pd.DataFrame([{'prob_1': prob1_mean,'prob_2': prob2_mean}])
            propability_means = pd.concat([propability_means,prob_mean], ignore_index=True)

            y = truth_id.mc_corsika_primary_id[truth_id.array_event_id==i].iloc[0]
            y = pd.Series(y,name='mc_corsika_primary_id')
            truth_unique = pd.concat([truth_unique,y], ignore_index=True)

        fpr2, tpr2, threshold2 = roc_curve(truth_unique.values, propability_means.iloc[:, 1].values, pos_label=101)
        tprs2.append(interp(mean_fpr, fpr2, tpr2))
        tprs2[-1][0] = 0.0
        roc_auc2 = auc(fpr2, tpr2)
        aucs2.append(roc_auc2)



    # ROC for unmeaned try

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=0.8)

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic")
    plt.legend(loc="best")
    #plt.show()
    plt.savefig('plots/ROC_RFClassifier.pdf')
    plt.close()
    print(r'AUC without mean over event_id: %0.2f $\pm$ %0.2f' % (mean_auc,std_auc))

    #ROC for meaned try

    mean_tpr2 = np.mean(tprs2, axis=0)
    mean_tpr2[-1] = 1.0
    mean_auc2 = auc(mean_fpr,mean_tpr2)
    std_auc2 = np.std(aucs2)
    plt.plot(mean_fpr, mean_tpr2, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc2, std_auc2), lw=2, alpha=0.8)

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic for meaned propabilitys")
    plt.legend(loc="best")
    #plt.show()
    plt.savefig('plots/ROC_RFClassifier_meaned.pdf')
    plt.close()
    print(r'AUC with mean over event_id: %0.2f $\pm$ %0.2f' % (mean_auc2,std_auc2))


calc_with_RandomForestClassifier()
