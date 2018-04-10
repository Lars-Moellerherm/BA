def fun():
    # Import data in h5py
    gammas = h5.File("MC_Daten/gammas.hdf5","r")
    protons = h5.File("MC_Daten/protons.hdf5","r")

    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

    proton_array_df = pd.DataFrame(data=dict(protons['array_events']))
    proton_runs_df = pd.DataFrame(data=dict(protons['runs']))
    proton_telescope_df = pd.DataFrame(data=dict(protons['telescope_events']))

    # Merging array_df and telescope_df
    gamma_merge = pd.merge(gamma_array_df, gamma_telescope_df,
                           on="array_event_id")
    proton_merge = pd.merge(proton_array_df, proton_telescope_df,
                            on="array_event_id")

    # Appending
    data = pd.concat([gamma_merge, proton_merge])

    # Shuffle data
    data = shuffle(data)

    # Dropping useless Attributes to DataFrame
    junk_attributes = list(['run_id_x', 'run_id_y', 'camera_name',
                            'telescope_event_id', 'telescope_id',
                            'telescope_type_name', 'x', 'y', 'phi', 'psi',
                            'pointing_altitude', 'pointing_azimuth'])
    junk = data[junk_attributes]
    data.drop(junk_attributes, axis=1, inplace=True)

    # Extracting target attribute truth and ID
    truth = data['mc_corsika_primary_id']
    ID = data['array_event_id']
    data.drop('array_event_id', axis=1, inplace=True)

    # Dropping MC-truth to DataFrame
    mc_attributes = list(['mc_alt', 'mc_az', 'mc_core_x', 'mc_core_y',
                          'mc_energy', 'mc_height_first_interaction'])
    mc = data[mc_attributes]
    data.drop(mc_attributes, axis=1, inplace=True)

    # computing correlation on truth
    corr = data.corr()
    target_corr = corr.mc_corsika_primary_id.abs()
    target_corr = target_corr.sort_values(ascending=False)
    target_corr.drop('mc_corsika_primary_id', inplace=True)

    # Dropping truth from data
    data.drop('mc_corsika_primary_id', axis=1, inplace=True)

    print('------------------------------------------------------------------')
    print('correlation with target attribute:')
    print(target_corr)
    print('------------------------------------------------------------------')

    # Binarizing data and converting to boolean (0 = gamma, 1 = photon)
    truth = truth / 101
    truth.astype(bool)

    # Using RandomForestClassifier with cross validation
    clf = RandomForestClassifier(criterion='entropy', min_samples_split=200,
                                 n_estimators=25)

    result = cross_validate(clf, X=data, y=truth, cv=5, scoring=['accuracy'],
                            return_train_score=True)
    print('Test accuracy:')
    print(result['test_accuracy'])
    print('Mean: ', np.mean(result['test_accuracy']))
    print('')
    print('Train accuracy:')
    print(result['train_accuracy'])
    print('Mean: ', np.mean(result['train_accuracy']))
    print('------------------------------------------------------------------')

    # Fit on data to optain ROC Curve
    X_train, X_test, y_train, y_test = train_test_split(data, truth,
                                                        test_size=0.5)

    clf.fit(X=X_train, y=y_train)
    prediction = clf.predict_proba(X_test)

    fp, tp, _ = roc_curve(y_test, prediction[:, 1])
    roc_auc = auc(fp, tp)

    plt.figure()
    plt.plot(fp, tp, color='orange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.clf()

    # Optain confusion matrix (function from
    # http://scikit-learn.org/stable/modules/generated/
    # sklearn.ensemble.RandomForestClassifier.html)

    prediction2 = clf.predict(X_test)
    conf = confusion_matrix(y_test, prediction2)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    plt.figure()
    plot_confusion_matrix(conf, classes=['gamma', 'proton'],
                          title='Confusion matrix, without normalization')
    plt.show()

    # Energy reconstruction
    truth2 = mc["mc_energy"]
    reg = RandomForestRegressor()
    result2 = cross_validate(clf, X=data, y=truth, cv=5,
                             scoring=['neg_mean_squared_error'],
                             return_train_score=True)
    print('Test score:')
    print(result2['test_score'])
    print('Mean: ', np.mean(result2['test_score']))
    print('------------------------------------------------------------------')
    h5.File.close()




if __name__ == '__main__':
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import h5py as h5
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_validate, train_test_split
    from sklearn.metrics import auc, roc_curve, confusion_matrix
    from sklearn.utils import shuffle
    from sklearn.tree import DecisionTreeClassifier
    fun()
