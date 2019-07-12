from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
print(__doc__)



def mainf(features , labels , func, printname):
    RANDOM_STATE = 42
    FIG_SIZE = (10, 7)

    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                        test_size=0.30,
                                                        random_state=RANDOM_STATE)


    # Fit to data and predict using pipelined GNB and PCA.
    unscaled_clf = make_pipeline(PCA(n_components=2), func)
    unscaled_clf.fit(features_train, labels_train)
    pred_test = unscaled_clf.predict(features_test)

    # Fit to data and predict using pipelined scaling, GNB and PCA.
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), func)
    std_clf.fit(features_train, labels_train)
    pred_test_std = std_clf.predict(features_test)

    # Show prediction accuracies in scaled and unscaled data.
    print('\nPrediction accuracy for the normal test dataset with PCA using :',printname)
    print('{:.2%}\n'.format(metrics.accuracy_score(labels_test, pred_test)))
    print('Precision score for the normal test dataset with PCA using :', printname)
    print('{:.2%}\n'.format(metrics.precision_score(labels_test, pred_test)))
    print('recall score for normal test dataset with PCA using :', printname)
    print('{:.2%}\n'.format(metrics.recall_score(labels_test, pred_test)))


    print('\nPrediction accuracy for the standardized test dataset with PCA using :', printname)
    print('{:.2%}\n'.format(metrics.accuracy_score(labels_test, pred_test_std)))
    print('Precision score for standardized test dataset with PCA using :', printname)
    print('{:.2%}\n'.format(metrics.precision_score(labels_test, pred_test_std)))
    print('recall score for standardized test dataset with PCA using :', printname)
    print('{:.2%}\n'.format(metrics.recall_score(labels_test, pred_test_std)))



    from learning import plot_learning_curve

    from sklearn.model_selection import ShuffleSplit
    title = "Learning Curves: " , func
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = func
    plot_learning_curve(estimator, title, features, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)


    # Extract PCA from pipeline
    # pca = unscaled_clf.named_steps['pca']
    # pca_std = std_clf.named_steps['pca']

    # Show first principal componenets
    # print('\nPC 1 without scaling:\n', pca.components_[0])
    # print('\nPC 1 with scaling:\n', pca_std.components_[0])

    # Scale and use PCA on X_train data for visualization.
    # scaler = std_clf.named_steps['standardscaler']
    # X_train_std = pca_std.transform(scaler.transform(features_train))

    # visualize standardized vs. untouched dataset with PCA performed
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)

    # for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    #     print features_train[labels_train == l, 0]
    #     ax1.scatter(features_train[labels_train == l, 0], features_train[labels_train == l, 1],
    #                 color=c,
    #                 label='class %s' % l,
    #                 alpha=0.5,
    #                 marker=m
    #                 )
    #
    # for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    #     ax2.scatter(X_train_std[labels_train == l, 0], X_train_std[labels_train == l, 1],
    #                 color=c,
    #                 label='class %s' % l,
    #                 alpha=0.5,
    #                 marker=m
    #                 )
    #
    # ax1.set_title('Training dataset after PCA')
    # ax2.set_title('Standardized training dataset after PCA')
    #
    # for ax in (ax1, ax2):
    #     ax.set_xlabel('1st principal component')
    #     ax.set_ylabel('2nd principal component')
    #     ax.legend(loc='upper right')
    #     ax.grid()
    #
    # plt.tight_layout()
    #
    # plt.show()