#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import make_pipeline






### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

poi="poi"
feature_1="salary"
feature_2="total_stock_value"
feature_3="bonus"
feature_4="exercised_stock_options"
feature_5 ="shared_receipt_with_poi"


features_list = [poi,feature_1,feature_2,feature_3,feature_4, feature_5] # You will need to use more features

nfeature_list = [poi, feature_1,feature_2]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s) !!!
### Store to my_dataset for easy export below.
key='TOTAL'
data_dict.pop( key, 0 )

my_dataset = data_dict



### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys = True)
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)


#######################################this is the pca version####################################################

#naive bayes
from mainfunction import mainf
from sklearn.naive_bayes import GaussianNB
mainf(features,labels, GaussianNB(), "Naive bayes")

#dt
from sklearn import tree
mainf(features,labels, tree.DecisionTreeClassifier(min_samples_split=40), "Decision tree")


# knn
from sklearn.neighbors import KNeighborsClassifier
mainf(features,labels, KNeighborsClassifier(n_neighbors=3), "K- nearest neighbour")

#randomf
from sklearn.ensemble import RandomForestClassifier
mainf(features,labels, RandomForestClassifier(max_depth=2), "RandomForestClassifier")


#adaboost
from sklearn.ensemble import AdaBoostClassifier
mainf(features,labels, AdaBoostClassifier(n_estimators=200,learning_rate=2.0), "Adaboost")

#kmeans
from sklearn.cluster import KMeans
mainf(features,labels, KMeans(n_clusters=2), "kmeans")

##################################################################################################################


ndata = featureFormat(my_dataset, nfeature_list)
nlabels, nfeatures = targetFeatureSplit(ndata)

from sklearn.preprocessing import MinMaxScaler

scalerC = MinMaxScaler()
scaledfeatures = scalerC.fit_transform(nfeatures)
#scaledlabels = scalerC.fit_transform(labels)




import matplotlib.pyplot

for point in scaledfeatures:
    salary = point[0]
    total_stock_value = point[1]
    matplotlib.pyplot.scatter( salary, total_stock_value )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("total_stock_value")
matplotlib.pyplot.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from class_vis import prettyPicture
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(scaledfeatures, nlabels, test_size=0.3, random_state=42)

from learning import plot_learning_curve


# naive bayes
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "naive bayes: accuracy score: ",clf.score(features_test,labels_test)
print "naive bayes: precision score: ",metrics.precision_score(labels_test, pred)
print "naive bayes: recall score: ",metrics.recall_score(labels_test, pred)

from sklearn.model_selection import ShuffleSplit
title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title,nfeatures, nlabels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)



try:
    prettyPicture(clf, features_test, labels_test, feature_1, feature_2, "naive.png")
except NameError:
    pass






# decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "decision tree: ",clf.score(features_test,labels_test)
print "decision tree: precision score: ",metrics.precision_score(labels_test, pred)
print "decision tree: recall score: ",metrics.recall_score(labels_test, pred)

title = "Learning Curves (Decision tree)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = tree.DecisionTreeClassifier()
plot_learning_curve(estimator, title,nfeatures, nlabels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
try:
    prettyPicture(clf, features_test, labels_test, feature_1, feature_2, "dtree.png")
except NameError:
    pass

#svm
# from sklearn.svm import SVC
# clf = SVC(kernel="linear")
# clf.fit(features_train,labels_train)
# print "svm",clf.score(features_test,labels_test)

# knn
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "knn: ",clf.score(features_test,labels_test)
print "knn: precision score: ",metrics.precision_score(labels_test, pred)
print "knn: recall score: ",metrics.recall_score(labels_test, pred)

title = "Learning Curves (KNeighborsClassifier(n_neighbors=3))"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = KNeighborsClassifier(n_neighbors=3)
plot_learning_curve(estimator, title,nfeatures, nlabels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

try:
    prettyPicture(clf, features_test, labels_test, feature_1, feature_2, "knn.png")
except NameError:
    pass

# random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "random forest: ", clf.score(features_test,labels_test)
print "random forest: precision score: ",metrics.precision_score(labels_test, pred)
print "random forest: recall score: ",metrics.recall_score(labels_test, pred)

title = "Learning Curves (RandomForestClassifier(max_depth=5))"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = RandomForestClassifier(max_depth=2)
plot_learning_curve(estimator, title,nfeatures, nlabels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

try:
    prettyPicture(clf, features_test, labels_test, feature_1, feature_2, "randomf.png")
except NameError:
    pass

# adaboost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print " adaboost: ",clf.score(features_test,labels_test)
print "adaboost: precision score: ",metrics.precision_score(labels_test, pred)
print "adaboost: recall score: ",metrics.recall_score(labels_test, pred)

title = "Learning Curves (AdaBoostClassifier(n_estimators=100))"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = AdaBoostClassifier(n_estimators=200,learning_rate=2.0)
plot_learning_curve(estimator, title,nfeatures, nlabels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

try:
    prettyPicture(clf, features_test, labels_test, feature_1, feature_2, "adaboost.png")
except NameError:
    pass


##kmeans
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaledfeatures_features = scaler.fit_transform(features)

from sklearn.cluster import KMeans
from visualizers import Draw
kmeans = KMeans(n_clusters=2, random_state=0).fit(scaledfeatures_features)
pred=kmeans.labels_
try:
    Draw(pred, scaledfeatures_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)