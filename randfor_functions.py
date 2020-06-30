import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn import model_selection, svm, datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, explained_variance_score, r2_score
from sklearn.datasets import make_classification
import datetime as dt
import numpy as np


#Runs a random forest regression model and returns data
def randfor_reg_report(x, y, n_estimators, max_depth, max_features, min_samples_split):
    ind_train, ind_test, dep_train, dep_test = train_test_split(x, y)
    rf_model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth,
                                     max_features = max_features, min_samples_split = min_samples_split)
    rf_model.fit(ind_train, dep_train)
    train_pred = rf_model.predict(ind_train)
    test_pred = rf_model.predict(ind_test)
    exp_train = explained_variance_score(dep_train, train_pred)
    exp_test = explained_variance_score(dep_test, test_pred)
    r_train_score = r2_score(dep_train, train_pred)
    r_test_score = r2_score(dep_test, test_pred)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return exp_train, exp_test, r_train_score, r_test_score, importances, indices


#Runs a random forest classification model and returns data
def randfor_class_report(x, y, n_estimators, max_depth, max_features, min_samples_split):
    ind_train, ind_test, dep_train, dep_test = train_test_split(x, y)
    rf_model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth,
                                     max_features = max_features, min_samples_split = min_samples_split)
    rf_model.fit(ind_train, dep_train)
    train_pred = rf_model.predict(ind_train)
    test_pred = rf_model.predict(ind_test)
    train_score = sklearn.metrics.accuracy_score(dep_train, train_pred)
    test_score = sklearn.metrics.accuracy_score(dep_test, test_pred)
    f1_score = sklearn.metrics.f1_score(dep_test, test_pred)
    cm = confusion_matrix(dep_test, test_pred)
    precision_score = sklearn.metrics.precision_score(dep_test, test_pred)
    recall_score = sklearn.metrics.recall_score(dep_test, test_pred)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return train_score, test_score, f1_score, cm, precision_score, recall_score, importances, indices


#Visualizes a random forest tree
def tree_vis(ind_vars, dep_vars, n_estimators, max_depth, max_features, max_leaf_nodes,
             min_samples_split, name):
    file_name = (name + '.dot')
    pic_name = (name + '.png')
    # Model (can also use single decision tree)
    ind_train, ind_test, dep_train, dep_test = train_test_split(ind_vars, dep_vars)
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features =\
                                   max_features, max_leaf_nodes = max_leaf_nodes,\
                                  min_samples_split = min_samples_split)

    # Train
    model.fit(ind_train, dep_train)
    # Extract single tree
    estimator = model.estimators_[5]

    # Export as dot file
    export_graphviz(estimator, out_file = file_name,
                    feature_names = ind_train.columns,
                    #temp hardcoding. TODO: replace with something non-terrible
                    class_names = 'rightcall',
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    #Added shell=True to get this to work
    call(['dot', '-Tpng', file_name, '-o', pic_name, '-Gdpi=600'], shell=True)

    # Display in jupyter notebook
    from IPython.display import Image
    Image('tree.png')
    print('done!')


# Runs a set of random forest classification models to find optimal model depth
def rand_find_depth(num_tests, ind_vars, dep_var, max_depth, min_samples_split):
    train_list = []
    test_list = []
    f_list = []
    for x in range(max_depth):
        train_score, test_score, f1_score, cm, precision_score, recall_score, importances, indices =\
         randfor_class_report(ind_vars, dep_var, num_tests, x + 2, None, min_samples_split)
        train_list.append(train_score)
        test_list.append(test_score)
        f_list.append(f1_score)
        print('done with test ', x)
    return train_list, test_list, f_list


#Sometimes I just want to know what's happening
def randfor_class_talk(x, y, n_estimators, max_depth, max_features, min_samples_split):
    starttime = dt.datetime.now()
    train_score, test_score, f1_score, cm, precision_score, recall_score, importances, indices =\
    randfor_class_report(x, y, n_estimators, max_depth, max_features, min_samples_split)
    print('train score:', train_score)
    print('test score:', test_score)
    print('f1 score', f1_score)
    print('confusion_matrix:')
    print(cm)
    print('precision score:', precision_score)
    print('recall score:', recall_score)
    print('columns:', len(x.columns))
    print('rows:', len(x))
    print('total data:', len(x) * len(x.columns))
    print('feature importances:')
    for f in range(len(indices)):
        print(x.columns[indices[f]]+':', importances[indices[f]])
    print('elapsed time:', dt.datetime.now()-starttime)

def randfor_reg_talk(x, y, n_estimators, max_depth, max_features, min_samples_split):
    starttime = dt.datetime.now()
    exp_train, exp_test, r_train_score, r_test_score, importances, indices =\
    randfor_reg_report(x, y, n_estimators, max_depth, max_features, min_samples_split)
    print('explained variance (train data)', exp_train)
    print('explained variance (test data)', exp_test)
    print('r^2 score(train data):', r_train_score)
    print('r^2 score (test data):', r_test_score)
    print('feature importances')
    for f in range(len(indices)):
        print(x.columns[indices[f]]+':', importances[indices[f]])
    print('elapsed time:', dt.datetime.now()-starttime)


def test_multiple_samples(samples, ind_names, n_estimators, max_depth, max_features, min_split):
    starttime = dt.datetime.now()
    train_scores = []
    test_scores = []
    f1_scores = []
    feature_weights = {}
    for x in range(len(samples)):
        ind_vars, dep_vars = get_class_vars(samples[x], ind_names)
        train_score, test_score, f1_score, cm, precision_score, recall_score, importances, indices =\
        rf.randfor_class_report(ind_vars, dep_vars, n_estimators, max_depth, max_features, min_split)
        train_scores.append(train_score)
        test_scores.append(test_score)
        f1_scores.append(f1_score)
        for f in range(len(indices)):
            if indices[f] not in feature_weights:
                feature_weights[indices[f]] = [importances[indices[f]]]
            else:
                feature_weights[indices[f]].append(importances[indices[f]])
        print('test complete')
    print('train scores mean, std: ', np.mean(train_scores), ', ', np.std(train_scores))
    print('test scores mean, std: ', np.mean(test_scores), ', ', np.std(test_scores))
    print('f1 scores mean, std: ', np.mean(f1_scores), ', ', np.std(f1_scores))
    print('feature weights mean, std:')
    for f in feature_weights:
        print(ind_vars.columns[indices[f]], ': ', np.mean(feature_weights[f]), ', ',np.std(feature_weights[f]))
    print('elapsed time: ', dt.datetime.now()-starttime)