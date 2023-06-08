import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split

from itertools import combinations

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

#----------------------------------------- PREPARE DATA ------------------------------------------

numeric = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical = ['gender', 'hypertension', 'heart_disease', 'diabetes']

def prep_diabetes(diabetes):
    diabetes.age = round(diabetes.age, 0).astype(int)
    diabetes = diabetes.drop(columns='smoking_history')
    diabetes = diabetes[diabetes.bmi != 27.32]
    diabetes = diabetes[diabetes.bmi < diabetes.bmi.quantile(.995)]
    diabetes = diabetes[diabetes.gender != 'Other']
    diabetes['gender_encoded'] = np.where(diabetes.gender == 'Female', 0, 1)
    return diabetes

#-----------------------------------------------------------

def split_data(df, target):
    '''
    Takes in a DataFrame and returns train, validate, and test DataFrames; stratifies on target argument.
    
    Train, Validate, Test split is: 56%, 24%, 20% of input dataset, respectively.
    '''
    # First round of split (train+validate and test)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])

    # Second round of split (train and validate)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    
    return train, validate, test



# ----------------------------------------------- EXPLORE DATA -------------------------------------------------------

def visualize_numerical(train):
    pairwise_combinations = list(combinations(train[numeric].columns, 2))

    plt.figure(figsize=(15, 8))
    x=1
    for i in pairwise_combinations:
        plt.subplot(2,3,x)
        sns.regplot(data=train, x=i[0], y=i[1], scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})
        plt.legend([], [], frameon=False)
        x = x+1
    plt.show()

#-----------------------------------------------------------

def visualize_cat_target(train):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(train[categorical]):
        if col != 'diabetes':
            if train[col].dtype == 'O':
                sns.barplot(ax=axes[i], data=train, x=col, y='diabetes')
            else:
                sns.barplot(ax=axes[i], data=train, x=col, y='diabetes').set_xticklabels(labels=['No','Yes'])

#-----------------------------------------------------------

def visualize_multivariate(train):
    pairwise_combinations = list(combinations(train[numeric].columns, 2))

    plt.figure(figsize=(15, 8))
    x=1
    for i in pairwise_combinations:
        plt.subplot(2,3,x)
        sns.scatterplot(data=train, x=i[0], y=i[1], hue='diabetes', palette=['green', 'red'], alpha=0.1)
        plt.legend([], [], frameon=False)
        x = x+1
    plt.show()

# ----------------------------------------------- STATS TESTS -------------------------------------------------------

def check_multicollinearity(train, list_to_check):
    results = pd.DataFrame()
    for x in range(len(list_to_check)):
        corr, p = stats.pearsonr(train[list_to_check[x][0]], train[list_to_check[x][1]])
        result = pd.DataFrame({'variable_1': [list_to_check[x][0]]
                               ,'variable_2': [list_to_check[x][1]]
                               ,'correlation': [corr]
                               ,'p-value': [p]})
        results = pd.concat([results, result])
    return results.reset_index(drop=True)

#-----------------------------------------------------------

def check_comparison_of_means(train, numeric):
    results = pd.DataFrame()
    for x in train[numeric]:
        stat, pval = stats.levene(train[train.diabetes == 0][x], train[train.diabetes == 1][x])
        if pval < 0.05:
            variance = False
            t, p = stats.ttest_ind(train[train.diabetes == 0][x], train[train.diabetes == 1][x], equal_var=False)
            result = pd.DataFrame({'variable': [x]
                                   ,'equal_variance': [variance]
                                   ,'t_stat': [t]
                                   ,'p_value': [p]})
            results = pd.concat([results, result])
        else:
            variance = True
            t, p = stats.ttest_ind(train[train.diabetes == 0][x], train[train.diabetes == 1][x], equal_var=False)
            result = pd.DataFrame({'variable': [x]
                                   ,'equal_variance': [variance]
                                   ,'t_stat': [t]
                                   ,'p_value': [p]})
            results = pd.concat([results, result])
    return results.sort_values('t_stat').reset_index(drop=True)

#-----------------------------------------------------------

def check_chi_squared(train, categorical):
    results = pd.DataFrame()
    for x in train[categorical]:
        observed = pd.crosstab(train[x], train.diabetes)
        chi2, p, degf, expected = stats.chi2_contingency(observed)
        result = pd.DataFrame({'variable': [x]
                               ,'chi_2': [chi2]
                               ,'p_value': [p]})
        results = pd.concat([results, result])
    return results.sort_values('p_value').reset_index(drop=True)

# ----------------------------------------------- MODELING -------------------------------------------------------

def get_best_model(X_train, y_train, X_validate, y_validate):

    best_models = pd.DataFrame()

    #Decision Tree

    tree_scores = pd.DataFrame({})

    for x in range(1,21):
        dtc = DecisionTreeClassifier(max_depth=x)
        dtc.fit(X_train, y_train)

        score = pd.DataFrame({'model': ['Decision Tree']
                            , 'train': [round(((dtc.score(X_train, y_train))*100), 2)]
                            , 'validate': [round(((dtc.score(X_validate, y_validate))*100), 2)]
                            , 'hyperparameter': ['max_depth']
                            , 'hp_value': [x]
                            , 'difference': [abs(dtc.score(X_train, y_train) - dtc.score(X_validate, y_validate))]})

        tree_scores = pd.concat([tree_scores, score])

    best_tree = tree_scores[tree_scores.difference == tree_scores.difference.min()]

    #Random Forest

    forest_scores = pd.DataFrame({})

    for x in range(1,11):
        rfc = RandomForestClassifier(max_depth=x, min_samples_leaf=(11-x), random_state=123)
        rfc.fit(X_train, y_train)
        score = pd.DataFrame({'model': ['Random Forest']
                            , 'train': [round(((rfc.score(X_train, y_train))*100), 2)]
                            , 'validate': [round(((rfc.score(X_validate, y_validate))*100), 2)]
                            , 'hyperparameter': ['max_depth, min_samples_leaf']
                            , 'hp_value': [f'{x}, {(11-x)}']
                            , 'difference': [abs(rfc.score(X_train, y_train) - rfc.score(X_validate, y_validate))]})

        forest_scores = pd.concat([forest_scores, score])

    best_forest = forest_scores[forest_scores.difference == forest_scores.difference.min()]

    #KNN Neighbors

    knn_scores = pd.DataFrame({})

    for x in range(1,21):
        knn = KNeighborsClassifier(n_neighbors=x)
        knn.fit(X_train, y_train)
        score = pd.DataFrame({'model': ['KNN']
                            , 'train': [round(((knn.score(X_train, y_train))*100), 2)]
                            , 'validate': [round(((knn.score(X_validate, y_validate))*100), 2)]
                            , 'hyperparameter': ['n_neighbors']
                            , 'hp_value': [x]
                            , 'difference': [abs(knn.score(X_train, y_train) - knn.score(X_validate, y_validate))]})

        knn_scores = pd.concat([knn_scores, score])

    best_knn = knn_scores[knn_scores.difference == knn_scores.difference.min()]

    #Logistic Regression

    log_scores = pd.DataFrame({})

    for x in range(-2,4):
        lr = LogisticRegression(C=10**x)
        lr.fit(X_train, y_train)

        score = pd.DataFrame({'model': ['Logistic Regression']
                            , 'train': [round(((lr.score(X_train, y_train))*100), 2)]
                            , 'validate': [round(((lr.score(X_validate, y_validate))*100), 2)]
                            , 'hyperparameter': ['C']
                            , 'hp_value': [10**x]
                            , 'difference': [abs(lr.score(X_train, y_train) - lr.score(X_validate, y_validate))]})

        log_scores = pd.concat([log_scores, score]).reset_index(drop=True)

    best_log = log_scores[log_scores.difference == log_scores.difference.min()]

    best_models = pd.concat([best_models, best_tree, best_forest, best_knn, best_log])

    return best_models.sort_values('difference').reset_index(drop=True)

#-----------------------------------------------------------

def run_best_model(X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier(max_depth=10)
    dtc.fit(X_train, y_train)
    score = dtc.score(X_test, y_test)
    return pd.DataFrame({'model': ['Decision Tree']
                        , 'test': [round((score*100), 2)]
                        , 'hyperparameter': ['max_depth']
                        , 'hp_value': [10]})