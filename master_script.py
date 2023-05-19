import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from numpy import percentile, absolute, arange, mean, std
from sklearn import linear_model, model_selection, tree, preprocessing, neighbors, svm, ensemble
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split, ShuffleSplit, cross_val_predict, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import learning_curve
from matplotlib import pyplot
import joblib
import matplotlib.pyplot as plt

# How to run: python3 master_script.py -in final_dataset.csv


##################
# set font sizes #
##################
SMALL_SIZE = 4
MEDIUM_SIZE = 6
BIGGER_SIZE = 8

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# This code reads the final dataset from a CSV file
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input_file", required=True, help="Path to input CSV file")
args = parser.parse_args()
df = pd.read_csv(args.input_file)


# Apply Label Encoding to convert categorical features to numerical values
labelencoder = LabelEncoder()
df['Industry'] = labelencoder.fit_transform(df['Industry'])
df['job_state'] = labelencoder.fit_transform(df['job_state'])
df['Location'] = labelencoder.fit_transform(df['Location'])
df['Type of ownership'] = labelencoder.fit_transform(df['Type of ownership'])
df['company_txt'] = labelencoder.fit_transform(df['company_txt'])
df['job_simp'] = labelencoder.fit_transform(df['job_simp'])



#avg_salary is not only the average of min_salary and max_salary but also the average from Glassdoor prediction.
#Therefore, removing the two features 'min salary exp', and 'max salary exp' as well
df.drop(['max salary exp','min salary exp'], axis=1, inplace=True)


# Extract the feature matrix X (all columns except the target)
X = df.drop(columns=['avg_salary'])

# Extract the target vector y (the 'target' column)
y = df['avg_salary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=13)


## Feature Selection 

# ## Linear Regression 

# Create linear regression model
lr_regressor = LinearRegression()

# Create RFE object with linear regression model and number of features to select
rfe = RFE(estimator=lr_regressor, n_features_to_select=14)

# Fit RFE to training data
rfe.fit(X_train, y_train)

# Get feature rankings
ranks = rfe.ranking_

# Get selected feature indices
selected_features = rfe.get_support(indices=True)

# Get names of selected features
selected_feature_names = X_train.columns[selected_features]

# Select the top features from the training and testing sets
X_train_rfe = X_train.iloc[:, selected_features]
X_test_rfe = X_test.iloc[:, selected_features]

# Train model on reduced feature set
lr_regressor.fit(X_train_rfe, y_train)

# Make predictions on test data using reduced feature set
y_pred = lr_regressor.predict(X_test_rfe)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

score=lr_regressor.score(X_test_rfe,y_test)

# Print mean squared error and selected feature names
print("Linear Regression:")
print("Mean Squared Error:", mse)
print("Selected feature names:", selected_feature_names)
print(f"R-squared: {score:.2f}")


# ## Lasso 

# Create Lasso regression model with alpha value
lasso_regressor = Lasso(alpha=0.1)

# Create RFE object with Lasso regression model and number of features to select
rfe = RFE(estimator=lasso_regressor, n_features_to_select=14)

# Fit RFE to training data
rfe.fit(X_train, y_train)

# Get feature rankings
ranks = rfe.ranking_

# Get selected feature indices
selected_features = rfe.get_support(indices=True)

# Get names of selected features
selected_feature_names = X_train.columns[selected_features]

# Select the top features from the training and testing sets
X_train_rfe = X_train.iloc[:, selected_features]
X_test_rfe = X_test.iloc[:, selected_features]

# Train model on reduced feature set
lasso_regressor.fit(X_train_rfe, y_train)

# Make predictions on test data using reduced feature set
y_pred = lasso_regressor.predict(X_test_rfe)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

score=lasso_regressor.score(X_test_rfe,y_test)

# Print mean squared error and selected feature names
print("LASSO Regression:")
print("Mean Squared Error:", mse)
print("Selected feature names:", selected_feature_names)
print(f"R-squared: {score:.2f}")

# ## Ridge

# Create Ridge regression model with alpha value
ridge_regressor = Ridge(alpha=0.1)

# Create RFE object with Ridge regression model and number of features to select
rfe = RFE(estimator=ridge_regressor, n_features_to_select=14)

# Fit RFE to training data
rfe.fit(X_train, y_train)

# Get feature rankings
ranks = rfe.ranking_

# Get selected feature indices
selected_features = rfe.get_support(indices=True)

# Get names of selected features
selected_feature_names = X_train.columns[selected_features]

# Select the top features from the training and testing sets
X_train_rfe = X_train.iloc[:, selected_features]
X_test_rfe = X_test.iloc[:, selected_features]

# Train model on reduced feature set
ridge_regressor.fit(X_train_rfe, y_train)

# Make predictions on test data using reduced feature set
y_pred = ridge_regressor.predict(X_test_rfe)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
score = ridge_regressor.score(X_test_rfe, y_test)

# Print mean squared error and selected feature names
print("Ridge Regression:")
print("Mean Squared Error:", mse)
print("Selected feature names:", selected_feature_names)
print(f"R-squared: {score:.2f}")


# ## Decision Tree regressor 

# Create decision tree regressor
dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)

# Create RFE object with decision tree regressor and number of features to select
rfe = RFE(estimator=dt_regressor, n_features_to_select=6)

# Fit RFE to training data
rfe.fit(X_train, y_train)

# Get feature rankings
ranks = rfe.ranking_

# Get selected feature indices
selected_features = rfe.get_support(indices=True)

# Get names of selected features
selected_feature_names = X_train.columns[selected_features]

# Select the top features from the training and testing sets
X_train_rfe = X_train.iloc[:, selected_features]
X_test_rfe = X_test.iloc[:, selected_features]

# Train model on reduced feature set
dt_regressor.fit(X_train_rfe, y_train)

# Make predictions on test data using reduced feature set
y_pred = dt_regressor.predict(X_test_rfe)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

score=rfe.score(X_test,y_test)

# Print mean squared error and selected feature names
print("Decision Tree Regressor:")
print("Mean Squared Error:", mse)
print("Selected feature names:", selected_feature_names)
print(f"R-squared: {score:.2f}")

# Final dataset
df = df[[ 'Type of ownership', 'hourly', 'job_state', 'python_yn', 'num_comp', 'higher education?', 'min_exp', 'senior','spark', 'aws','excel','R_yn','jr','job_simp','avg_salary']]

df.to_csv('generalized_data.csv', index=False)

## Regression Models 
 
method_name = {
    "adaboost": "AdaBoost Regression",
    "dtr": "Decision Tree Regression",
    "gbr": "Gradient Boosting Regression",
    "lasso": "Least Absolute Shrinkage and Selection Operator (LASSO) Regression",
    "linear": "Linear Regression",
    "rf": "Random Forest",
    "ridge": "Linear Least Squares with l2 Regularization (Ridge) Regression"
}

scoring = { 'MAE':'neg_mean_absolute_error',
            'MSE':'neg_mean_squared_error',
            'RMSE':'neg_root_mean_squared_error',
            'R2':'r2',
            'Explained variance':'explained_variance',
            'Max Error':'max_error' # multi-output not supported
        }

# Set regressor model
# Return: regressor and method name (for printing purposes)
def set_regressor(method):
    if (method == "adaboost"):
        regressor = AdaBoostRegressor(random_state=0, n_estimators=100)

    elif (method == "dtr"):
        regressor = tree.DecisionTreeRegressor()

    elif (method == "gbr"):
        regressor = GradientBoostingRegressor(n_estimators=100)

    elif (method == "lasso"):
        regressor = linear_model.Lasso(alpha=0.1)

    elif (method == "linear"):
        regressor = LinearRegression()

    elif (method == "rf"):
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    elif (method == "ridge"):
        regressor = linear_model.Ridge(alpha=.5)

    else:
        print("\nError: Invalid method name:" + method + "\n")
        parser.print_help()
        sys.exit(0)
    return regressor, method_name[method]

# Evaluate regressor using K-fold cross validation
def eval_model(regressor, num_sp, num_rep):
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    # n_split = 10 and n_repeat = 3
    kfold = RepeatedKFold(n_splits=num_sp, n_repeats=num_rep, random_state=1)

    num_characters = 20
    print("Model".ljust(num_characters),":", method_name)
    print("K-folds".ljust(num_characters),":", kf)
    print("Num splits".ljust(num_characters),":", num_rep)

    for name,score in scoring.items():
        results = model_selection.cross_val_score(regressor, X, y, cv=kfold, scoring=score, n_jobs=-1)
        print(name.ljust(num_characters), ": %.3f (%.3f)" % (np.absolute(results.mean()), np.absolute(results.std())))

# Plot predicted values against true values
def plot_predictions(regressor, num_sp):
    predicted = cross_val_predict(regressor, X, y, cv=num_sp, n_jobs=-1)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

# Assumptions:  last column in the file represents the predictor/dependent variable
#               all data is numeric and has been properly pre-processed
# Return: X and y vectors
def read_data():

    # separate input and output variables
    varray  = df.values
    nc      = len(varray[0,:])-1
    X       = varray[:,0:nc]
    y       = varray[:,nc]
    return X, y

# Set method, kf and number of repeats 
method = "linear"
kf = 5
num_repeats = 3


# set regressor based on user choice
regressor, method_name = set_regressor(method)

# load data from file
X, y = read_data()

# evaluate model
eval_model(regressor, kf, num_repeats)

# plot predicted values
plot_predictions(regressor, kf)

### Learning Curves 
## Set regressor model
# Return regressor and method name (for printing purposes)
def set_regressor(method):
    method_name = {
        "adaboost": "AdaBoost Regression",
        "dtr": "Decision Tree Regression",
        "gbr": "Gradient Boosting Regression",
        "lasso": "LASSO Regression",
        "linear": "Linear Regression",
        "rf": "Random Forest",
        "ridge": "Ridge Regression",
    }

    if (method == "adaboost"):
        regressor = AdaBoostRegressor(random_state=0, n_estimators=100)

    elif (method == "dtr"):
        regressor = tree.DecisionTreeRegressor()

    elif (method == "gbr"):
        regressor = GradientBoostingRegressor(n_estimators=100)

    elif (method == "lasso"):
        regressor = Lasso(alpha=0.1)

    elif (method == "linear"):
        regressor = LinearRegression()

    elif (method == "rf"):
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    elif (method == "ridge"):
        regressor = Ridge(alpha=.5)

    else:
        print("\nError: Invalid method name:" + method + "\n")
        parser.print_help()
        sys.exit(0)
    return regressor, method_name[method]


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(8, 2))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------

# save arguments in separate variables
filename    = df
method1     = "linear" 
method2     = "rf"
num_splits  = 10 

# set two regressors based on user choices
regressor1, method_name1 = set_regressor(method1)
regressor2, method_name2 = set_regressor(method2)

# separate input and output variables
varray  = df.values
nc      = len(varray[0,:])-1
X       = varray[:,0:nc]
y       = varray[:,nc]

fig, axes = plt.subplots(3, 2, figsize=(10, 15))

#############
# First model
#############
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
title = r"Learning Curves " + method_name1
plot_learning_curve(regressor1, title, X, y, axes=axes[:, 0], cv=cv, n_jobs=4)

##############
# Second model
##############
# Cross validation with 100 iterations to get smoother mean test and train
# Score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
title = r"Learning Curves " + method_name2
#estimator = RandomForestRegressor()
plot_learning_curve(regressor2, title, X, y, axes=axes[:, 1], cv=cv, n_jobs=4)

# Save plots in file
#plt.savefig('learning_curves.png')
plt.show()


### Hyper-parameter Optimization 

# Load dataset
dataset = df

# Summarize data #
# -------------- #
print('Data summarization')
print('------------------')
# Shape
print('\nDataset size: ', dataset.shape)

# Head
print('\nFirst 10 lines of data:\n', dataset.head(10))

# Descriptions
print('\nSummary stats of data:\n', dataset.describe())

# Separate data into training/validation and testing datasets
array = dataset.values
X = array[:,:-1]
y = array[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

# Initialize models
models = []
models.append(('LASSO', Lasso()))
models.append(('RIDGE', Ridge()))
models.append(('RF', RandomForestRegressor()))
models.append(('GB', GradientBoostingRegressor()))

# Evaluate models
print('\nModel evaluation - training')
print('--------------------------')
results = []
names = []
for name, model in models:
    kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise')

    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare models based on training results
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison - before optimization')
pyplot.show()

# Improve accuracy with hyper-parameter tuning
print('\nModel evaluation - hyper-parameter tuning')
print('-----------------------------------------')
model_params = dict()
model_params['LASSO'] = dict()
model_params['LASSO']['alpha'] = [0.001, 0.01, 0.1, 1, 10]

model_params['RIDGE'] = dict()
model_params['RIDGE']['alpha'] = [0.001, 0.01, 0.1, 1, 10]

model_params['RF'] = dict()
model_params['RF']['n_estimators'] = [10, 50, 100]
model_params['RF']['max_features'] = ['auto', 'sqrt']

model_params['GB'] = dict()
model_params['GB']['n_estimators'] = [10, 50, 100]
model_params['GB']['learning_rate'] = [0.001, 0.01, 0.1]

best_params = dict()
for name, model in [('LASSO', Lasso()), ('RIDGE', Ridge()), ('RF', RandomForestRegressor()), ('GB', GradientBoostingRegressor())]:
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    rand_search = RandomizedSearchCV(estimator=model, param_distributions=model_params[name], n_iter=5, n_jobs=-1, cv=cv, scoring='neg_mean_squared_error')
    #rand_search = RandomizedSearchCV(estimator=model, param_distributions=model_params[name], n_iter=5, n_jobs=-1, cv=cv, scoring='neg_mean_squared_error')
    rand_result = rand_search.fit(X_train, Y_train)
    print("Model %s -- Best: %f using %s" % (name, rand_result.best_score_, rand_result.best_params_))
    best_params[name] = rand_result.best_params_

# Re-initialize models using best parameter settings
optimized_models = []
optimized_models.append(('LASSO', Lasso(alpha=best_params['LASSO']['alpha'])))
optimized_models.append(('RIDGE', Ridge(alpha=best_params['RIDGE']['alpha'])))
optimized_models.append(('RF', RandomForestRegressor(n_estimators=best_params['RF']['n_estimators'], max_features=best_params['RF']['max_features'])))
optimized_models.append(('GB', GradientBoostingRegressor(n_estimators=best_params['GB']['n_estimators'], learning_rate=best_params['GB']['learning_rate'])))


print('\nModel evaluation - optimized')
print('--------------------------')
results = []
names = []
for name, model in optimized_models:
    kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1, error_score='raise')
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare optimized models based on training results
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison - after optimization')
pyplot.show()

# Fit and save optimized models
for name, model in optimized_models:
    model.fit(X_train, Y_train)
    filename = name + '_optimized_model.sav'
    joblib.dump(model, filename)

# Testing
print('\nModel testing')
print('-------------')
for name, model in optimized_models:
    model.fit(X_train, Y_train)
    predicted_results = model.predict(X_test)
    #mse_results= neg_mean_squared_error(predicted_results, Y_test)
    mape_result = mean_absolute_percentage_error(predicted_results,Y_test)
    print('%s Negative Mean Squared Error: %f' % (name, mape_result))
    pyplot.scatter(Y_test,predicted_results)
    pyplot.title('Test results for ' + name)
    pyplot.xlabel('Ground truth')
    pyplot.ylabel('Predicted results')
    xpoints = ypoints = max(pyplot.xlim(),pyplot.ylim())
    pyplot.plot(xpoints, ypoints, linestyle='--', color='gray', lw=1, scalex=False, scaley=False)
    pyplot.xlim(xpoints)
    pyplot.ylim(ypoints)
    pyplot.show()


