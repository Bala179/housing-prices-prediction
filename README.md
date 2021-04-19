# Predicting Housing Prices
This is my first ML project, which uses the simple 'boston' dataset in sklearn and aims to build a model to predict housing prices from the data in this dataset.

## Loading Data
Data is first loaded using the `load_boston()` method in `sklearn.datasets` and converted to a pandas DataFrame. The contents of the dataset are examined using the `head()` function and it is also checked for missing values (in this case there are none). Then the features and the target variable (in this case, MEDV - the median value of owner-occupied homes in $1000’s) are then separated.

## Training the models
The estimators that I have used have made use of *pipelines*, that is, estimators making use of a number of sub-steps chained together. This makes it possible to use the new, compund, estimator in cross-validation (remember that, if you are scaling the data, you would need to fit the scaler on the *training data* only, this can be done in cross-validation using a pipeline). I tried 3 different models to make predictions - linear regression, polynomial ridge regression and support vector regression. In the last 2 models I used `GridSearchCV` with 5 folds to find the best set of parameters - additionally, I had to perform a train-test split for these 2 cases, so that a portion of unseen data is available for testing the model at the end. Since linear regression has no hyperparameters to tune, I simply used `cross_val_score()` with 5 folds to fit and evaluate the model, without any train-test splitting. (Note: Because of shuffling in the K-Fold cross-validation, the results of the program may vary.)

The scoring metric which I used is R<sup>2</sup>, which should be as close to 1 as possible for a good fit to the data. In the first case, the mean R<sup>2</sup> was evaluated by taking the mean of the 5 cross-validation scores; in the other 2 cases, it was evaluated using the best found parameter(s) on the unseen test data.

# Conclusion
None of the models got a perfect R<sup>2</sup> of 1; however, the best score (among the 3 models) of around 0.85 was found for **polynomial ridge regression with alpha = 3 and degree of polynomial = 2** (in one execution of the program - the result may vary if it is run again).
