from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# k-fold cross validation
def k_fold_c_v(X, y, k, model_class, model_args, distance_metric, random_state=42):
    """
    Perform K-Fold Cross-Validation.
    
    Parameters:
        X : Feature matrix.
        y : Target vector.
        k : Number of folds.
        model_class : The model class to use (e.g., KNN).
        model_args : Arguments to initialize the model.
        distance_metric : Distance metric function.
    
    Returns:
        float: Mean accuracy across all folds.
        float: Standard deviation of accuracy across all folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        model = model_class(distance_metric=distance_metric, **model_args)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)




# stratified k-fold cross validation
def k_fold_c_v(X, y, k, model_class, model_args, distance_metric, random_state=42):
    """
    Perform K-Fold Cross-Validation.
    
    Parameters:
        X : Feature matrix.
        y : Target vector.
        k : Number of folds.
        model_class : The model class to use (e.g., KNN).
        model_args : Arguments to initialize the model.
        distance_metric : Distance metric function.
    
    Returns:
        float: Mean accuracy across all folds.
        float: Standard deviation of accuracy across all folds.
    """
    stratified_kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    accuracies = []

    for train_index, test_index in stratified_kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        model = model_class(distance_metric=distance_metric, **model_args)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)
