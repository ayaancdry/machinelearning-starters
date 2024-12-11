from eda import X, y, X_train, X_test, y_train, y_test
from distance_metrics import *
from sklearn.metrics import accuracy_score

class WeightedKNN: 
    def __init__(self, k=None, distance_metric=None):
        ''' 
        k : number of neighbours to consider
        distance_metric : which metric to use
        '''
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        ''' 
        Store the training data
        X : training datapoints
        y : training labels
        '''
        self.X = X
        self.y = y

    def predict(self, X_test, k=None, distance_metric=None):
        k = k or self.k
        distance_metric = distance_metric or self.distance_metric

        y_pred = []
        for x in X_test:
            distances = []
            for i, X_train in enumerate(self.X):
                distance = distance_metric(x, X_train)
                distances.append((distance, self.y[i]))

            distances.sort(key=lambda x: x[0])
            distances = distances[:k]
            
            votes = {}
            for distance, label in distances:
                weight = 1 / (distance + 1e-5)  # Add a small constant to avoid division by zero
                if label not in votes:
                    votes[label] = 0
                votes[label] += weight

            predicted_label = max(votes, key=votes.get)
            y_pred.append(predicted_label)

        return y_pred