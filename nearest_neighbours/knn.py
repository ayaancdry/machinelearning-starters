# Function for Implementation of KNN from scratch

''' 
Ensure the following things : 
- The training data should be stored.
- Evaluation should be performed on single datapoint as well as multiple data points.
- Take k and distance metric as input during prediction.
'''

''' 
Don't run this py file as it will run eda.ipynb first
'''

from eda import X, y, X_train, X_test, y_train, y_test
from distance_metrics import *
from sklearn.metrics import accuracy_score

class KNN: 
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
                distance = distance_metric(X, X_train)
                distances.append((distance, self.y[i]))

            distances.sort(key=lambda x: x[0])
            distances = distances[:self.k]
            
            votes = {}
            for distance in distances:
                if distance[1] not in votes:
                    votes[distance[1]] = 0
                votes[distance[1]] += 1 

        return y_pred
