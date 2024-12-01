# Implementing the city-block distance & euclidean distance from scratch.

import numpy as np

def city_block_distance(pt1, pt2):
    return np.sum(np.abs(pt1 - pt2))

def euclidean_distance(pt1, pt2):
    return np.sqrt(np.sum((pt1 - pt2) ** 2))