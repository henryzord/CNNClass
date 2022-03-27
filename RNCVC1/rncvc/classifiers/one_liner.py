"""
	Simple script for testing the KNN implementation.
"""

import numpy as np
import itertools as it
from sklearn import datasets, cross_validation, neighbors
from collections import Counter


def sq_euclidean(x1, x2):
	return ((x1 - x2) ** 2.).sum()


def KNN(X_train, X_test, y_train, k=3):
	predictions = map(
		lambda x1: Counter(y_train[np.array(
			map(
				lambda x2: sq_euclidean(x1, x2),
				X_train
			)).argsort()[:k]]).most_common()[0][0],
		X_test
	)
	return predictions


if __name__ == '__main__':
	_k = 3

	data = datasets.load_iris()
	X, y = data.data, data.target

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10, random_state=42)
	my_predictions = KNN(X_train, X_test, y_train, k=_k)

	real_knn = neighbors.KNeighborsClassifier(
		n_neighbors=_k,
		weights='uniform',
		algorithm='brute',
		metric='minkowski',
		p=2
	)
	real_knn.fit(X_train, y_train)
	real_predictions = real_knn.predict(X_test)

	_comparisson = it.izip(data.target_names[y_test], data.target_names[real_predictions],
	                       data.target_names[my_predictions])

	print 'real, sklearn and mine:'
	for y, h1, h2 in _comparisson:
		print y, h1, h2
