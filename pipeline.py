#simple pipline for text learning

from __future__ import print_function

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

import pullData

print(__doc__)

logging.basicConfig(level=logging.INFO, format='%s(asctime)s %(levelname)s %message)s')

#filename = "../data/travelodge_Chicago.csv"
filename = "../data/palazzo_LasVegas.csv"
#filename = "../data/hilton_SF.csv"

X_train, y_train, X_hold = pullData.feed_clf(filename)

pipeline = Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	#('clf', svm.SVC()),
	#('clf', MultinomialNB()),
	('clf', SGDClassifier()),
])

parameters = {
	'vect__max_df': (0.5, 0.75, 1.0),
	'vect__max_features': (None, 5000, 10000, 50000),
	'vect__ngram_range': ((1,1), (1,2)),
	'tfidf__use_idf': (True, False),
	'tfidf__norm': ('l1', 'l2'),
	'clf__alpha': (0.00001, 0.000001),
	'clf__penalty': ('l2', 'elasticnet'),
	'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
	
	print("Performing grid search ...")
	print("pipeline:", [name for name, _ in pipeline.steps])
	print("parameters:")
	pprint(parameters)
	t0 = time()
	grid_search.fit(X_train, y_train)
	print("done in %0.3fs" % (time()-t0))
	print()
	
	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param in sorted(parameters.keys()):
		print("\t%s: %r" % (param, best_parameters[param]))
