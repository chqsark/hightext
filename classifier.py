import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


	
def classify(X, y, X_hold):
    np.random.seed(42)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_all = vectorizer.fit_transform(X)
    X_holdout = vectorizer.transform(X_hold)
    permute = np.random.permutation(len(y))
    X_all = X_all[permute]
    y = np.array(y)
    y = y[permute]
    n = 2*len(y)/3
    X_train = X_all[:n, :]
    y_train = y[:n]
    X_test = X_all[n:,:]
    y_test = y[n:]
    
    clf = RandomForestClassifier(n_estimators=100,
                            criterion='gini', 
                            bootstrap=True,
                            n_jobs=1)
    
    #clf = MultinomialNB(alpha=.01)
    #clf = svm.SVC(gamma=.001, C=100)
    clf.fit(X_train.toarray(), y_train)
    y_pred = clf.predict(X_test.toarray())
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
    print "precision: ", precision[1]
    print "recall: ", recall[1]
    print "f-score: ", f1_score[1]

    X_houldout = X_holdout[:100]
    y_result = clf.predict(X_holdout.toarray())
    return y_result

if __name__=="__main__":
    classify()
