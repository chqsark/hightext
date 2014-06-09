import pullData
import nlp_module
import sys
import classifier

def driver(filename, n):

    X, y, X_hold = pullData.feed_clf(filename)
    y_p = classifier.classify(X, y, X_hold)
    for i in range(len(y_p)):
        print X_hold[i], y_p[i]
    #sentences = pullData.from_db(filename)
    #words, sents = pullData.word_standardize()
    #ngrams = nlp_module.ngram_collocation(words, sents, n)


if __name__=="__main__":
    #filename = sys.argv[1]
    #n = sys.argv[2]
    driver('../data/palazzo_LasVegas.csv', 4)
