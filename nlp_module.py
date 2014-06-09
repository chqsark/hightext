# this module deals with nlp problems, such as ngram generation, PMI, etc
import nltk
from nltk.probability import FreqDist
from nltk.collocations import AbstractCollocationFinder, BigramCollocationFinder, TrigramCollocationFinder, BigramAssocMeasures, TrigramAssocMeasures

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import math
from util import most_freq_word, extend_ngrams

def print_ngram_sentence_mapping(sents, ngrams):
    for ngram in ngrams:
        print "-------", ngram, "--------"
        for s in sents:
            s = s.lower()
            #if set(gram).issubset(set(word_tokenize(s))):
            if all(word in s for word in ngram):
                print s

def print_ngrams(ngrams):
    for ngram in ngrams:
        print ngram

def ngram_collocation(words, sents, n, support=10, topK=200):

    if n>=4: 
        finder = TrigramCollocationFinder.from_words(words)
        ngram_measures = TrigramAssocMeasures()
        finder.apply_freq_filter(support)
        pmi_ngrams = finder.nbest(ngram_measures.pmi, topK)
        ext_ngrams = NgramCollocationExtender(pmi_ngrams, sents, support/3, 0.3)
        print_ngrams(ext_ngrams)
        return ext_ngrams
        #pmi_ngrams = NgramCollocationFinder(words, 2, lowFreq, topK)
        #the current collocation measure is PMI
    else:
        if n==2:
            finder = BigramCollocationFinder.from_words(words)
            ngram_measures = BigramAssocMeasures()
        if n==3:
            finder = TrigramCollocationFinder.from_words(words)
            ngram_measures = TrigramAssocMeasures()

        finder.apply_freq_filter(support)
        pmi_ngrams = finder.nbest(ngram_measures.pmi, topK)

    print_ngrams(pmi_ngrams)
    return pmi_ngrams

def NgramCollocationExtender(ngrams, sents, support, confidence):

    ext_ngrams = []
    sents_dict = {}
    for gram in ngrams:
        sent_contain = []
        for s in sents:
            #s = [ w.lower() for w in s]
            if all(word in s for word in gram):
                sent_contain.append(s)
        sents_dict[gram] = sent_contain 
    
    for key in sents_dict:
        #print "extending" , key 
        sents_map = sents_dict[key]
        #print "in sentences:", sents_map
	pivot_word = key[0]
	leftExt = extend_ngrams(sents_map, pivot_word, -1, support, confidence)
        #print "=== left ext ==="
	#print leftExt
	pivot_word = key[-1]
	rightExt = extend_ngrams(sents_map, pivot_word, 1, support, confidence)
        #print "=== right ext ==="
	#print rightExt
	ext_ngrams.append(leftExt + list(key) + rightExt)

    return ext_ngrams

def NgramCollocationFinder(words, n, support=10, topK=200): 
    uni_vect = CountVectorizer(ngram_range=(1,1), stop_words=('english'))
    n_vect = CountVectorizer(ngram_range=(n,n), stop_words=('english'))
    X_uni = uni_vect.fit_transform([' '.join(words)])
    X_n = n_vect.fit_transform([' '.join(words)])
    ngrams = zip(n_vect.inverse_transform(X_n)[0], X_n.A[0])
    ngrams = (t for t in ngrams if t[1]>=support)
    unigrams = zip(uni_vect.inverse_transform(X_uni)[0], X_uni.A[0])
    unigrams = (t for t in unigrams if t[1]>=support)
    
    ngram_freq_total = 0
    Ngrams = []
    for t in ngrams:
        ngram_freq_total += t[1]
        Ngrams.append(t)
        #print t

    unigram_freq_total = 0
    Unigrams = []
    for t in unigrams:
        #print t
        unigram_freq_total += t[1]
        Unigrams.append(t)

 
    for i in range(len(Ngrams)):
        I_nominator = math.log((1.0*Ngrams[i][1])/ngram_freq_total, 2)
        count=0
        I_denominator = 0
        for w in Unigrams:
            if count==n:
                count = 0
                break
            if w[0] in Ngrams[i][0]:
                count = count+1
                I_denominator += math.log((1.0*w[1])/unigram_freq_total, 2) 
        Ngrams[i] = Ngrams[i] + (I_nominator-I_denominator,)

    Ngrams = sorted(Ngrams, key=lambda x: x[2], reverse=True)
    
    return Ngrams
