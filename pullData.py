import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

def feed_clf(filename):
    """
        prepare data for classifier
    	1. 6 subcategories: ['valueScore', 'locationScore', 'sleepScore', 'roomScore', 'cleanScore', 'serviceScore']
        2. find sentences contain ['value', 'location', 'sleep', 'room', 'clean', 'service']
    	3. if the related score is 1,2, put negative label
    	   if the related score is 4,5, put positive label
    	4. train a classifier
    	5. classify all sentences
    """
    df = pd.read_csv(filename, sep='\t')
    df = df[df.language=='en']
    cols = ['title', 'text', 'valueScore', 'locationScore', 'sleepScore', 'roomScore', 'cleanScore', 'serviceScore']
    keywords = ['value', 'location', 'sleep', 'room', 'clean', 'service']
    df = df[cols]
    X=[] 
    y=[]
    X_test = []
    ids=[]
    scores = []
    for i in range(len(df)):
        review = df.iloc[i]
        if any(sc<0 for sc in review[2:]): continue
	sentences = sent_tokenize(review[1])
	for sent in sentences:
            if any(w in sent for w in keywords):
	        for j in range(len(keywords)):
	            if keywords[j] in sent:
	                if review[j+2]<3:
	                    y.append(-1)
	                elif review[j+2]>=4:
	                    y.append(1)
	                else: y.append(0)
	                X.append(sent)
                        ids.append(i)
                        scores.append(review[2:])
                        #y.append(sent_label)	    
	                break
	    else:
                X_test.append(sent)				
	
    #print len(X), len(y)
    #for i in range(len(X)):
    #    print ids[i], X[i], y[i],scores[i] 
	
    return X, y, X_test
	
def from_db(filename):
    df = pd.read_csv(filename, sep='\t')
    df = df[df.language=='en']
    reviews = df.text
    sentences = []
    for review in reviews:
        sentences.extend(sent_tokenize(review)) 

    return sentences
	
	
def from_file(filename):
    
    open_file = open(filename, 'r')
    line = open_file.read()
    sentences = sent_tokenize(nltk.clean_html(line))
    
    return sentences
	
def word_standardize(sentences): 	
    tokens = []
    sentences_st = []

    for sent in sentences:
        tokens.extend(word_tokenize(sent))
        sentences_st.append(word_tokenize(sent))
	
    words = tokens
    
    st = LancasterStemmer()

    words = [w.lower() for w in words]
    words = [w for w in words if not w in stopwords.words('english')]
    words = [w for w in words if not w in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
    st_words = [st.stem(w) for w in words]

    sent_result = []
    for sent in sentences_st:
        sent = [w.lower() for w in sent]
        sent = [w for w in sent if not w in stopwords.words('english')]
        sent = [w for w in sent if not w in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
        sent_result.append(sent)

    return st_words, sent_result
