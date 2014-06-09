from itertools import groupby as g

def most_freq_word(L):
    fword = max(set(L), key = L.count)
    cnt = L.count(fword)
    return (fword, cnt)
    #m = [(key, len(list(group))) for key, group in g(L)]
    #return sorted(m, key = lambda x: x[1], reverse=True)[0]


def extend_ngrams(sents_map, pivot_word, direction, support, confidence):
    ngramExt = []
    while True: 
        ExtWordList = []
	for sentence in sents_map:
	    if pivot_word in sentence:
                extIdx = sentence.index(pivot_word)
                #print "extIdx:", extIdx
		if extIdx+direction>=0 and extIdx+direction<len(sentence):
		    ExtWordList.append(sentence[extIdx+direction])
        #print "==="
        #print ExtWordList
        if len(ExtWordList)>0:
            t = most_freq_word(ExtWordList)
            #print t
	    if t[1] > support and (1.0*t[1])/len(sents_map)>confidence:
                if t[0] in ngramExt: break
	        ngramExt.append(t[0])
	    	pivot_word = t[0]
         
	    else: break
	else:
            break			
	
    return ngramExt
