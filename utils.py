from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
def beautiprint(title,list, toFile=None):
    print(title, file=toFile)
    print("#######################", file=toFile)
    print("", file=toFile)
    for l in list:        
        print(l, file=toFile)
    print("", file=toFile)   
    
    
#Unused ngrams method
def getObservedValue(self,n,ngram):
    nbNgrams = self.getNbNgrams(n)
    return self.ngrams[n][ngram][0]/nbNgrams
    
def getEstimatedValue(self,n,ngram):
    nbNgrams = self.getNbNgrams(n)
    tokens = self.tokenize(ngram)
    ps = [self.getValueSimilarKey(self.ngrams[n],t)/nbNgrams if not self.isStopword(t) else 1 for t in tokens]        
    prod = 1
    for p in ps:
        prod *= p  
    return prod
    
def getNbNgrams(self,n):
    res = 0
    for f in self.ngrams[n].values():
        res += f[0]
    return res
def getDice(self,n,ngram):  
    nbNgrams = self.getNbNgrams(self.ngrams,n)
    tokens = self.tokenize(ngram)
    ps = [self.getValueSimilarKey(self.ngrams[n],t) if not self.isStopword(t) else 0 for t in tokens]     
    sum = 0
    for p in ps:
        sum += p  
    return (self.ngrams[n][ngram][0]*n)/sum
    
def getPoissonStirling(self,n,ngram):  
    pAll = self.getObservedValue(n,ngram)
    ps = self.getEstimatedValue(n,ngram)
    
    return pAll*(math.log(pAll)-math.log(ps)-1)
    
def getZscore(self,dict,n,ngram):
    pAll = self.getObservedValue(n,ngram)
    ps = self.getEstimatedValue(n,ngram)
    
    return (pAll - ps)/math.sqrt(ps)
    
def getTscore(self,n,ngram):
    pAll = self.getObservedValue(n,ngram)
    ps = self.getEstimatedValue(n,ngram)
    
    return (pAll - ps)/math.sqrt(pAll)

def getPMI(self,n,ngram):
    pAll = self.getObservedValue(n,ngram)
    ps = self.getEstimatedValue(n,ngram)
    return math.log(pAll/ps)        