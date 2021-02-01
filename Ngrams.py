from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
import numpy as np
import contractions
import nltk
import math
import re
from utils import *
from statistics import mean 
    
##########################        
#ngramsFinder
##########################
        
#ngram cache index
NGRAM_FREQ                 = 0
NGRAM_POS                  = 1
NGRAM_SIZE                 = 2
NGRAM_TOKENS               = 3
NGRAM_CACHE_JACCARD        = 4
NGRAM_CACHE_NORMFREQUENCY  = 5
NGRAM_CACHE_NORMPMI        = 6
NGRAM_CACHE_PMI            = 7
NGRAM_CACHE_FREQUENCY      = 8
NGRAM_CACHE_DICE           = 9
NGRAM_CACHE_NORMDICE       = 10

#class
class NgramsFinder:
    def __init__(self,nMax):
        #Main members
        self.nMAx               = nMax+1
        self.stopwords          = set(stopwords.words('english')) 
        self.ngrams             = [{} for j in range(self.nMAx)]  
        self.superNgram         = {}
        self.nonValidWords      = set()
        self.superGramMaxSize   = 10
        self.freqThreshold      = 0
        
        #caches
        self.supergrams        = []
        self.cleanedText        = ""        
        self.tokenedData        = []       
        self.maxFreq            = [-1 for j in range(self.nMAx)]
        self.maxPMI             = [-1 for j in range(self.nMAx)]
        self.maxDICE            = [-1 for j in range(self.nMAx)]
        self.docSize            = 0
        self.tokenFrequency     = {}
        
        #stopwords
        self.stopwords.remove("own")
        self.stopwords.add("regardless")
        self.stopwords.add("without")
        self.stopwords.add("due")
        self.stopwords.add("thereof")
        self.stopwords.add("please")
        self.stopwords.add("with")
        self.stopwords.add("'s")
        self.stopwords.add("(")
        self.stopwords.add(")")
        self.stopwords.add("[")
        self.stopwords.add("]")
        self.stopwords.add("{")
        self.stopwords.add("}")
        
        #noisy words
        self.nonValidWords.add("fields")
        self.nonValidWords.add("field")
        self.nonValidWords.add("sections")
        self.nonValidWords.add("section")
        self.nonValidWords.add("articles")
        self.nonValidWords.add("article")
        self.nonValidWords.add("table")
        self.nonValidWords.add("annex")
        self.nonValidWords.add("shall")
        self.nonValidWords.add("whether")
        self.nonValidWords.add("subparagraph") 
        self.nonValidWords.add("paragraph")  
        self.nonValidWords.add("where")  
        self.nonValidWords.add("referred")  
        self.nonValidWords.add("within")  
        self.nonValidWords.add("may")  
        self.nonValidWords.add("to")  
        self.nonValidWords.add("is")  
        self.nonValidWords.add("not")  
        self.nonValidWords.add("according")  
        
        #noisy symbols
        self.nonValidWords.add("one")
        self.nonValidWords.add("two")
        self.nonValidWords.add("three")
        self.nonValidWords.add("four")
        self.nonValidWords.add("five")
        self.nonValidWords.add("six")
        self.nonValidWords.add("seven")
        self.nonValidWords.add("eight")
        self.nonValidWords.add("nine")
        self.nonValidWords.add("ten")
        self.nonValidWords.add("''")
        self.nonValidWords.add("``")
        self.nonValidWords.add("`")
        self.nonValidWords.add("(-)")
        self.nonValidWords.add("a.")
        self.nonValidWords.add("b.")
        self.nonValidWords.add("c.")
        self.nonValidWords.add("d.")
        self.nonValidWords.add("e.")
        self.nonValidWords.add("f.")
        self.nonValidWords.add("g.")
        self.nonValidWords.add("h.")
        self.nonValidWords.add("i.")
        self.nonValidWords.add("j.")
        self.nonValidWords.add("k.")
        self.nonValidWords.add("l.")
        self.nonValidWords.add("m.")
        self.nonValidWords.add("n.")
        self.nonValidWords.add("o.")
        self.nonValidWords.add("p.")
        self.nonValidWords.add("q.")
        self.nonValidWords.add("r.")
        self.nonValidWords.add("s.")
        self.nonValidWords.add("t.")
        self.nonValidWords.add("u.")
        self.nonValidWords.add("v.")
        self.nonValidWords.add("w.")
        self.nonValidWords.add("x.")
        self.nonValidWords.add("y.")
        self.nonValidWords.add("z.")
        self.nonValidWords.add("'")
        self.nonValidWords.add("\"")
        self.nonValidWords.add("ii")
        self.nonValidWords.add("iii")
        self.nonValidWords.add("iv")
        self.nonValidWords.add("vi")
        self.nonValidWords.add("vii")
        self.nonValidWords.add("viii")
        self.nonValidWords.add("oj")
        self.nonValidWords.add("ix")
        self.nonValidWords.add("xi")
        self.nonValidWords.add("third")
        self.nonValidWords.add("fourth")      
        self.nonValidWords.add("cr")
        self.nonValidWords.add("equ")      
        self.nonValidWords.add("irb")      
        self.nonValidWords.add("columns")      
        self.nonValidWords.add("column")      
        self.nonValidWords.add("rows")      
        self.nonValidWords.add("row")      
        self.nonValidWords.add("item")      
        self.nonValidWords.add("items")      
        
    #Supergram discovery
    #---------------------------    
    def getNgramOccurenceList(self):
        #create vector posAndNgram containing ngrams sorted by their corresponding position
        #an ngram may appear multiple time at different poisition in the result
        posAndNgram = []                
        for i in range(2,self.nMAx): 
        
            #extract n-grams using custom measure 1
            ngrams = [x[1] for x in self.getSortedNgrams(i,self.getOne,False)]
            #expand n-gram list into occurence position list
            for ngram in ngrams:
                for pos in self.ngrams[i][ngram][NGRAM_POS]:
                    posAndNgram.append([pos, 
                                        ngram, 
                                        self.ngrams[i][ngram][NGRAM_SIZE], 
                                        self.ngrams[i][ngram][NGRAM_TOKENS]])
                    
        return sorted(posAndNgram, key=lambda tup: tup[0])
    
    def getPrunedSuperGram(self,tokenizedSN):
        #prune token
        j = 0
        res = []
        for i,t in enumerate(tokenizedSN):
            if not t in self.stopwords:
                j += 1
            if(j == self.superGramMaxSize): 
                res = tokenizedSN[0:i+1]
                break
        
        #produce string from tokens
        ret = ""
        for r in res:
            ret += r+" "            
        return ret[0:-1]
    
    def getSuperGramSize(self,tokenizedSN):
        #count non-stopwords tokens
        res = 0
        for t in tokenizedSN:
            if not t in self.stopwords:
                res += 1
        return res
    
    def superGramCleaning(self):
        #remove infrequent supergram
        InfrequentSuperNgram = [k for k in self.superNgram if self.superNgram[k][0] < self.freqThreshold]
        for k in InfrequentSuperNgram:
            try:
                del self.superNgram[k]
                continue
            except KeyError:
                pass    
        
        #get pruned SN to add and their old version to delete
        toDel = []
        toAdd = []
        for sn in self.superNgram.keys():
            tokenizedSN = self.tokenize(sn)
            if self.getSuperGramSize(tokenizedSN) > self.superGramMaxSize:                                      
                prunedSn = self.getPrunedSuperGram(tokenizedSN)
                toAdd.append(prunedSn)
                toDel.append(sn)
        
        #add pruned SN to dict
        for k in toAdd:        
            if k in self.superNgram.keys():
                self.superNgram[k][1].extend(self.superNgram[sn][1])
                self.superNgram[k][3].extend(self.superNgram[sn][3])
            else:
                self.superNgram[k] = self.superNgram[sn]
            self.superNgram[k][2] = self.superGramMaxSize
            
        #Remove old version of SN
        for k in toDel:
            check = True
            for a in toAdd:
                if(k == a): 
                    check = False
                    break
            if(check):
                try:
                    del self.superNgram[k]
                    continue
                except KeyError:
                    pass      
                
        #Find supergrams that are subset of others
        toDel = []
        for toTest in self.superNgram.keys():
            for sng in self.superNgram.keys():
                if(toTest != sng and toTest in sng):
                    toDel.append(toTest)
                    
        #Remove supergrams found
        for k in toDel:
            try:
                del self.superNgram[k]
                continue
            except KeyError:
                pass      
    
    def findSuperNgrams(self,tokens):
        posAndNgram = self.getNgramOccurenceList()
        i = 0
        while(i < len(posAndNgram)):

            #create chain of validly linked subElements
            links = []            
            subElements = [list(posAndNgram[i])]
            for j in range(1,len(posAndNgram)-i):
                                
                #find and check link. 
                #If invalid => break 
                validLink, link = self.areValidlyLinked(tokens,subElements[-1],posAndNgram[i+j])
                links.append(link)
                if(not validLink):
                    i += len(subElements)-1
                    break
                                    
                #Check if overlap between two subElements (ngrams). 
                #If overalap remove overlap in current last subElements
                overlap = subElements[-1][0]+subElements[-1][2]-posAndNgram[i+j][0]
                if(overlap > 0):
                    tmp = subElements[-1][3]
                    tmp = tmp[0:-overlap]
                    res = ""
                    for t in tmp:
                        if t != "":
                            res += t+" "
                    subElements[-1][1] = res[0:-1]
                    
                #finally append subElements
                subElements.append(list(posAndNgram[i+j]))
                
                
            i+=1
            #if valid add super-ngrams to list
            if(len([s[1] for s in subElements if s[1] != ""]) > 1):
                superNgram, size, _ = self.fuse([subElements[j][1] for j in range(len(subElements))],links)   
                tokenizedSN = self.tokenize(superNgram)
                size = self.getSuperGramSize(tokenizedSN)                 
                if(size > self.nMAx-2):
                    pos = subElements[0][0]                
                    #add new supergram to list or increment existing one
                    if superNgram in self.superNgram.keys() and pos not in self.superNgram[superNgram][1]:                    
                        self.superNgram[superNgram][1].append(pos)
                    else:
                        self.superNgram[superNgram] = [self.cleanedText.count(superNgram),[pos],size,tokenizedSN]
                    
        #clean set of supergram
        self.superGramCleaning()
    
    #Ngram discovery
    #---------------------------   
    def feedText(self,text):        
        #empty caches
        self.top        = [[] for j in range(self.nMAx)]
        self.maxFreq    = [-1 for j in range(self.nMAx)]
        self.maxPMI     = [-1 for j in range(self.nMAx)]
        self.maxDICE    = [-1 for j in range(self.nMAx)]
        for n in range(len(self.ngrams)):
            for ngram in self.ngrams[n]:
                self.ngrams[n][ngram][NGRAM_CACHE_JACCARD] = -1
                self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY] = -1
                self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI] = -1
                self.ngrams[n][ngram][NGRAM_CACHE_PMI] = -1
                self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY] = -1
                self.ngrams[n][ngram][NGRAM_CACHE_DICE] = -1
                self.ngrams[n][ngram][NGRAM_CACHE_NORMDICE] = -1
                
        #init
        tokens              = self.cleanText(text) 
        self.docSize        = len(tokens)
        self.freqThreshold  = (self.docSize//30000) + 2
        print("Doc size : ",self.docSize)
        print("Dynamic frequency treshold choosen => ", self.freqThreshold)
            
        #find ngrams for n 1->self.nMAX
        print("discovering n-grams")
        for i in range(2,self.nMAx):
            wordsVector     = [0 for j in range(i)]
            stopwordsMatrix = [[] for j in range(i)]            
            
            #for each word either add to stopwords list 
            #or shift wordsVector and stopwordsMatrix content 
            #and add new ngram to dict
            for pos,t in enumerate(tokens):
                if t in self.stopwords:
                    stopwordsMatrix[-1].append(t)
                else:
                    self.shift(t,wordsVector,stopwordsMatrix,i)
                    if wordsVector[0] != 0:
                        #check for symbols
                        hasNoSymbols = self.hasNoSymbols(wordsVector)
                        if(hasNoSymbols):
                            for stopwordsVector in stopwordsMatrix:
                                if(not self.hasNoSymbols(wordsVector)):
                                    hasNoSymbols = False
                                    break
                        
                        #add entry if no symbols                    
                        if hasNoSymbols:
                            self.addEntry(wordsVector,stopwordsMatrix,i,pos)
        
        #find super-ngrams
        print("discovering supergrams")
        self.findSuperNgrams(tokens)
                
    def fuse(self,wordsVector,stopwordsMatrix):
        #init
        res = ""
        size = len(wordsVector)
        tokens = []
        
        #fuse wordsVector and stopwordsMatrix properly
        for i,(w,sws) in enumerate(zip(wordsVector,stopwordsMatrix)):            
            #add word
            if(len(w) > 0):
                res += w +" "
                tokens.append(w)
            
            #add stopwords (ignores last column)
            if i < len(wordsVector)-1:
                size += len(sws)
                for sw in sws:
                    if(len(sw) > 0):
                        res += sw+" "
                        tokens.append(sw)
                    
        #remove last space and return
        return res[0:-1],size,tokens
              
    def addEntry(self,wordsVector,stopwordsMatrix,i,pos):        
        #get fused ngram
        ngram, size, tokens = self.fuse(wordsVector,stopwordsMatrix)       
            
        #If exist, increment count and add new position
        if ngram in self.ngrams[i].keys():
            self.ngrams[i][ngram][NGRAM_FREQ] += 1
            self.ngrams[i][ngram][NGRAM_POS].append(pos-size+1)
            
        #Else if new and valid, set count, add position, set size, set tokens, and init caches
        else:
            if(self.isValidTerm(ngram)):
                self.ngrams[i][ngram] = [1,[pos-size+1],size,tokens,-1,-1,-1,-1,-1,-1,-1]
                
    def shift(self,word,wordsVector,stopwordsMatrix,i):        
        #shift all content
        for j in range(i):
            if(j > 0):
                stopwordsMatrix[j-1] = stopwordsMatrix[j]
                wordsVector[j-1]     = wordsVector[j]   
                
        #set last
        stopwordsMatrix[-1] = []
        wordsVector[-1]     = word
    
    #Getter
    #-------------    
    def getPos(self,ngram,dict):
        return dict[ngram][NGRAM_POS]
        
    
    #Get sorted results
    #-------------    
    def getSortedSuperNgrams(self):
        if(self.supergrams):
            return self.supergrams
            
        #get supergrams
        for k in self.superNgram.keys():
            self.superNgram[k][1] = list(set(self.superNgram[k][1]))
        res = [(self.superNgram[k][0],k) for k in self.superNgram if self.superNgram[k][0] >= self.freqThreshold]
            
        #return sorted
        res = sorted(res, key=lambda tup: tup[0], reverse = True)
        
        self.supergrams = [x[1] for x in res]
        return self.supergrams
        
    def getSortedNgrams(self,n, func, filterSuperGram = True): 
        if(filterSuperGram):
            supergrams = self.getSortedSuperNgrams()
        else:
            supergrams = []
            
        #aplly frequency and part of term filter  
        filteredNgrams = []
        for ngram in self.ngrams[n].keys():
            if(self.getFreq(n,ngram) >= self.freqThreshold and not any([ngram in sg for sg in supergrams])):
                filteredNgrams.append(ngram)
        
        #score ngrams
        res = []
        for ngram in filteredNgrams:  
            score = func(n,ngram)    
            res.append((score,ngram))
            
        #return sorted
        return sorted(res, key=lambda tup: tup[0], reverse = True)
    
        
    #Text processing
    #--------
    def cleanText(self,data):  
        #check cache
        if(self.tokenedData):
            return self.tokenedData
            
        #clean
        data = data.lower()
        data = contractions.fix(data)       
        data = data.replace('implementing technical standards with regard to supervisory reporting of institutions according to regulation','')
        data = data.replace('exposuresthe','exposures. the')
        data = data.replace('.','. ')
        data = data.replace(' - ',' |-| ')
        data = data.replace('\r',' | ')
        data = data.replace('\n',' | ')
        
        #cache and return
        self.cleanedText = data        
        self.tokenedData = self.tokenize(data) 
        return self.tokenedData
        
    def tokenize(self,txt):
        #simple tokenization
        tokens = word_tokenize(txt)  
        
        #Fuse bracketed words with their surrouding bracket tokens
        toDel = []
        for i,t in enumerate(tokens):
            if(i > 0 and i < len(tokens)-1):
                if(tokens[i-1] == "(" and tokens[i+1] == ")" or
                    tokens[i-1] == "[" and tokens[i+1] == "]" or
                    tokens[i-1] == "{" and tokens[i+1] == "}"):
                    tokens[i] = tokens[i-1]+tokens[i]+tokens[i+1]
                    toDel.append(i-1)
                    toDel.append(i+1)
                    
                if(tokens[i] == "-"):                    
                    tokens[i] = tokens[i-1]+tokens[i]+tokens[i+1]
                    toDel.append(i-1)
                    toDel.append(i+1)
                    
        #remove bracket tokens
        toDel.reverse()
        for d in toDel:
            del tokens[d]
            
        return tokens
        
    #Checks
    #--------                       
    def areValidlyLinked(self,tokens,elem1,elem2):
        link = tokens[elem1[0]+elem1[2]:elem2[0]]        
        if(not self.hasNoSymbols(link)):
            return False, link
        for l in link:
            if not l in self.stopwords:
                return False, link
        return True, link
        
    def isValidTerm(self,ngram):
        #check if number
        if(any(c.isdigit() for c in ngram)): return False
        
        #check if too small token
        for t in self.tokenize(ngram):
            if(len(t) < 2):
                return False
            
        #check if invalid token
        for w in self.nonValidWords:
            if(w in ngram):
                return False
                
        return True
        
    def hasNoSymbols(self,list):
        symbols = ["’",".",",",";",":","?","!","(",")","[","]","{","}","-","—","_","$","£",
                    "€","|","&","#","%","+","*","/","\\","<",">"]
        for s in symbols:
            for l in list:
                if(s in l and len(l) < 4):
                    return False
        return True
            
    #Ngram Measures
    #--------                
    def getFactStopwords(self,n,ngram):
        tokens = self.ngrams[n][ngram][NGRAM_TOKENS]
        
        #count nb stopwords
        res = 0        
        for t in tokens:
            if t in self.stopwords:
                res += 1
               
        #normalize and return
        return res/len(tokens)      
    def getInvFactStopwords(self,n,ngram):
        tokens = self.ngrams[n][ngram][NGRAM_TOKENS]
        
        #count nb stopwords
        res = 0        
        for t in tokens:
            if t in self.stopwords:
                res += 1
               
        #normalize and return
        return (1-res/len(tokens))  
        
    def getFreq(self,n,ngram): 
        if(self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY]
            
        res = self.cleanedText.count(ngram+" ")
        res += self.cleanedText.count(ngram+".")
        
        self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY] = res
        return self.ngrams[n][ngram][NGRAM_CACHE_FREQUENCY]
        
        
    def getNormalizedFreq(self,n,ngram): 
        if(self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY]
    
        if(self.maxFreq[n] == -1):
            max = 0
            for k in self.ngrams[n]:
                if(self.getFreq(n,k) > max):
                    max = self.getFreq(n,k)
            self.maxFreq[n] = max
        
        self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY] = self.getFreq(n,ngram)/self.maxFreq[n]
        return self.ngrams[n][ngram][NGRAM_CACHE_NORMFREQUENCY]
        
    def getJaccard(self,n,ngram):
        #check cache and return if possible
        if(self.ngrams[n][ngram][NGRAM_CACHE_JACCARD] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_JACCARD]
            
        #init
        tokens = [t for t in self.ngrams[n][ngram][NGRAM_TOKENS] if not t in self.stopwords]
        pAll   = self.getFreq(n,ngram)
        if(pAll == 0):
            return 0
            
        #compute jaccard
        containSubset = set()
        for s in tokens:
            for k in self.ngrams[n]:
                if(s in k):
                    containSubset.add(k)
        ps = 0
        for k in containSubset:
            ps += self.getFreq(n,k)
                
        #cache result and return     
        self.ngrams[n][ngram][NGRAM_CACHE_JACCARD] = (pAll/ps)
        return self.ngrams[n][ngram][NGRAM_CACHE_JACCARD]       
        
    def getCPS2(self,n,ngram):
        f   = self.getNormalizedFreq(n,ngram)
        j   = self.getJaccard(n,ngram)
        isw = self.getInvFactStopwords(n,ngram)             
        c = self.getCvalue(n,ngram)             
        ts = self.getTScore(n,ngram)             
        pmi = self.getNormalizedPMI(n,ngram)            
        return c*pmi*isw*isw
        
    def getCT(self,n,ngram):
        f   = self.getNormalizedFreq(n,ngram)
        j   = self.getJaccard(n,ngram)
        isw = self.getInvFactStopwords(n,ngram)             
        c = self.getCvalue(n,ngram)             
        ts = self.getTScore(n,ngram)             
        pmi = self.getNormalizedPMI(n,ngram)            
        return c*ts
    
    def getCTS2(self,n,ngram):
        f   = self.getNormalizedFreq(n,ngram)
        j   = self.getJaccard(n,ngram)
        isw = self.getInvFactStopwords(n,ngram)             
        c = self.getCvalue(n,ngram)             
        ts = self.getTScore(n,ngram)             
        pmi = self.getNormalizedPMI(n,ngram)            
        return c*ts*isw*isw
        
    def getTokenFrequency(self,n,token):
        if(token in self.tokenFrequency):
            return self.tokenFrequency[token]
            
        if(self.maxFreq[n] == -1):
            max = 0
            for k in self.ngrams[n]:
                if(self.getFreq(n,k) > max):
                    max = self.getFreq(n,k)
            self.maxFreq[n] = max
            
        res = self.cleanedText.count(token)/self.maxFreq[n]
                
        self.tokenFrequency[token] = res
        return self.tokenFrequency[token]
        
    def getNormalizedPMI(self,n,ngram):
        if(self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI]
                
        if(self.maxPMI[n] == -1):
            max = 0
            for k in self.ngrams[n]:
                if(self.getPMI(n,k) > max):
                    max = self.getPMI(n,k)
            self.maxPMI[n] = max
        
        self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI] = self.getPMI(n,ngram)/self.maxPMI[n]
        return self.ngrams[n][ngram][NGRAM_CACHE_NORMPMI]
    
    def getPMI(self,n,ngram):
        if(self.ngrams[n][ngram][NGRAM_CACHE_PMI] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_PMI]
            
        fAll    = self.getNormalizedFreq(n,ngram)
        if(fAll == 0):
            self.ngrams[n][ngram][NGRAM_CACHE_PMI] = 0
            return self.ngrams[n][ngram][NGRAM_CACHE_PMI]
        tokens  = [t for t in self.ngrams[n][ngram][NGRAM_TOKENS] if not t in self.stopwords]
        N       = len(self.tokenedData)    
        fs      = [self.getTokenFrequency(n,t) for t in tokens]
        denum   = 1
        for f in fs:
            denum*=f/N
            
        self.ngrams[n][ngram][NGRAM_CACHE_PMI] = math.log(fAll/(denum*N))
        return self.ngrams[n][ngram][NGRAM_CACHE_PMI]
        
    
    def getNormalizedDice(self,n,ngram):
        if(self.ngrams[n][ngram][NGRAM_CACHE_NORMDICE] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_NORMDICE]
                
        if(self.maxDICE[n] == -1):
            max = 0
            for k in self.ngrams[n]:
                if(self.getDice(n,k) > max):
                    max = self.getDice(n,k)
            self.maxDICE[n] = max
        
        self.ngrams[n][ngram][NGRAM_CACHE_NORMDICE] = self.getDice(n,ngram)/self.maxDICE[n]
        return self.ngrams[n][ngram][NGRAM_CACHE_NORMDICE]
        
    def getDice(self,n,ngram):
        if(self.ngrams[n][ngram][NGRAM_CACHE_DICE] != -1):
            return self.ngrams[n][ngram][NGRAM_CACHE_DICE]
            
        tokens  = [t for t in self.ngrams[n][ngram][NGRAM_TOKENS] if not t in self.stopwords]
        fAll    = self.getNormalizedFreq(n,ngram)
        if(fAll == 0):
            return 0
        N       = len(self.tokenedData)    
        fs      = [self.getTokenFrequency(n,t) for t in tokens]
        denum   = 0
        for f in fs:
            denum+=f   
            
        self.ngrams[n][ngram][NGRAM_CACHE_DICE] = fAll*N/denum
        return self.ngrams[n][ngram][NGRAM_CACHE_DICE]
        
    def getTScore(self,n,ngram):
        tokens  = [t for t in self.ngrams[n][ngram][NGRAM_TOKENS] if not t in self.stopwords]
        fAll    = self.getNormalizedFreq(n,ngram)
        if(fAll == 0):
            return 0
        N       = len(self.tokenedData)    
        fs      = [self.getTokenFrequency(n,t) for t in tokens]
        prod    = 1
        for f in fs:
            prod*=f/N
            
        return (fAll-(prod*N))/math.sqrt(fAll)
        
    def getOne(self,n,ngram):                  
        return 1
    
    def getCvalue(self,n,ngram):                  
        fAll        = self.getNormalizedFreq(n,ngram)     
        superList   =  []
        for i in range(n+1,self.nMAx):
            res = self.getSortedNgrams(i,self.getNormalizedFreq,False)[0:20]#TODO MAKE IT HIGHER 40,60
            for elem in res:
                if(ngram in elem[1]):
                    superList.append(elem[0])
        if(not superList):
            superList.append(0)
        return fAll-mean(superList)