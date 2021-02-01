from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from os.path import isfile, join
from Ngrams import NgramsFinder 
from threading import Thread
from os import listdir
import contractions
from utils import *
import time
import sys
import re

#---------------------
#TERM EXTRACTION
#---------------------
def termExtraction(finder, lowerTexts):    
    sn = []
    ng3 = []
    ng2 = []   
    ng4 = []
    for ng in finder.getSortedSuperNgrams():
        sn.append(ng)    
    for ng in sorted([x[1] for x in finder.getSortedNgrams(2,finder.getCPS2)], key=lambda tup: finder.ngrams[2][tup][0], reverse = True):
        ng2.append(ng)   
    for ng in sorted([x[1] for x in finder.getSortedNgrams(3,finder.getCPS2)], key=lambda tup: finder.ngrams[3][tup][0], reverse = True):
        ng3.append(ng)   
    for ng in sorted([x[1] for x in finder.getSortedNgrams(4,finder.getCPS2)], key=lambda tup: finder.ngrams[4][tup][0], reverse = True):
        ng4.append(ng)   
    
    
    #Remove subgrams
    sn, ng2, ng3, ng4 = removeSubGram(sn, ng2, ng3, ng4)
    
    #format
    snMix   = formatTermData(1, sn, finder, lowerTexts)
    mix2    = formatTermData(2, ng2, finder, lowerTexts)
    mix3    = formatTermData(3, ng3, finder, lowerTexts)
    mix4    = formatTermData(4, ng4, finder, lowerTexts)
    return snMix, mix2, mix3, mix4
        
def removeSubGram(sns, ng2, ng3, ng4):
    ToDel2 = []
    ToDel3 = []
    ToDel4 = []
    for sn in sns:
        for ng in ng2:
            if(ng in sn):
                ToDel2.append(ng)
        for ng in ng3:
            if(ng in sn):
                ToDel3.append(ng)
        for ng in ng4:
            if(ng in sn):
                ToDel4.append(ng)
    for ng in ToDel2:
        if(ng in ng2):
            ng2.remove(ng)        
    for ng in ToDel3:
        if(ng in ng3):
            ng3.remove(ng)
    for ng in ToDel4:
        if(ng in ng4):
            ng4.remove(ng)
    return sns, ng2, ng3, ng4
        
def formatTermData(n, list, finder, texts):    
    out = []
    for ng in list:        
        if(n == 1):
            out.append(str(finder.superNgram[ng][0]).rjust(5)+' | '+ng.ljust(100)+' | '+str(finder.superNgram[ng][1]))   
        else:
            out.append(str(finder.ngrams[n][ng][0]).rjust(5)+' | '+ng.ljust(100)+' | '+str(finder.ngrams[n][ng][1]))   
    return out
        
        
#---------------------
#ABBREVIATION EXTRACTION
#---------------------
def extractAbbvTerm(tokens,i):
    abbv = tokens[i]
    k = 1
    for j,c in enumerate(abbv[::-1]):
        sw = set(stopwords.words('english')) 
        while(i-j-k >= 0 and tokens[i-j-k][0] != c and (tokens[i-j-k] in sw or tokens[i-j-k][0] == '(')):
            k += 1
        if(i-j-k < 0 or tokens[i-j-k][0] != c):
            return None
    
    res = ""
    for t in tokens[i-len(abbv)-k+1:i]:
        if(t != "("):
            res +=t+" "    
    return res
        
def getPosAbbv(texts, text, list):
    res = []
    
    for abbv in list:
        pos = [m.start() for m in re.finditer(abbv[0], text)]
        res.append(str(len(pos)).rjust(5)+" | "+abbv[0].ljust(10)+" | "+abbv[1].ljust(85)+" | "+str(pos))
    return res    
        
def extractAbbv(tokens):
    sw = set(stopwords.words('english')) 
    res = []    
    for i,t in enumerate(tokens):
        prop = sum(1 for c in t if c.isupper())/len(t)
        if(prop > 0.5 and 
            len(t) < 6 and 
            len(t) > 1 and 
            t.lower() not in sw and 
            sum(1 for c in t if c == 'V' or c=='I') != len(t) and 
            t.isalpha()):
            term = extractAbbvTerm(tokens,i)
            if(term is not None):
                res.append((t,term))
    return list(set(res))

#---------------------
#MAIN FUNCTION
#---------------------
    
def analyzeFiles(files):
    texts = {}    
    for fileName in files:
        f = open("docs/"+fileName, "r", encoding="utf-8")
        texts[fileName] = [f.readline(), f.read()]
        f.close()
        
    lowerTexts = []
    for f in texts:
        lowerTexts.append(texts[f][1].lower())
        
    for fileName in files:
        analyzeFile(fileName,texts,lowerTexts)
    
def analyzeFile(fileName,texts, lowerTexts):
    print("Analyzing : ",fileName, file=sys.stderr)
    
    finder = NgramsFinder(4)   
    finder.feedText(texts[fileName][1])
    snMix, mix2, mix3, mix4 = termExtraction(finder, lowerTexts)
    
    with open("out/out_"+str(fileName), 'w+', encoding='utf-8') as out:
        out.write("METADATA :\n")
        out.write("-------------\n")
        out.write(texts[fileName][0]+"\n")
        out.write("\n")
        
        print("Extracting terms")
        print("-----------------------------", file=out)    
        print("       Term extracted        ", file=out)    
        print("-----------------------------", file=out)    
        print("", file=out)   
        print("", file=out)   
        print("Doc size : ",finder.docSize," words", file=out)  
        print("Dynamic frequency treshold choosen => ", finder.freqThreshold, file=out)
        
        header = "Freq".rjust(5)+' | '+"Term".ljust(100)+' | '+"Position"
        snMix.insert(0,header)
        mix2.insert(0,header)
        mix3.insert(0,header)
        mix4.insert(0,header)
        
        beautiprint("SN",snMix, toFile=out)
        beautiprint("2-grams", mix2, toFile=out)    
        beautiprint("3-grams", mix3, toFile=out)    
        beautiprint("4-grams", mix4, toFile=out) 
        
        print("-----------------------------", file=out)    
        print("         Abbreviation        ", file=out)    
        print("-----------------------------", file=out)    
        abbv = extractAbbv(word_tokenize(texts[fileName][1]))
        abbv = getPosAbbv(texts,texts[fileName][1], abbv)    
        header = "Freq".rjust(5)+' | '+"Abbv".ljust(10)+' | '+"Term".ljust(85)+' | '+"Position"  
        abbv.insert(0,header)   
        beautiprint("Abbv", abbv, toFile=out)         
        
    print("Done : ",fileName, file=sys.stderr)
            
if __name__ == '__main__':     
    analyzeFiles([f for f in listdir("docs/") if isfile(join("docs/", f))])
    