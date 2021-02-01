from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from os.path import isfile, join
from Ngrams import NgramsFinder 
from threading import Thread
from os import listdir
import contractions
import time
import sys
import re

def beautiprint(title,list, toFile=None):
    print(title, file=toFile)
    print("#######################", file=toFile)
    print("", file=toFile)
    for l in list:        
        print(l, file=toFile)
    print("", file=toFile)   
    
for filename in [f for f in listdir("docs/") if isfile(join("docs/", f))]:
    f = open("docs/"+filename, "r", encoding="utf-8")
    f.readline()
    text = f.read()
    f.close()

    finder = NgramsFinder(4)   
    finder.feedText(text)

    print("Ranking...")
    with open("out/measuresexp"+filename, 'w+', encoding='utf-8') as out:
        top = 30
        beautiprint("TScore",[x[1] for x in finder.getSortedNgrams(3,finder.getTScore)[0:top]], toFile=out)
        beautiprint("Dice",[x[1] for x in finder.getSortedNgrams(3,finder.getDice)[0:top]], toFile=out)
        beautiprint("PMI",[(str(x[0]),x[1]) for x in finder.getSortedNgrams(3,finder.getNormalizedPMI)[0:top]], toFile=out)
        beautiprint("Jaccard",[x[1] for x in finder.getSortedNgrams(3,finder.getJaccard)[0:top]], toFile=out)
        beautiprint("C",[x[1] for x in finder.getSortedNgrams(3,finder.getCvalue)[0:top]], toFile=out)
        beautiprint("freq",[x[1] for x in finder.getSortedNgrams(3,finder.getTokenFrequency)[0:top]], toFile=out)
        beautiprint("CPS",[x[1] for x in finder.getSortedNgrams(3,finder.getCPS2)[0:top]], toFile=out)
        beautiprint("CTS",[x[1] for x in finder.getSortedNgrams(3,finder.getCTS2)[0:top]], toFile=out)
        beautiprint("CT",[x[1] for x in finder.getSortedNgrams(3,finder.getCT)[0:top]], toFile=out)
        beautiprint("SD",[x[1] for x in finder.getSortedNgrams(3,finder.getInvFactStopwords)[0:top]], toFile=out)
        beautiprint("SG",finder.getSortedSuperNgrams(), toFile=out)