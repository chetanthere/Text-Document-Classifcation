#This programm will classify text documents by Naive Bayes
#Chetan There, Machine Learning Fall 2016
#17/09/2016


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import math
import os
from collections import Counter
import datetime

startdt = datetime.datetime.now()

#Step 1: Preprocessing
#   a.  Read first 500 files from every folder and put in mega class doc
#   b.  Form a vocabulary and calculate length by merging all class docs
#   c.  Form a test folder by adding remaining 500 docs from each folder

#a.
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stopwordslist = (sorted(stopwords.words('english')))

path = 'C:/Users/Chetan There/Google Drive/My Masters/Fall 2016/Machine Learning/20_newsgroups'
classlist = []

for filename in os.listdir(path):
    classlist.append(filename)

classdict = {}
classlen_dict = {}
traindoc_dict = {}

for i in classlist:
    newpath = path + "/" +str(i)
    print(newpath)
    listl = []
    trainl = []
    k = 0

    for filename in os.listdir(newpath):
        f = open(os.path.join(newpath, filename), "r")
        trainl.append(filename)
        doc = f.read()
        f.close()
        doc = doc.lower()
        listl.append(doc)
        k = k + 1
        if(k == 500):
            break

    traindoc_dict[i] = trainl
    tokens = []
    for m in listl:
        tl = tokenizer.tokenize(m)
        tl = [x for x in tl if x not in stopwordslist]
        for n in tl:
            tokens.append(n)

    tokensd = dict(Counter(tokens))
    classdict[i] = tokensd

vocablist = []
for k,v in classdict.items():
    vl = list(v.keys())
    vlen = sum(v.values())
    classlen_dict[k] = vlen
    for n in vl:
        vocablist.append(n)

#b. Vocabulary and length
vacabset = set(vocablist)
vocab = list(vacabset)
vocab_len = (len(vocab))
print("vocab_len",vocab_len)
print("classlen_dict",classlen_dict)

#c. Form test documents list
# Here we will deal with each folder separately to calculate accuracy
# Form a list of remaining docs of folder; calculate results and find correctly classification # for this folder
# Repeat same for all folders and take combine accuracy.

testdoc_dict = {}
correct_class_dict = {}
for i in classlist:
    newpath = path + "/" +str(i)
    print(newpath)
    trainl = traindoc_dict[i]
    listl = []
    testl = []
    correct_class = 0

    #testdoc_class_fin = {}
    for filename in os.listdir(newpath):
        if filename not in trainl:
            f = open(os.path.join(newpath, filename), "r")
            testl.append(filename)
            doc = f.read()
            f.close()
            doc = doc.lower()
            listl.append(doc)

    testdoc_dict[i] = testl
    tokens = []

    for m in listl:
        tl = tokenizer.tokenize(m)
        tl = [x for x in tl if x not in stopwordslist]
        for n in tl:
            tokens.append(n)
        tokensd = {}
        tokensd = dict(Counter(tokens))
        #this is current doc to test of class i
        docprob_dict = {}
        for k in classlist:
            docprob = 1
            doc_log_prob = 0
            wordcount_tot = classlen_dict[k]
            # addding laplace correction
            wordcount_tot = wordcount_tot + vocab_len
            for j in list(tokensd.keys()):
                wordcount_dict = classdict[k]
                try:
                    wordcount = wordcount_dict[j]
                except KeyError:
                    wordcount = 0
                    
                #addding laplace correction
                wordcount = wordcount + 1
                word_prob = wordcount / wordcount_tot

                #make power of # times it occurs in current doc
                word_occur = tokensd[j]
                #word_prob = pow(word_prob,word_occur)
                word_log_prob =  word_occur * math.log10(word_prob)

                #docprob = docprob * word_prob
                doc_log_prob = doc_log_prob + word_log_prob

            #docprob_dict[k] = docprob
            docprob_dict[k] = doc_log_prob

        #find max prob and decide class
        class_res = max(docprob_dict, key=lambda p: docprob_dict[p])

        compwx = []
        #print("class_res",class_res)
        if(class_res == i):
            correct_class = correct_class + 1

    correct_class_dict[i] = correct_class

print("correct_class_dict",correct_class_dict)

csum = sum(correct_class_dict.values())
print("csum",csum)
tsum = 0
for i in classlist:
    ls = testdoc_dict[i]
    tsum = tsum + len(ls)
print("total test docs",tsum)
acc = (csum / tsum) * 100
print("acc",acc)

enddt = datetime.datetime.now()
print("dttime elapsed",enddt - startdt)

