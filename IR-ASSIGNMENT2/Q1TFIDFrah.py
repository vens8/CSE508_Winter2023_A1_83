import os
import operator

import numpy as np
# Preprocessing
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

path = r"C:\Users\91911\Desktop\IR-ASSIGNMENT2\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"
os.chdir(path)

l1 = []

myFile = open('output.txt', 'w')
def remove_stop(s):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(s)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return remove_punc(filtered_text)


def remove_punc(s):
    for i in range(len(s)):
        s[i] = s[i].translate(str.maketrans('', '', string.punctuation))
    return remove_space(s)


def remove_space(s):
    for i in range(len(s)):
        s[i] = " ".join(s[i].split())
    return s


def lowercase(s):
    return remove_stop(s.lower())





def extract(filePath):
    with open(filePath, 'r+') as myFile:
        fin = myFile.read()
        l2 = fin.split()
        l1.append(l2)

#l1 contains the list of list of words in each file.


for file in os.listdir():
    if file.endswith(""):
        filePath = f"{path}\{file}"
        extract(filePath)


import numpy as np

def calculate_idf():
    word_doc_frequency = {}
    for doc in l1:
        for word in list(set(doc)):
            word_doc_frequency[word] = word_doc_frequency.get(word, 0) + 1
    num_documents = len(l1)
    inverse_doc_frequency = {}
    for word in word_doc_frequency.keys():
        inverse_frequency = np.log((num_documents) / (word_doc_frequency[word] + 1))
        inverse_doc_frequency[word] = inverse_frequency

    return word_doc_frequency, inverse_doc_frequency


documentFrequency,inverse_doc_freq= calculate_idf()
print("The Document Frequency is \n",documentFrequency)
print("\n")
print("\n")
print("The inverse Document Frequency is \n",inverse_doc_freq)
print("\n")
print("\n")



def calculate_tf():
    tf_values = []
    for doc in l1:
        term_frequency = {}
        for word in doc:
            term_frequency[word] = term_frequency.get(word, 0) + 1
        tf_values.append(term_frequency)

    return tf_values



termFrequency = calculate_tf()
print("The term frequency is \n",termFrequency)
print("\n")
print("\n")



def evaluateTFIDF(frequencyValue):
    tfIdfFinalList = []
    for term in frequencyValue:
        tfIDFDict = {}
        for word, freq in term.items():
            if word in inverse_doc_freq:
                tfIDFDict[word] = freq * inverse_doc_freq[word]
        tfIdfFinalList.append(tfIDFDict)
    return tfIdfFinalList




def evaluateBinaryTf(inputTermFrequency):
    binaryTFList= []
    for singleTerm in inputTermFrequency:
        tf = {}
        for singleWord in singleTerm.keys():
            if singleTerm[singleWord] <= 0:
                tf[singleWord] = 0
            else:
                tf[singleWord] = 1
        binaryTFList.append(tf)
    return binaryTFList



binaryTF=evaluateBinaryTf(termFrequency)
binaryTFIDF=evaluateTFIDF(binaryTF)
print("The TF IDF for Binary Weighting Scheme is \n",binaryTFIDF)
print("\n")
print("\n")

def calculate_raw_tf(term_frequencies):
    raw_tf = term_frequencies
    return raw_tf

rawCountTF = calculate_raw_tf(termFrequency)
rawCountTFIDF = evaluateTFIDF(rawCountTF)
print("The TF IDF for Raw Count Weighting Scheme is \n",rawCountTFIDF)
print("\n")
print("\n")


def evaluateTermFrequency(inputTermFrequency):
    termFrequencyList = []
    for term in inputTermFrequency:
        tf = {}
        values = list(term.values())
        t = sum(values)
        for word in term.keys():
            tf[word] = term.get(word, 0) / t
        termFrequencyList.append(tf)
    return termFrequencyList
#
termFrequencyFinal=evaluateTermFrequency(termFrequency)
termFrequencyFinalTFIDF=evaluateTFIDF(termFrequencyFinal)
print("The TF IDF for Term Frequency Weighting Scheme is \n",termFrequencyFinalTFIDF)
print("\n")
print("\n")


def evaluatelogTF(inputTermFrequency):
    logTF = []
    for term in inputTermFrequency:
        tf = {}
        for word, freq in term.items():
            tf[word] = np.log(1 + freq)
        logTF.append(tf)
    return logTF

logTFFinal=evaluatelogTF(termFrequency)
logTFIDFFinal=evaluateTFIDF(logTFFinal)
print("The TF IDF for Log Normalization Weighting Scheme is \n",logTFIDFFinal)
print("\n")
print("\n")

def evaluateDoubleLogTF(inputTermFrequency):
    doubleLogTFList = []
    for term in inputTermFrequency:
        tf = {}
        tfMax = 0
        for i in term.values():
            tfMax = max(tfMax,i)
        if tfMax <= 0:
            tfMax = 1
        for word in term.keys():
            tf[word]= 0.5 + 0.5*(term.get(word)/tfMax)
        doubleLogTFList.append(tf)
    return doubleLogTFList

doubleLogFinal=evaluateDoubleLogTF(termFrequency)
doubleLogFinalTFIDF = evaluateTFIDF(doubleLogFinal)
print("The TF IDF for Double Normalization Weighting Scheme is \n",doubleLogFinalTFIDF)
print("\n")
print("\n")


def get_top_files(sorted_doc_scores):
    top_doc_ids = list(sorted_doc_scores.keys())[:5]
    top_file_names = ["cranfield" + str(doc_id) for doc_id in top_doc_ids]
    return top_file_names

def calculate_top(tfidf_values, input_query):
    doc_scores = {}
    for i, doc_tfidf in enumerate(tfidf_values, start=1):
        doc_score = sum(doc_tfidf.get(word, 0.0) for word in input_query)
        doc_scores[i] = doc_score
    sortedDict = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))
    print("The score is",sortedDict)
    print("\n")
    return get_top_files(sortedDict)


query = input("Write your query here: ")
print("\n")
qprocessed = lowercase(query)
binaryScore = calculate_top(binaryTFIDF,qprocessed)
print("The Top 5 relevant files from Binary TDIDF is ",binaryScore)
print("\n")
rawscore = calculate_top(rawCountTFIDF,qprocessed)
print("The Top 5 relevant files from RAW TDIDF is ",rawscore)
print("\n")
termfreqscore = calculate_top(termFrequencyFinalTFIDF,qprocessed)
print("The Top 5 relevant files from Term Frequency TDIDF is ",termfreqscore)
print("\n")
logscore = calculate_top(logTFIDFFinal,qprocessed)
print("The Top 5 relevant files from Log Normalization TDIDF is ",logscore)
print("\n")
dlogscore = calculate_top(doubleLogFinalTFIDF,qprocessed)
print("The Top 5 relevant files from Double Normalization TDIDF is ",dlogscore)
print("\n")


