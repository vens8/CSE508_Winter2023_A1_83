import json
import os
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

path = r"C:\Users\91911\Desktop\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"

os.chdir(path)
dictionary = {}

def check(file_path):
    with open(file_path,'r') as myFile:
        qwe = myFile.read()
        x = qwe.split()
        for i in range(len(x)-1):
            op = x[i]+" "+x[i+1]
            if op in dictionary:
                dictionary[op].append(file_path[len(file_path)-13:len(file_path)])
            else:
                dictionary[op] = [file_path[len(file_path)-13:len(file_path)]]






def remove_stop(s):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(s)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

def remove_punc(s):
    translator = str.maketrans('', '', string.punctuation)
    return remove_stop(s.translate(translator))

def remove_space(s):
    return remove_punc(" ".join(s.split()))

def lowercase(s):
    return remove_space(s.lower())

for file in os.listdir():
    if file.endswith(""):
        file_path = f"{path}\{file}"
        check(file_path)

# print(dictionary)
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

    # return string
    return str1


jh = input()
ggf = lowercase(jh)
upi = ggf.split()
kk = list()
for i in range(len(upi)-1):
    ho = upi[i]+" "+upi[i+1]
    kk.append(ho)

for i in range(len(kk)):
    lpu = dictionary.get(kk[i])
    print("Total number of occurences of word "+kk[i]+" is "+str(len(lpu)))
    print("The files having the word was "+listToString(lpu))
    print("")

# geeky_file = open('convert.txt', 'a')
# geeky_file.write((dictionary))
# # geeky_file.close()
#
# with open('fin.pickle','wb') as hand:
#     pickle.dump(dictionary,hand,protocol=pickle.HIGHEST_PROTOCOL)

