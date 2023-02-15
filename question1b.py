import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

path = r"C:\Users\91911\Desktop\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"

os.chdir(path)
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

def preprocessfiles(file_path):
    with open(file_path,'r+') as myfile:
        fin = lowercase(myfile.read())
    with open(file_path,'w+') as myfile:
        myfile.write(" ".join(fin))



for file in os.listdir():
    if file.endswith(""):
        file_path = f"{path}\{file}"
        preprocessfiles(file_path)

