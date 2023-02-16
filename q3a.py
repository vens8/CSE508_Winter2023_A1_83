import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

path = r"C:\Users\Rahul Maddula\PycharmProjects\IR\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"

os.chdir(path)
dictionary = {}
scriptDir = os.path.dirname(__file__)
relPath = "pickles/index.pickle"
indexPickle = os.path.join(scriptDir, relPath)


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


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele + " "

    # return string
    return str1


def save_index(index, file):
    with open(file, 'wb') as f:
        pickle.dump(index, f)


def load_index(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


save_index(dictionary, indexPickle)
loadedDictionary = load_index(indexPickle)
print(loadedDictionary)

jh = input()
ggf = lowercase(jh)
kk = list()
for i in range(len(ggf)-1):
    ho = ggf[i]+" "+ggf[i+1]
    kk.append(ho)

for i in range(len(kk)):
    if kk[i] in dictionary.keys():
        lpu = dictionary[kk[i]]
        if not len(lpu):
            print("Total number of occurences of word "+ kk[i] + " is " + str(len(lpu)))
            print("The files having the word was " + listToString(lpu))
            print("")

# geeky_file = open('convert.txt', 'a')
# geeky_file.write((dictionary))
# # geeky_file.close()
#
# with open('fin.pickle','wb') as hand:
#     pickle.dump(dictionary,hand,protocol=pickle.HIGHEST_PROTOCOL)



