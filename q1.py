import os

# Preprocessing
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

path = r"C:\Users\Rahul Maddula\PycharmProjects\IR\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"
os.chdir(path)


def updateTextFile(filePath):
    lines, teline, tiline = [], [], []
    a, b, c, d = 0, 0, 0, 0
    with open(filePath, 'rt', encoding='unicode_escape') as myFile:
        for myline in myFile:  # For each line, read to a string,
            lines.append(myline)
    for i in range(len(lines)):
        if lines[i] == "<TITLE>\n":
            a = i
        if lines[i] == "</TITLE>\n":
            b = i
        if lines[i] == "<TEXT>\n":
            c = i
        if lines[i] == "</TEXT>\n":
            d = i
    for i in range(a + 1, b):
        tiline.append(lines[i])
    for i in range(c + 1, d):
        teline.append(lines[i])

    d1 = ""
    for string in tiline:
        d1 = d1 + string[0:len(string) - 1] + " "
    for string in teline:
        d1 = d1 + string[0:len(string) - 1] + " "

    with open(filePath, 'w+') as myFile:
        myFile.write(d1)
        # print(myFile.read())


for file in os.listdir():
    if file.endswith(""):
        filePath = f"{path}\{file}"
        updateTextFile(filePath)


# nltk.download('stopwords')
# nltk.download('punkt')


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


def preprocessFiles(filePath):
    with open(filePath, 'r+') as myFile:
        fin = lowercase(myFile.read())
    with open(filePath, 'w+') as myFile:
        myFile.write(" ".join(fin))


for file in os.listdir():
    if file.endswith(""):
        filePath = f"{path}\{file}"
        preprocessFiles(filePath)
