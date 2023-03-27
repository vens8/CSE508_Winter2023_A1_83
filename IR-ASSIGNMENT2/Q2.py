# Importing libaries
import nltk
import tqdm
import string
import os
import glob
import numpy as np
from collections import Counter
from typing import List, Any
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk import tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from prettytable import PrettyTable

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Fetching stopwords
stop_words = set(stopwords.words('english'))

path = r"C:\Users\Rahul Maddula\PycharmProjects\IR\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"
os.chdir(path)

l1 = []


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


# l1 contains the list of list of words in each file.


for file in os.listdir():
    if file.endswith(""):
        filePath = f"{path}\{file}"
        extract(filePath)


# Creating paths from the datapath provided

def get_files_list() -> List[str]:
    return ["sci.med", "sci.space", "talk.politics.misc", 'comp.graphics', "rec.sport.hockey", ]


def get_folder_path() -> str:
    return "20_newsgroups"


def get_file_paths(data_path: str) -> List[str]:
    return glob.glob(data_path)


def process_file(file_path: str, files_list: List[str], folder_path: str) -> List[str]:
    try:
        name_of_file, tail_of_file = os.path.split(file_path)
        path_list = []
        i = 0
        while i < len(files_list):
            if tail_of_file == files_list[i]:
                j = 0
                while True:
                    try:
                        file = glob.glob(file_path + "/*")[j]
                    except IndexError:
                        break
                    name_of_file1, tail_of_file1 = os.path.split(file)
                    path_list.append(str(folder_path) + "/" + tail_of_file + "/" + tail_of_file1)
                    j += 1
            i += 1
        return path_list
    except Exception as e:
        print("An error occurred:", e)


def paths_creation(data_path: str) -> tuple[list[str], list[str | bytes | Any]]:
    files_list = get_files_list()
    folder_path = get_folder_path()
    path_list = []
    LA = []
    files_path = get_file_paths(data_path)
    for f in tqdm(files_path, leave=True, position=0, desc="File Processed Till now"):
        path_list.extend(process_file(f, files_list, folder_path))
        LA.append(os.path.split(f)[-1])
    return path_list, LA


try:
    # Defining directory path
    dirpath = "20_newsgroups/*"

    # Creating path and name of files list
    pathGot, namesGot = paths_creation(dirpath)

    """**Creating Dataframe and Defining other Requisite Utility functions**"""

    # Creating Dataframe
    document_iterator = 0
    documentsList = []
    print("Creating dataframe...")
    i = 0
    while i < len(pathGot):
        try:
            dataText = open(pathGot[i], mode='r', errors='ignore', encoding='UTF8').read().strip()
            tokensCreated = tokenize.RegexpTokenizer(r'\w+').tokenize(dataText)
            dataTokens = preprocessingData(tokensCreated)
            documentsList.append(dataTokens)
            document_iterator += 1
        except Exception as e:
            print("An error occurred while processing file", pathGot[i], ":", e)
        i += 1
    df = pd.DataFrame([documentsList, namesGot]).T
    print("Dataframe Created Successfully")

    # Displaying top 5 rows of dataframe created above
    df.head()
except Exception as e:
    print("An error occurred:", e)

# Creating file classes list
fileClassesList = ['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc']


def load_data():
    # Load data from source
    data = pd.read_csv('data.csv', sep='\t', header=None)
    return data


def split_data(data, ratio):
    # Split data into train and test sets
    random_boolean = np.random.rand(len(data)) < ratio
    train_data, test_data = data[random_boolean], data[~random_boolean]

    return train_data, test_data


def reset_index(train_data, test_data):
    # Reset index of train and test sets
    train_final = train_data.reset_index(drop=True)
    test_final = test_data.reset_index(drop=True)
    class_count = Counter(train_final[1])
    print("Class count: " + str(class_count))
    print()
    return train_final, test_final, class_count


def count_words(train_data):
    # Count number of words in each class
    word_dict = {}
    for i in range(len(train_data)):
        try:
            word_dict[train_data[1][i]] += train_data[0][i]
        except KeyError:
            word_dict[train_data[1][i]] = train_data[0][i]
    return word_dict


def count_unique_words(word_dict):
    # Count number of unique words in each class
    unique_words = set()
    for words in word_dict.values():
        unique_words |= set(words)
    unique_word_list = len(unique_words)
    return unique_words, unique_word_list


def find_class_frequency(word_dict):
    # Count class frequency for each word
    cf_values = {}
    file_classes = ['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc']
    for word in set(sum(word_dict.values(), [])):
        cf_values[word] = [file_class for file_class in file_classes if word in word_dict[file_class]]
    for word, classes in cf_values.items():
        cf_values[word] = len(classes)
    return cf_values


def inverse_class_frequency(w, cf_values):
    # Evaluate inverse class frequency for a word
    try:
        return np.log(len(fileClassesList) / cf_values[w])
    except KeyError:
        return 0


def tf_icf(word_dict, cf_values):
    # Calculate TF-ICF value for each word in each class
    tf_icf_values = {}
    N = 5
    for file_class, words in word_dict.items():
        temp_dict = {}
        count = Counter(words)
        word_count = len(words)
        for word in set(words):
            term_frequency = count[word] / word_count
            inverse_class_frequency = inverse_class_frequency(word, cf_values)
            temp_dict[word] = term_frequency * inverse_class_frequency
        tf_icf_values[file_class] = temp_dict
    return tf_icf_values


def find_top_k_features(tf_icf_values, k):
    # Find top k features using TF-ICF value
    final_features_list = []
    k_features_dict = {}
    for file_class, values in tf_icf_values.items():
        sorted_features = sorted(values, key=values.get, reverse=True)
        k_features = sorted_features[:k]
        k_features_dict[file_class] = k_features
        final_features_list.extend(k_features)
    return k_features_dict, final_features_list


def count_frequencies(class_list, tf_icf_features, k_features):
    class_one = {}
    class_two = {}

    # Loop through each class in the class list
    i = 0
    while i < len(class_list):
        cls = class_list[i]
        # Count the top K features for the current class
        count_top_k_features = Counter(k_features[cls])

        # Loop through each word in the TF-ICF features dictionary
        j = 0
        while j < len(tf_icf_features):
            word = tf_icf_features[j]
            # Store the frequency of the word for the current class
            try:
                class_one[(cls, word)] = count_top_k_features[word]
            except KeyError:
                pass

            # Add the frequency of the word to the total frequency for the current class
            try:
                class_two[cls] += count_top_k_features[word]
            except KeyError:
                class_two[cls] = count_top_k_features[word]

            j += 1
        i += 1

    return class_one, class_two


def frequency(w, l, classOne, classTwo):
    try:
        return classOne[l, w], classTwo[l]
    except:
        return 0, classTwo[l]


# Creating function for Naive-Bayes algorithm driver code
import numpy as np


def nbAlgoFunction(disctintWordsC, classTrainSplit, trainData, testData, allClasses, classOne, classTwo):
    truthValues = []
    predictedValues = []
    iterator = 0
    while iterator < testData.shape[0]:
        try:
            truthValues.append(testData[1][iterator])
            classWordProbability = []
            l_iterator = 0
            while l_iterator < len(allClasses):
                l = allClasses[l_iterator]
                wordProbability = 0
                w_iterator = 0
                while w_iterator < len(testData[0][iterator]):
                    w = testData[0][iterator][w_iterator]
                    try:
                        freq, count = frequency(w, l, classOne, classTwo)
                        pp = (freq + 1) / (count + disctintWordsC)
                        wordProbability += np.log(pp)
                    except KeyError:
                        pass
                    w_iterator += 1
                wordProbability += np.log(classTrainSplit[l] / trainData.shape[0])
                classWordProbability.append(wordProbability)
                l_iterator += 1
            predictedValues.append(allClasses[np.argmax(classWordProbability)])
        except IndexError:
            pass
        iterator += 1

    return truthValues, predictedValues


def accuracy_evaluation(predicted_values, truth_values):
    try:
        value_one = len([1 for i in range(len(predicted_values)) if predicted_values[i] == truth_values[i]])
        return value_one / len(predicted_values)
    except ZeroDivisionError:
        print("Error: Division by zero occurred.")
        return None


def evaluate_confusion_matrix(predicted_values, truth_values, classes):
    try:
        c_matrix = np.zeros((len(classes), len(classes))).astype(int)
        for i in range(len(predicted_values)):
            c_matrix[classes.index(predicted_values[i])][classes.index(truth_values[i])] += 1
        return c_matrix
    except Exception as e:
        print("Error occurred during confusion matrix evaluation:", e)
        return None


def heatmap(conf_mat):
    try:
        sns.heatmap(conf_mat / np.sum(conf_mat), annot=True, fmt='.2%', cmap='Blues')
    except Exception as e:
        print("Error occurred while generating heatmap:", e)


"""Driver Code"""


# Creating Driver function
def helperFunction(dataFiles, ratio):
    trainDataFiles, testData = split_data(dataFiles, ratio)
    trainDataFiles, testData, train_class_split = reset_index(trainDataFiles, testData)
    wordDictionary = count_words(testData)
    distinct_words, distinct_words_c = count_unique_words(wordDictionary)
    CFValues = find_class_frequency(wordDictionary)
    valueTFICF = find_top_k_features(wordDictionary, CFValues)
    valuesFeature = [10, 20, 40, 50, 60, 70]
    i = 0
    while i < len(valuesFeature):
        fValue = valuesFeature[i]
        k_feature, featTFICF = find_top_k_features(valueTFICF, fValue)
        class_f, class_c = count_frequencies(fileClassesList, featTFICF, k_feature)
        truthValues, predictedValues = nbAlgoFunction(distinct_words_c, train_class_split, trainDataFiles, testData,
                                                      fileClassesList, class_f, class_c)
        accuracyEvaluated = accuracy_evaluation(predictedValues, truthValues)
        print("Training data size: " + str(ratio * 100) + " %")
        print("Feature Selected: " + str(fValue))
        print("Accuracy Achieved: " + str("{:.2f}".format(accuracyEvaluated * 100)) + " %")
        confusionMatrixEvaluated = evaluate_confusion_matrix(predictedValues, truthValues, fileClassesList)
        print("Confusion Matrix: ")
        print(confusionMatrixEvaluated)
        print()
        featureList.append(fValue)
        listPref.append(accuracyEvaluated)
        listTrainSize.append(ratio)
        i += 1


# Defining features, train dataset and ratio/proportion list
featureList = []
listPref = []
listTrainSize = []
proportionUsed = [0.5, 0.7, 0.8]
s = 1
for currentRatio in proportionUsed:
    print("Case " + str(s))
    helperFunction(df, currentRatio)
    print("----------------------------------")
    print("----------------------------------")
    print()
    s += 1

myTable = PrettyTable()
myTable.field_names = ["Training Data Proportion",
                       "Features Selected/Class (On basis of TF-ICF) ",
                       "Accuracy Achieved"]

i = 0
while i < len(listTrainSize):
    try:
        myTable.add_row([str(round(listTrainSize[i] * 100)) + " %",
                         str(round(featureList[i])),
                         str("{:.2f}".format(listPref[i] * 100)) + " %"])
        i += 1
    except IndexError:
        print("IndexError: Out of range for list index. Please check the input lists.")
        break

print(myTable)

# Plotting the graphs

# Import required libraries
import matplotlib.pyplot as plt
import numpy as np

# Define train:test ratios and accuracy values
train_test_ratios = [0.5, 0.7, 0.8]
accuracies = [listPref[0:4], listPref[4:8], listPref[8:12]]

# Create subplots for each train:test ratio
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Loop through each subplot and plot accuracy values
for i, ax in enumerate(axs):
    ax.plot(featureList[i * 4:(i + 1) * 4], accuracies[i], color='cyan', linewidth=4,
            marker=(5, 1), markerfacecolor='black', markersize=12)
    ax.set_title("Number of features selected vs accuracy with split ratio of " + str(train_test_ratios[i]) + ":"
                 + str(1 - train_test_ratios[i]))
    ax.set_xlabel("Feature count")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([min(accuracies[i]) - 0.1, max(accuracies[i]) + 0.1])
    ax.text(0.05, 0.95, 'Accuracy values:\n' + str([round(x * 100, 2) for x in accuracies[i]]),
            transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Create plot for accuracy vs training data size
plt.figure(figsize=(8, 6))
plt.plot(listTrainSize[3::4], [x * 100 for x in listPref[3::4]], color='red', linewidth=4,
         marker=(5, 1), markerfacecolor='white', markersize=12)
plt.title("Accuracy vs training data size graph")
plt.xlabel("Proportion of training data")
plt.ylabel("Accuracy")
plt.ylim([min(listPref) * 100 - 1, max(listPref) * 100 + 1])
plt.text(0.05, 0.95, 'Accuracy values:\n' + str([round(x * 100, 2) for x in listPref[3::4]]),
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Display all plots
plt.tight_layout()
plt.show()
