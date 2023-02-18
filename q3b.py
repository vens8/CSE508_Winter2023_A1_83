import os
import pickle

path = r"C:\Users\Rahul Maddula\PycharmProjects\IR\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"
os.chdir(path)
scriptDir = os.path.dirname(__file__)
relPath = "pickles/index.pickle"
indexPickle = os.path.join(scriptDir, relPath)


def build_positional_index():
    index = {}
    for file in os.listdir():
        if file.endswith(""):
            file_path = f"{path}\{file}"
            with open(file, 'r') as f:
                line_no = 0
                for line in f:
                    line_no += 1
                    words = line.strip().split()
                    for pos, word in enumerate(words):
                        if word not in index:
                            index[word] = {}
                        if file not in index[word]:
                            index[word][file] = []
                        index[word][file].append((line_no, pos))  # Can remove line number
    return index


def save_index(index, file):
    with open(file, 'wb') as f:
        pickle.dump(index, f)


def load_index(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


index = build_positional_index()
# print(index)


save_index(index, indexPickle)
loaded_index = load_index(indexPickle)
print(loaded_index)
