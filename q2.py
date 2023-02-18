import pickle
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

nltk.download('stopwords')
nltk.download("punkt")

path = r"C:\Users\Rahul Maddula\PycharmProjects\IR\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"
os.chdir(path)
unigram_inverted_index = {}
scriptDir = os.path.dirname(__file__)
relPath = "pickles/unigramIndex.pickle"
indexPickle = os.path.join(scriptDir, relPath)
index = 1
zerothInd, firstInd = 0, 1


def read_text_file(file_path, index):
	with open(file_path, 'r', encoding='unicode_escape') as f:
		text_lst = f.read().split()
		for i in text_lst:
			if i in unigram_inverted_index.keys():
				if index not in unigram_inverted_index[i]:
					unigram_inverted_index[i].append(index)
			else:
				unigram_inverted_index[i] = [index]


for file in sorted(os.listdir()):
	file_path = f"{path}/{file}"
	read_text_file(file_path, index)
	index += 1
print(unigram_inverted_index)

with open('unigram_inverted_index', 'wb') as handle:
	pickle.dump(unigram_inverted_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('unigram_inverted_index', 'rb') as handle:
	b = pickle.load(handle)


def AND(documents2, documents1):
	documents = []
	comparisons = 0

	while len(documents1) > 0 and len(documents2) > 0:
		if documents1[zerothInd] > documents2[zerothInd]:
			comparisons = comparisons + 1
			documents2 = documents2[firstInd]
		elif documents1[zerothInd] < documents2[zerothInd]:
			comparisons = comparisons + 1
			documents1 = documents1[firstInd]
		else:
			comparisons = comparisons + 1
			documents.append(documents1[zerothInd])
			documents1 = documents1[firstInd]
			documents2 = documents2[firstInd]

	return documents, comparisons


def OR(documents2, documents1):
	documents, comparisons = [], 0

	while len(documents1) > 0 and len(documents2) > 0:
		if documents2[zerothInd] > documents1[zerothInd]:
			documents.append(documents1[zerothInd])
			documents1 = documents1[firstInd:]
			comparisons += 1

		elif documents2[zerothInd] < documents1[zerothInd]:
			documents.append(documents2[zerothInd])
			documents2 = documents2[firstInd:]
			comparisons += 1

		else:
			documents.append(documents1[zerothInd])
			documents1 = documents1[firstInd:]
			documents2 = documents2[firstInd:]
			comparisons += 1
	while 0 < len(documents1):
		documents.append(documents1[zerothInd])
		documents1 = documents1[firstInd:]
	while zerothInd < len(documents2):
		documents.append(documents2[zerothInd])
		documents2 = documents2[firstInd:]

	return documents, comparisons


def AND_NOT(documents2, documents1):
	common_documents, comparisons = AND(documents2.copy(), documents1.copy())
	documents = []

	while len(documents1) > 0:
		if len(common_documents) == 0:
			documents.append(documents1[zerothInd])
			documents1 = documents1[firstInd:]
		elif documents1[zerothInd] == common_documents[zerothInd]:
			documents1 = documents1[firstInd:]
			common_documents = common_documents[firstInd]
			comparisons = comparisons + 1
		elif documents1[zerothInd] < common_documents[zerothInd]:
			documents.append(documents1[zerothInd])
			documents1 = documents1[firstInd:]
			comparisons = comparisons + 1

	return documents, comparisons


def OR_NOT(documents2, documents1):
	documents_not_having_2 = []
	comparisons_all = 0
	prev = 1
	for a in documents2:
		for x in range(prev, a):
			documents_not_having_2.append(x)
			# comparisons_all = comparisons_all+1
		prev = a + 1
	for x in range(prev, 1401):
		documents_not_having_2.append(x)
	documents_not_having_2 = sorted(documents_not_having_2)

	documents, comparisons_or = OR(documents_not_having_2.copy(), documents1.copy())

	return documents, comparisons_all + comparisons_or


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


n = int(input())
for i in range(n):
	myPhrase = lowercase(input())
	list_q = input().split(",")
	totComps = 0
	if len(myPhrase) == len(list_q) + 1:
		for i, op in enumerate(list_q):
			if i == 0:  # for the first word in the query
				if "OR" is op:
					documents, comparisons = OR(unigram_inverted_index[myPhrase[i + firstInd]].copy(), unigram_inverted_index[myPhrase[i]].copy())
				elif "AND NOT" is op:
					documents, comparisons = AND_NOT(unigram_inverted_index[myPhrase[i + firstInd]].copy(), unigram_inverted_index[myPhrase[i]].copy())
				elif "AND" is op:
					documents, comparisons = AND(unigram_inverted_index[myPhrase[i + firstInd]].copy(), unigram_inverted_index[myPhrase[i]].copy())
					
				elif "OR NOT" is op:
					documents, comparisons = OR_NOT(unigram_inverted_index[myPhrase[i + firstInd]].copy(), unigram_inverted_index[myPhrase[i]].copy())
				else:
					print("Sorry, Invalid Query")

			else:  # for the next query words, check with previous result got in documents
				if "AND NOT" is op:
					documents, comparisons = AND_NOT(unigram_inverted_index[myPhrase[i + 1]].copy(), documents.copy())
				elif "OR NOT" is op:
					documents, comparisons = OR_NOT(unigram_inverted_index[myPhrase[i + 1]].copy(), documents.copy())
				elif "AND" is op:
					documents, comparisons = AND(unigram_inverted_index[myPhrase[i + 1]].copy(), documents.copy())
				elif "OR" is op:
					documents, comparisons = OR(unigram_inverted_index[myPhrase[i + 1]].copy(), documents.copy())
				else:
					print("Sorry, Invalid Query")

			totComps = totComps + comparisons

	print(f"This Query:", " ".join([x for y in zip(myPhrase, list_q) for x in y]))
	print(f"Number of documents retrieved for this query:", len(documents))
	print(f"Names of the documents retrieved for this query:", documents)
	print(f"Number of comparisons required for this query:", totComps)


def save_index(index, file):
	with open(file, 'wb') as f:
		pickle.dump(index, f)


def load_index(file):
	with open(file, 'rb') as f:
		return pickle.load(f)


index = unigram_inverted_index

save_index(index, indexPickle)
loaded_index = load_index(indexPickle)
# print(loaded_index)
