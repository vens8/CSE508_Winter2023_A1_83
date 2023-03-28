import pandas as pd
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import math
import seaborn as sns
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")


def tf_icf(train_counts, train_df):
    try:
        tf = train_counts.toarray()
    except AttributeError:
        tf = train_counts

    N = len(train_df)
    icf = np.zeros(tf.shape[1])
    j = 0
    while j < tf.shape[1]:
        n_j = 0
        i = 0
        while i < tf.shape[0]:
            if tf[i, j] > 0:
                n_j += 1
            i += 1
        if n_j == 0:
            icf[j] = -999999
        else:
            icf[j] = math.log(N / n_j)
        j += 1

    tf_icf = np.zeros(tf.shape)
    i = 0
    while i < tf.shape[0]:
        j = 0
        while j < tf.shape[1]:
            tf_icf[i, j] = tf[i, j] * icf[j]
            j += 1
        i += 1

    return tf_icf


def to_lower_case(df):
    df['Text'] = df['Text'].map(str.lower)
    return df


def remove_punctuation(df):
    df["Text"] = df['Text'].str.replace('[^\w\s]', '', regex=True)
    return df


def remove_stopwords(df):
    df['Text'].apply(lambda x: [item for item in x if item not in stopwords])
    return df


def tokenize_text(df):
    df['Text'] = df['Text'].apply(word_tokenize)
    return df


def lemmatize_text(df):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_text = []
    for text in df['Text']:
        lemmatized_words = [lemmatizer.lemmatize(w) for w in text]
        lemmatized_text.append(lemmatized_words)
    df['Text'] = lemmatized_text
    return df


def join_tokens(df):
    df['Text'] = df['Text'].apply(lambda x: ' '.join(x))
    return df


def preProcessData(df):
    df = to_lower_case(df)
    df = remove_punctuation(df)
    df = remove_stopwords(df)
    df = tokenize_text(df)
    df = lemmatize_text(df)
    df = join_tokens(df)
    return df


df = pd.read_csv('BBC News Train.csv')
df = preProcessData(df)
print(df)


def train_test_split_data(df, test_size, random_state):
    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df
    except Exception as e:
        print(f"An error occurred during train test split: {e}")
        return None, None


def create_tf_icf(train_counts, train_df):
    try:
        tficf = tf_icf(train_counts, train_df)
        return tficf
    except Exception as e:
        print(f"An error occurred during tf_icf creation: {e}")
        return None


def fit_naive_bayes(train_tf_icf, train_category):
    try:
        nb = MultinomialNB().fit(train_tf_icf, train_category)
        return nb
    except Exception as e:
        print(f"An error occurred during Naive Bayes fit: {e}")
        return None


def predict(test_counts, test_df, nb):
    try:
        test_tf_icf = tf_icf(test_counts, test_df)
        predicted = nb.predict(test_tf_icf)
        return predicted
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None


def print_metrics(test_category, predicted):
    try:
        accuracy = accuracy_score(test_category, predicted)
        print("Accuracy:", accuracy)
        print("Prediction results for 70:30 split")
        print(classification_report(test_category, predicted))
        confusion = confusion_matrix(test_category, predicted)
        print('Confusion matrix of 70:30 split')
        print(confusion)
        return confusion
    except Exception as e:
        print(f"An error occurred during metric printing: {e}")
        return None


def plot_confusion_matrix(confusion):
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion matrix of 70:30 split')
        plt.show()
    except Exception as e:
        print(f"An error occurred during confusion matrix plotting: {e}")
        return None


def processSplits(size, state):
    train_df, test_df = train_test_split_data(df, size, state)
    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(train_df['Text'])
    train_tf_icf = create_tf_icf(train_counts, train_df)
    nb = fit_naive_bayes(train_tf_icf, train_df['Category'])
    test_counts = count_vect.transform(test_df['Text'])
    predicted = predict(test_counts, test_df, nb)
    if predicted is not None:
        confusion = print_metrics(test_df['Category'], predicted)
        if confusion is not None:
            plot_confusion_matrix(confusion)


# 70:30 split
processSplits(0.3, 11)

# 60:40 split
processSplits(0.4, 11)

# 50:50 split
processSplits(0.5, 11)

# 80:20 split
processSplits(0.2, 11)
