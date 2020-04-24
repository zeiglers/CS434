import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re

# Importing the dataset
imdb_data = pd.read_csv('IMDB.csv', delimiter=',')
imdb_labels = pd.read_csv('IMDB_labels.csv', delimiter=',')

def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

if __name__ == '__main__':

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        # Tuning for max_df and min_df
        #max_df = 1.0
        #min_df = 1
        # Add limit on max_features
        max_features = 2000
    )

    # fit the vectorizer on the text
    vectorizer.fit(imdb_data['review'])

    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

    # Write vocab to file for testing
    file = open("vocab.txt","w")
    for ele in vocabulary:
        file.write(ele+'\n')
    file.close
    #print(vocabulary)

    # Create BOW vectors for each set of the data
    vector = vectorizer.transform(imdb_data.get("review"))
    vectors = vector.toarray()
    temp = np.split(vectors, [30000, 40000])
    train_v = temp[0]
    valid_v = temp[1]
    test_v = temp[2]

    # Create arrays for training and validation labels
    temp = np.split(imdb_labels.to_numpy(), [30000])
    train_labels = temp[0]
    valid_labels = temp[1]

    # Learning P(y = 0) and P(y = 1) from training labels
    # Learning P(wi|y = [0,1])
    y0_total_words = 0
    y0_word_v = train_v[0]
    y1_total_words = 0
    y1_word_v = train_v[0]
    leng = len(y0_word_v)
    for i in range(leng):
        y0_word_v[i] = 0
        y1_word_v[i] = 0

    num_neg = 0
    num_pos = 0
    leng = len(train_labels)
    for i in range(leng):
        lengt = len(y0_word_v)
        if train_labels[i] == "negative":
            num_neg += 1
            for j in range(lengt):
                y0_total_words += train_v[i][j]
                y0_word_v[j] += train_v[i][j]
        else:
            num_pos += 1
            for j in range(lengt):
                y1_total_words += train_v[i][j]
                y1_word_v[j] += train_v[i][j]
    py0 = num_neg/30000
    py1 = num_pos/30000
    y0_P_word_v = np.divide(y0_word_v, y0_total_words)
    y1_P_word_v = np.divide(y1_word_v, y1_total_words)
    print("P(y = 0):")
    print(py0)
    print("P(y = 1):")
    print(py1)
    print(y0_total_words)
    print(y0_word_v)
    print(y0_P_word_v)
    print(y1_total_words)
    print(y1_word_v)
    print(y1_P_word_v)

