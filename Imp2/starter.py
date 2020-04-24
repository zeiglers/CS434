import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re

# Importing the dataset
imdb_data = pd.read_csv('IMDB.csv', delimiter=',')


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
    print(train_v)
    print(valid_v)
    print(test_v)

    #temp = clean_text(imdb_data.get("review")[1])
    #print(temp)

